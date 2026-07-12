"""Core benchmarking logic: run all three simulator phases and collect metrics.

Defaults match the reference paper's experimental setup (arXiv:2604.08467,
Sec. IV-C): PTSBE non-final batch_size=10, final_batch_size=28, 1 shot per
non-final batch, 100 hypersamples; Traditional (CUDA-Q-equivalent baseline)
batch_size=24, 1 hypersample. Speedup aggregation uses geometric mean/std
(Sec. IV-D) since throughput spans multiple orders of magnitude.
"""
from __future__ import annotations

import json
import math
import signal
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import gmean, gstd

from .circuit_generator import generate_ensemble
from .timing import Timer, gpu_device_info
from tn_noise_sim.simulators import (
    TraditionalTrajectorySimulator,
    UnoptimizedPTSBESimulator,
    OptimizedPTSBESimulator,
)
from tn_noise_sim.contraction import ContractionPathCache

# Default shots per circuit instance
DEFAULT_SHOTS = 100

# Paper's Sec. IV-C default hyperparameters
PAPER_BATCH_SIZE = 10
PAPER_FINAL_BATCH_SIZE = 28
PAPER_NUM_HYPERSAMPLES = 100
PAPER_TRADITIONAL_BATCH_SIZE = 24


class _Timeout(Exception):
    pass


def _run_with_timeout(fn, timeout_s: Optional[float]):
    """Run fn() under a SIGALRM wall-clock budget; raise _Timeout if exceeded.

    timeout_s=None disables the budget. Unix-only (SIGALRM); acceptable since
    this harness targets Linux GPU servers.
    """
    if not timeout_s:
        return fn()

    def _handler(signum, frame):
        raise _Timeout(f"exceeded {timeout_s}s budget")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _throughput(num_bitstrings: int, elapsed_s: float) -> float:
    """Unique bitstrings per GPU/CPU second."""
    if elapsed_s <= 0:
        return float("inf")
    return num_bitstrings / elapsed_s


def _time_simulator(sim, circuit: dict, noise_model, num_shots: int, use_gpu: bool, **sample_kwargs) -> tuple:
    """Run simulator and return (results, elapsed_seconds)."""
    t = Timer(use_gpu=use_gpu)
    with t:
        results = sim.sample(circuit, noise_model, num_shots=num_shots, **sample_kwargs)
    return results, t.elapsed_s


def _cold_warm_from_engine(sim) -> Dict[str, Optional[float]]:
    """Extract a (path-finding, contraction-per-call) time split from a
    simulator's last-used ContractionEngine.call_log, per design.md D3: the
    first compute_reduced_density_matrix() call for a given batch_index pays
    cuTensorNet's lazy path-finding cost; later calls for the same
    batch_index reuse the cached plan."""
    engine = getattr(sim, "_last_engine", None)
    log = getattr(engine, "call_log", None) if engine is not None else None
    if not log:
        return {"path_finding_time_s": None, "contraction_time_per_call_s": None}
    cold = [elapsed for _, is_first, elapsed in log if is_first]
    warm = [elapsed for _, is_first, elapsed in log if not is_first]
    return {
        "path_finding_time_s": float(np.mean(cold)) if cold else None,
        "contraction_time_per_call_s": float(np.mean(warm)) if warm else (float(np.mean(cold)) if cold else None),
    }


def run_benchmark(
    n: int,
    g: int,
    num_instances: int = 10,
    num_shots: int = DEFAULT_SHOTS,
    num_error_sets: int = 10,
    batch_size: int = PAPER_BATCH_SIZE,
    final_batch_size: int = PAPER_FINAL_BATCH_SIZE,
    num_hypersamples: int = PAPER_NUM_HYPERSAMPLES,
    mode: str = "non_proportional",
    output_path: Optional[str] = None,
    fast: bool = False,
    use_gpu: bool = True,
    timeout_s: Optional[float] = 600.0,
    baseline_num_shots: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Benchmark all three simulator phases on an ensemble of random circuits.

    Parameters
    ----------
    n: number of qubits
    g: number of gates
    num_instances: circuits to benchmark
    num_shots: shots requested from OptimizedPTSBESimulator (== the paper's
        `mi` for proportional-mode sweeps, which may be swept up to 10,000)
    num_error_sets: E for PTSBE simulators
    batch_size: non-final batch qubit count (paper default 10)
    final_batch_size: final batch qubit count, capped at n (paper default 28)
    num_hypersamples: path optimizer iterations for PTSBE (paper default 100)
    mode: "proportional" | "non_proportional" for OptimizedPTSBESimulator
    output_path: if set, write JSON results to this path
    fast: shortcut for num_hypersamples=1 (dev/test mode, not paper-matching)
    use_gpu: use real bounded-memory GPU contraction (default True)
    timeout_s: per-instance-per-simulator wall-clock budget; instances that
        exceed it (or raise) are recorded with success=False and excluded
        from geometric-mean aggregation, matching the paper's >80%/<80%
        success-rate convention (Sec. IV-D)
    baseline_num_shots: shots requested from Traditional/Unoptimized PTSBE,
        independent of num_shots. Throughput is a per-shot rate, so the
        baseline only needs enough shots for a stable rate estimate; it
        does not need to match num_shots/mi. Defaults to num_shots when
        unset, preserving prior behavior for non-proportional sweeps.
        Set this explicitly (small, e.g. 20-50) for proportional-mode
        sweeps where num_shots/mi is swept up to 10,000 — Traditional's
        per-shot GPU network rebuild (~1-2s each) makes matching mi
        directly impractical.

    Returns
    -------
    List of per-instance result dicts.
    """
    if fast:
        num_hypersamples = 1
    final_batch_size = min(final_batch_size, n)
    batch_size = min(batch_size, n)
    trad_batch_size = min(PAPER_TRADITIONAL_BATCH_SIZE, n)
    baseline_shots = baseline_num_shots if baseline_num_shots is not None else num_shots

    gpu_info = gpu_device_info() if use_gpu else None

    print(f"\nBenchmarking n={n}, g={g}, instances={num_instances}, shots={num_shots} "
          f"(baseline_shots={baseline_shots}), mode={mode}")
    print(f"  batch_size={batch_size}, final_batch_size={final_batch_size}, "
          f"hypersamples={num_hypersamples}, E={num_error_sets}, use_gpu={use_gpu}")
    if gpu_info:
        print(f"  GPU: {gpu_info['gpu_device_name']} "
              f"({gpu_info['gpu_memory_total_bytes'] / 2**30:.1f} GiB)")

    circuits = generate_ensemble(n, g, num_instances)
    all_results: List[Dict[str, Any]] = []

    for inst_id, circuit_data in enumerate(circuits):
        circuit = {k: v for k, v in circuit_data.items() if k != "noise_model"}
        noise_model = circuit_data["noise_model"]

        record: Dict[str, Any] = {
            "n": n, "g": g, "instance_id": inst_id, "mode": mode, "success": True,
            "num_shots_requested": num_shots, "baseline_num_shots": baseline_shots,
            "final_batch_size": final_batch_size,
            "gpu_device_name": gpu_info["gpu_device_name"] if gpu_info else None,
            "gpu_memory_total_bytes": gpu_info["gpu_memory_total_bytes"] if gpu_info else None,
        }

        try:
            ContractionPathCache.clear()
            sim1 = TraditionalTrajectorySimulator(batch_size=trad_batch_size, rng_seed=inst_id, use_gpu=use_gpu)
            r1, t1 = _run_with_timeout(
                lambda: _time_simulator(sim1, circuit, noise_model, baseline_shots, use_gpu), timeout_s
            )
            tp1 = _throughput(len(r1), t1)

            ContractionPathCache.clear()
            sim2 = UnoptimizedPTSBESimulator(
                batch_size=batch_size, num_error_sets=num_error_sets, rng_seed=inst_id, use_gpu=use_gpu
            )
            r2, t2 = _run_with_timeout(
                lambda: _time_simulator(sim2, circuit, noise_model, baseline_shots, use_gpu), timeout_s
            )
            tp2 = _throughput(len(r2), t2)

            ContractionPathCache.clear()
            sim3 = OptimizedPTSBESimulator(
                batch_size=batch_size, final_batch_size=final_batch_size,
                num_hypersamples=num_hypersamples, num_error_sets=num_error_sets,
                rng_seed=inst_id, use_gpu=use_gpu,
            )
            r3, t3 = _run_with_timeout(
                lambda: _time_simulator(sim3, circuit, noise_model, num_shots, use_gpu, mode=mode), timeout_s
            )
            tp3 = _throughput(len(r3), t3)

            record.update({
                "throughput_traditional": tp1,
                "throughput_unoptimized": tp2,
                "throughput_optimized": tp3,
                "speedup_unoptimized_vs_traditional": tp2 / tp1 if tp1 > 0 else float("nan"),
                "speedup_optimized_vs_traditional": tp3 / tp1 if tp1 > 0 else float("nan"),
                "elapsed_traditional_s": t1,
                "elapsed_unoptimized_s": t2,
                "elapsed_optimized_s": t3,
                "num_bitstrings_traditional": len(r1),
                "num_bitstrings_unoptimized": len(r2),
                "num_bitstrings_optimized": len(r3),
            })
            record.update({f"optimized_{k}": v for k, v in _cold_warm_from_engine(sim3).items()})
            record.update({f"traditional_{k}": v for k, v in _cold_warm_from_engine(sim1).items()})

            print(f"  [{inst_id+1}/{num_instances}] speedup optimized/traditional: "
                  f"{record['speedup_optimized_vs_traditional']:.1f}x")
        except (_Timeout, MemoryError, RuntimeError) as exc:
            record["success"] = False
            record["failure_reason"] = f"{type(exc).__name__}: {exc}"
            print(f"  [{inst_id+1}/{num_instances}] FAILED: {record['failure_reason']}")

        all_results.append(record)

    _print_summary(all_results, n, g, gpu_info)

    if output_path:
        _write_json(all_results, output_path)

    return all_results


def run_batch_size_sweep(
    n: int,
    g: int,
    batch_sizes: List[int],
    num_instances: int = 3,
    num_shots: int = DEFAULT_SHOTS,
    num_error_sets: int = 10,
    num_hypersamples: int = PAPER_NUM_HYPERSAMPLES,
    output_path: Optional[str] = None,
    use_gpu: bool = True,
    timeout_s: Optional[float] = 600.0,
) -> List[Dict[str, Any]]:
    """Sweep the non-final batch size bj for a fixed (n, g), recording
    per-batch contraction+sampling time (paper Fig. 7). For each bj, the
    final batch is also set to bj (a single sweep parameter, matching the
    paper's Fig. 7 setup where a single bj is varied)."""
    gpu_info = gpu_device_info() if use_gpu else None
    circuits = generate_ensemble(n, g, num_instances)
    all_results: List[Dict[str, Any]] = []

    for bj in batch_sizes:
        fbs = min(bj, n)
        for inst_id, circuit_data in enumerate(circuits):
            circuit = {k: v for k, v in circuit_data.items() if k != "noise_model"}
            noise_model = circuit_data["noise_model"]
            record: Dict[str, Any] = {
                "n": n, "g": g, "batch_size": bj, "instance_id": inst_id, "success": True,
                "gpu_device_name": gpu_info["gpu_device_name"] if gpu_info else None,
            }
            try:
                ContractionPathCache.clear()
                sim = OptimizedPTSBESimulator(
                    batch_size=min(bj, n), final_batch_size=fbs,
                    num_hypersamples=num_hypersamples, num_error_sets=num_error_sets,
                    rng_seed=inst_id, use_gpu=use_gpu,
                )
                _, elapsed = _run_with_timeout(
                    lambda: _time_simulator(sim, circuit, noise_model, num_shots, use_gpu), timeout_s
                )
                cold_warm = _cold_warm_from_engine(sim)
                record["per_batch_time_s"] = cold_warm["contraction_time_per_call_s"]
                record["elapsed_s"] = elapsed
            except (_Timeout, MemoryError, RuntimeError) as exc:
                record["success"] = False
                record["failure_reason"] = f"{type(exc).__name__}: {exc}"
            all_results.append(record)
            print(f"  bj={bj} [{inst_id+1}/{num_instances}] "
                  f"per_batch_time_s={record.get('per_batch_time_s')}")

    if output_path:
        _write_json(all_results, output_path)
    return all_results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _successful(results: List[Dict]) -> List[Dict]:
    return [r for r in results if r.get("success", True)]


def _valid_speedups(results: List[Dict], key: str) -> List[float]:
    return [r[key] for r in _successful(results) if key in r and math.isfinite(r[key]) and r[key] > 0]


def _geo_stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"geomean": float("nan"), "geostd": float("nan")}
    arr = np.asarray(vals, dtype=float)
    return {
        "geomean": float(gmean(arr)),
        "geostd": float(gstd(arr)) if len(arr) > 1 else 1.0,
    }


def _print_summary(
    results: List[Dict], n: int, g: int, gpu_info: Optional[Dict[str, Any]] = None
) -> None:
    speedups_opt = _valid_speedups(results, "speedup_optimized_vs_traditional")
    speedups_unopt = _valid_speedups(results, "speedup_unoptimized_vs_traditional")
    success_rate = len(_successful(results)) / len(results) if results else float("nan")

    s_opt = _geo_stats(speedups_opt)
    s_unopt = _geo_stats(speedups_unopt)

    print("\n" + "=" * 78)
    if gpu_info:
        print(f"GPU: {gpu_info['gpu_device_name']}")
    print(f"{'n':>4} {'g':>5} {'simulator':>20} {'geomean_speedup':>16} {'geostd':>10} {'success':>9}")
    print("-" * 78)
    print(f"{n:>4} {g:>5} {'unoptimized_ptsbe':>20} {s_unopt['geomean']:>16.2f} "
          f"{s_unopt['geostd']:>10.2f} {success_rate:>8.0%}")
    print(f"{n:>4} {g:>5} {'optimized_ptsbe':>20} {s_opt['geomean']:>16.2f} "
          f"{s_opt['geostd']:>10.2f} {success_rate:>8.0%}")
    print("=" * 78)


def _write_json(results: List[Dict], path: str) -> None:
    import os
    existing = []
    if os.path.exists(path):
        with open(path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    existing.extend(results)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Results written to {path}")
