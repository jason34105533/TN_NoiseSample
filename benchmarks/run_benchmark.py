"""Core benchmarking logic: run all three simulator phases and collect metrics."""
from __future__ import annotations

import json
import math
import statistics
from typing import Any, Dict, List, Optional

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


def _throughput(num_bitstrings: int, elapsed_s: float) -> float:
    """Unique bitstrings per GPU/CPU second."""
    if elapsed_s <= 0:
        return float("inf")
    return num_bitstrings / elapsed_s


def _time_simulator(sim, circuit: dict, noise_model, num_shots: int, use_gpu: bool = False) -> tuple:
    """Run simulator and return (results, elapsed_seconds)."""
    t = Timer(use_gpu=use_gpu)
    with t:
        results = sim.sample(circuit, noise_model, num_shots=num_shots)
    return results, t.elapsed_s


def run_benchmark(
    n: int,
    g: int,
    num_instances: int = 10,
    num_shots: int = DEFAULT_SHOTS,
    num_error_sets: int = 10,
    batch_size: int = 10,
    final_batch_size: int = 14,
    num_hypersamples: int = 1,
    output_path: Optional[str] = None,
    fast: bool = False,
    use_gpu: bool = False,
) -> List[Dict[str, Any]]:
    """
    Benchmark all three simulator phases on an ensemble of random circuits.

    Parameters
    ----------
    n: number of qubits
    g: number of gates
    num_instances: circuits to benchmark
    num_shots: shots per circuit instance
    num_error_sets: E for PTSBE simulators
    batch_size: non-final batch qubit count
    final_batch_size: final batch qubit count (capped at n)
    num_hypersamples: path optimizer iterations (1 for fast/test mode)
    output_path: if set, write JSON results to this path
    fast: shortcut for num_hypersamples=1
    use_gpu: use CUDA timing (requires cupy)

    Returns
    -------
    List of per-instance result dicts.
    """
    if fast:
        num_hypersamples = 1
    final_batch_size = min(final_batch_size, n)
    batch_size = min(batch_size, n)

    gpu_info = gpu_device_info() if use_gpu else None

    print(f"\nBenchmarking n={n}, g={g}, instances={num_instances}, shots={num_shots}")
    print(f"  batch_size={batch_size}, final_batch_size={final_batch_size}, "
          f"hypersamples={num_hypersamples}, E={num_error_sets}")
    if gpu_info:
        print(f"  GPU: {gpu_info['gpu_device_name']} "
              f"({gpu_info['gpu_memory_total_bytes'] / 2**30:.1f} GiB)")

    circuits = generate_ensemble(n, g, num_instances)
    all_results: List[Dict[str, Any]] = []

    for inst_id, circuit_data in enumerate(circuits):
        circuit = {k: v for k, v in circuit_data.items() if k != "noise_model"}
        noise_model = circuit_data["noise_model"]

        ContractionPathCache.clear()

        # ── Phase 1: Traditional ──────────────────────────────────────────────
        sim1 = TraditionalTrajectorySimulator(batch_size=batch_size, rng_seed=inst_id)
        r1, t1 = _time_simulator(sim1, circuit, noise_model, num_shots, use_gpu)
        tp1 = _throughput(len(r1), t1)

        ContractionPathCache.clear()

        # ── Phase 2: Unoptimized PTSBE ────────────────────────────────────────
        sim2 = UnoptimizedPTSBESimulator(
            batch_size=batch_size,
            num_error_sets=num_error_sets,
            rng_seed=inst_id,
        )
        r2, t2 = _time_simulator(sim2, circuit, noise_model, num_shots, use_gpu)
        tp2 = _throughput(len(r2), t2)

        ContractionPathCache.clear()

        # ── Phase 3: Optimized PTSBE ──────────────────────────────────────────
        sim3 = OptimizedPTSBESimulator(
            batch_size=batch_size,
            final_batch_size=final_batch_size,
            num_hypersamples=num_hypersamples,
            num_error_sets=num_error_sets,
            rng_seed=inst_id,
        )
        r3, t3 = _time_simulator(sim3, circuit, noise_model, num_shots, use_gpu)
        tp3 = _throughput(len(r3), t3)

        record: Dict[str, Any] = {
            "n": n,
            "g": g,
            "instance_id": inst_id,
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
            "gpu_device_name": gpu_info["gpu_device_name"] if gpu_info else None,
            "gpu_memory_total_bytes": gpu_info["gpu_memory_total_bytes"] if gpu_info else None,
        }
        all_results.append(record)
        print(f"  [{inst_id+1}/{num_instances}] speedup optimized/traditional: "
              f"{record['speedup_optimized_vs_traditional']:.1f}x")

    _print_summary(all_results, n, g, gpu_info)

    if output_path:
        _write_json(all_results, output_path)

    return all_results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _valid_speedups(results: List[Dict], key: str) -> List[float]:
    return [r[key] for r in results if math.isfinite(r[key])]


def _print_summary(
    results: List[Dict], n: int, g: int, gpu_info: Optional[Dict[str, Any]] = None
) -> None:
    speedups_opt = _valid_speedups(results, "speedup_optimized_vs_traditional")
    speedups_unopt = _valid_speedups(results, "speedup_unoptimized_vs_traditional")

    def _stats(vals):
        if not vals:
            return {"mean": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan")}
        return {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }

    s_opt = _stats(speedups_opt)
    s_unopt = _stats(speedups_unopt)

    print("\n" + "=" * 70)
    if gpu_info:
        print(f"GPU: {gpu_info['gpu_device_name']}")
    print(f"{'n':>4} {'g':>5} {'simulator':>20} {'mean_speedup':>14} {'std':>10}")
    print("-" * 70)
    print(f"{n:>4} {g:>5} {'unoptimized_ptsbe':>20} {s_unopt['mean']:>14.2f} "
          f"{s_unopt['std']:>10.2f}")
    print(f"{n:>4} {g:>5} {'optimized_ptsbe':>20} {s_opt['mean']:>14.2f} "
          f"{s_opt['std']:>10.2f}")
    print("=" * 70)


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
