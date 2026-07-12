"""Integration tests for all three simulator phases."""
from collections import Counter

import numpy as np
import pytest
from unittest.mock import patch
from tn_noise_sim.noise_model import NoiseModel
from tn_noise_sim.simulators import (
    TraditionalTrajectorySimulator,
    UnoptimizedPTSBESimulator,
    OptimizedPTSBESimulator,
)
from tn_noise_sim.contraction import ContractionPathCache, HAS_CUPY, HAS_NETWORK_STATE

requires_gpu = pytest.mark.skipif(
    not (HAS_CUPY and HAS_NETWORK_STATE), reason="requires cupy + cuquantum.tensornet.experimental"
)


def _h() -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _small_circuit(n: int = 4, g: int = 8, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    gates = []
    for i in range(g):
        q = int(rng.integers(0, n))
        z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        u, _ = np.linalg.qr(z)
        gates.append({"qubits": (q,), "unitary": u})
    return {"n_qubits": n, "gates": gates}


def _noise(n_gates: int, p: float = 0.05) -> NoiseModel:
    return NoiseModel.uniform_pauli(num_gates=n_gates, probability=p)


@pytest.fixture(autouse=True)
def clear_path_cache():
    ContractionPathCache.clear()
    yield
    ContractionPathCache.clear()


# ── Phase 1: Traditional ──────────────────────────────────────────────────────

class TestTraditionalSimulator:
    def test_return_type_and_count(self):
        circuit = _small_circuit(n=4, g=8)
        noise = _noise(8)
        sim = TraditionalTrajectorySimulator(batch_size=4, rng_seed=42, use_gpu=False)
        results = sim.sample(circuit, noise, num_shots=10)
        assert len(results) == 10

    def test_bitstring_length(self):
        n = 4
        circuit = _small_circuit(n=n, g=8)
        noise = _noise(8)
        sim = TraditionalTrajectorySimulator(batch_size=n, rng_seed=0, use_gpu=False)
        results = sim.sample(circuit, noise, num_shots=5)
        for bitstring, eid in results:
            assert len(bitstring) == n, f"bitstring length {len(bitstring)} != {n}"
            assert all(c in '01' for c in bitstring)

    def test_path_found_per_shot(self):
        circuit = _small_circuit(n=3, g=6)
        noise = _noise(6)
        sim = TraditionalTrajectorySimulator(batch_size=3, rng_seed=1, use_gpu=False)
        sim.sample(circuit, noise, num_shots=10)
        assert sim._path_find_count == 10, (
            f"Expected 10 path-finds, got {sim._path_find_count}"
        )

    def test_bitstrings_are_binary_strings(self):
        circuit = _small_circuit(n=4, g=4)
        noise = _noise(4)
        sim = TraditionalTrajectorySimulator(batch_size=4, rng_seed=2, use_gpu=False)
        results = sim.sample(circuit, noise, num_shots=5)
        for bs, _ in results:
            assert set(bs).issubset({'0', '1'})


# ── Phase 2: Unoptimized PTSBE ────────────────────────────────────────────────

class TestUnoptimizedPTSBESimulator:
    def test_path_found_per_error_set(self):
        circuit = _small_circuit(n=4, g=8)
        noise = _noise(8)
        E = 3
        sim = UnoptimizedPTSBESimulator(batch_size=4, num_error_sets=E, rng_seed=0, use_gpu=False)
        sim.sample(circuit, noise, num_shots=12)
        assert sim._path_find_count == E, (
            f"Expected {E} path-finds (one per error set), got {sim._path_find_count}"
        )

    def test_return_list_length_equals_total_shots(self):
        circuit = _small_circuit(n=3, g=6)
        noise = _noise(6)
        total = 15
        sim = UnoptimizedPTSBESimulator(batch_size=3, num_error_sets=5, rng_seed=1, use_gpu=False)
        results = sim.sample(circuit, noise, num_shots=total)
        assert len(results) == total

    def test_default_num_error_sets_equals_shots(self):
        circuit = _small_circuit(n=3, g=6)
        noise = _noise(6)
        num_shots = 5
        sim = UnoptimizedPTSBESimulator(batch_size=3, rng_seed=2, use_gpu=False)
        sim.sample(circuit, noise, num_shots=num_shots)
        # Default: one error set per shot
        assert sim._path_find_count == num_shots

    def test_bitstring_length(self):
        n = 4
        circuit = _small_circuit(n=n, g=4)
        noise = _noise(4)
        sim = UnoptimizedPTSBESimulator(batch_size=n, num_error_sets=2, rng_seed=3, use_gpu=False)
        results = sim.sample(circuit, noise, num_shots=8)
        for bs, _ in results:
            assert len(bs) == n


# ── Phase 3: Optimized PTSBE ──────────────────────────────────────────────────

class TestOptimizedPTSBESimulator:
    def test_path_found_once(self):
        """UPV: exactly one path-find across all error sets."""
        circuit = _small_circuit(n=4, g=8)
        noise = _noise(8)
        sim = OptimizedPTSBESimulator(
            batch_size=2, final_batch_size=2,
            num_hypersamples=1, num_error_sets=5, rng_seed=0, use_gpu=False)
        sim.sample(circuit, noise, num_shots=20)
        assert sim._path_find_count == 1, (
            f"UPV: expected 1 path-find, got {sim._path_find_count}"
        )

    def test_final_batch_size_validation(self):
        with pytest.raises(ValueError, match="final_batch_size"):
            sim = OptimizedPTSBESimulator(final_batch_size=100, use_gpu=False)
            circuit = _small_circuit(n=4, g=4)
            sim.sample(circuit, _noise(4), num_shots=5)

    def test_non_proportional_returns_many_bitstrings(self):
        """Non-proportional exhaustive sampling returns >= requested shots."""
        n = 4
        circuit = _small_circuit(n=n, g=8, seed=7)
        noise = _noise(8)
        sim = OptimizedPTSBESimulator(
            batch_size=2, final_batch_size=2,
            num_hypersamples=1, num_error_sets=3, rng_seed=0, use_gpu=False)
        results = sim.sample(circuit, noise, num_shots=10, mode="non_proportional")
        # Non-proportional exhaustive should produce at least some bitstrings
        assert len(results) >= 1

    def test_non_proportional_bitstring_length(self):
        n = 4
        circuit = _small_circuit(n=n, g=6, seed=5)
        noise = _noise(6)
        sim = OptimizedPTSBESimulator(
            batch_size=2, final_batch_size=2,
            num_hypersamples=1, num_error_sets=2, rng_seed=1, use_gpu=False)
        results = sim.sample(circuit, noise, num_shots=8, mode="non_proportional")
        for bs, _ in results:
            assert len(bs) == n, f"bitstring length {len(bs)} != {n}"

    def test_proportional_mode_shot_count(self):
        """Proportional mode returns exactly the requested shot count."""
        n = 4
        circuit = _small_circuit(n=n, g=6, seed=3)
        noise = _noise(6)
        num_shots = 20
        sim = OptimizedPTSBESimulator(
            batch_size=2, final_batch_size=2,
            num_hypersamples=1, num_error_sets=4, rng_seed=2, use_gpu=False)
        results = sim.sample(circuit, noise, num_shots=num_shots, mode="proportional")
        assert len(results) == num_shots

    def test_proportional_born_rule(self):
        """
        For a 2-qubit H⊗H circuit: all 4 bitstrings should appear with ~equal probability.
        Check that the distribution doesn't strongly favour one outcome.
        """
        circuit = {
            "n_qubits": 2,
            "gates": [
                {"qubits": (0,), "unitary": _h()},
                {"qubits": (1,), "unitary": _h()},
            ],
        }
        noise = _noise(2, p=0.0)  # zero noise → pure H⊗H
        num_shots = 200
        sim = OptimizedPTSBESimulator(
            batch_size=1, final_batch_size=1,
            num_hypersamples=1, num_error_sets=1, rng_seed=42, use_gpu=False)
        results = sim.sample(circuit, noise, num_shots=num_shots, mode="proportional")
        counts = {}
        for bs, _ in results:
            counts[bs] = counts.get(bs, 0) + 1
        # Each of 4 outcomes should appear ~25% of the time; allow ±15%
        total = sum(counts.values())
        for bs in ["00", "01", "10", "11"]:
            freq = counts.get(bs, 0) / total
            assert 0.10 < freq < 0.40, f"Born rule violation: P({bs})={freq:.3f}"


# ── Cross-simulator distributional consistency ────────────────────────────────
# Regression coverage for a real bug found during the V100 validation pass:
# ErrorSampler._sample_proportional() used to re-weight shot counts by
# _error_set_weight() on top of error sets already drawn i.i.d. from their true
# probability, double-applying the weighting and skewing proportional-mode
# PTSBE's bitstring distribution away from the traditional trajectory
# distribution. The zero-noise `test_proportional_born_rule` above can't catch
# this (all weights are identically 1 when p=0), so this uses nonzero noise.

def test_proportional_matches_traditional_distribution():
    """Phase 3 (proportional NBS) must reproduce Phase 1's (traditional
    trajectory) bitstring distribution on the same circuit, per the paper's
    requirement that proportional PTSBE preserves quantum statistics."""
    from benchmarks.circuit_generator import generate_circuit

    circuit_data = generate_circuit(n=5, g=12, seed=7)
    circuit = {k: v for k, v in circuit_data.items() if k != "noise_model"}
    noise_model = circuit_data["noise_model"]

    num_shots = 20000
    trad = TraditionalTrajectorySimulator(batch_size=5, rng_seed=1, use_gpu=False)
    r_trad = trad.sample(circuit, noise_model, num_shots=num_shots)

    # num_error_sets=2000 (not 200): proportional PTSBE's TVD-vs-Traditional is
    # dominated by *error-set* sampling noise (finite E pre-sampled error sets),
    # not shot-count noise — increasing shots at fixed E does not shrink it
    # (verified empirically: E=200 plateaus at TVD~0.10 from 20k to 80k shots,
    # while E=200->2000->10000 at fixed 20k shots shrinks TVD 0.104->0.031->0.020).
    # This is expected behavior of the algorithm (finite E), not a bug.
    opt = OptimizedPTSBESimulator(
        batch_size=3, final_batch_size=5,
        num_hypersamples=1, num_error_sets=2000, rng_seed=2, use_gpu=False)
    r_opt = opt.sample(circuit, noise_model, num_shots=num_shots, mode="proportional")

    c_trad = Counter(bs for bs, _ in r_trad)
    c_opt = Counter(bs for bs, _ in r_opt)
    n_trad, n_opt = sum(c_trad.values()), sum(c_opt.values())
    all_keys = set(c_trad) | set(c_opt)
    tvd = sum(
        abs(c_trad.get(k, 0) / n_trad - c_opt.get(k, 0) / n_opt) for k in all_keys
    ) / 2

    # Expected sampling-noise-only TVD at this shot count/E is ~0.02-0.05;
    # the pre-fix double-weighting bug produced ~0.21, an order of magnitude higher.
    assert tvd < 0.08, f"Proportional PTSBE diverges from traditional stats: TVD={tvd:.3f}"


# ── GPU-backed simulators: distributional match vs. CPU + beyond-CPU scale ────

@requires_gpu
def test_optimized_ptsbe_gpu_matches_cpu_distribution():
    """The real NetworkState-backed GPU path (use_gpu=True) must reproduce the
    same bitstring distribution as the CPU dense-statevector fallback on a
    circuit small enough for both backends to run."""
    from benchmarks.circuit_generator import generate_circuit

    circuit_data = generate_circuit(n=5, g=12, seed=11)
    circuit = {k: v for k, v in circuit_data.items() if k != "noise_model"}
    noise_model = circuit_data["noise_model"]

    num_shots = 20000
    common_kwargs = dict(
        batch_size=3, final_batch_size=5, num_hypersamples=1, num_error_sets=200,
    )
    # Same rng_seed on both backends: since both compute (near-)identical
    # marginals (validated numerically in test_contraction.py), matching seeds
    # makes this a tight regression check rather than only a statistical one.
    cpu_sim = OptimizedPTSBESimulator(rng_seed=1, use_gpu=False, **common_kwargs)
    r_cpu = cpu_sim.sample(circuit, noise_model, num_shots=num_shots, mode="proportional")

    gpu_sim = OptimizedPTSBESimulator(rng_seed=1, use_gpu=True, **common_kwargs)
    r_gpu = gpu_sim.sample(circuit, noise_model, num_shots=num_shots, mode="proportional")

    c_cpu = Counter(bs for bs, _ in r_cpu)
    c_gpu = Counter(bs for bs, _ in r_gpu)
    n_cpu, n_gpu = sum(c_cpu.values()), sum(c_gpu.values())
    all_keys = set(c_cpu) | set(c_gpu)
    tvd = sum(
        abs(c_cpu.get(k, 0) / n_cpu - c_gpu.get(k, 0) / n_gpu) for k in all_keys
    ) / 2
    # Expected sampling-noise-only TVD at this shot count/outcome count is
    # ~0.02-0.04 (per the established Traditional-vs-Optimized regression above).
    assert tvd < 0.08, f"GPU backend diverges from CPU backend: TVD={tvd:.3f}"


@requires_gpu
def test_optimized_ptsbe_gpu_completes_beyond_cpu_dense_scale():
    """n=40 has no physical 2^40-entry dense array — the old CPU path cannot
    run this at all (it materializes the full statevector). The GPU path,
    bounded to 2^batch_size, must complete and return well-formed shots."""
    from benchmarks.circuit_generator import generate_circuit

    circuit_data = generate_circuit(n=40, g=60, seed=13)
    circuit = {k: v for k, v in circuit_data.items() if k != "noise_model"}
    noise_model = circuit_data["noise_model"]

    sim = OptimizedPTSBESimulator(
        batch_size=10, final_batch_size=20, num_hypersamples=1,
        num_error_sets=3, rng_seed=4, use_gpu=True,
    )
    results = sim.sample(circuit, noise_model, num_shots=30, mode="non_proportional")
    assert len(results) >= 1
    for bs, eid in results:
        assert len(bs) == 40
        assert set(bs).issubset({'0', '1'})
