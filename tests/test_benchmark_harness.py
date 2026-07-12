"""Tests for the benchmarking harness's GPU device metadata reporting."""
import numpy as np
import pytest

from benchmarks.timing import gpu_device_info
from tests.conftest import requires_gpu


def test_gpu_device_info_none_without_gpu(monkeypatch):
    import benchmarks.timing as timing_mod

    monkeypatch.setattr(timing_mod, "HAS_CUPY", False)
    assert gpu_device_info() is None


@requires_gpu
def test_gpu_device_info_reports_real_device():
    info = gpu_device_info()
    assert info is not None
    assert isinstance(info["gpu_device_name"], str) and info["gpu_device_name"]
    assert isinstance(info["gpu_memory_total_bytes"], int)
    assert info["gpu_memory_total_bytes"] > 0


def test_run_benchmark_record_has_gpu_fields_when_no_gpu():
    from benchmarks.run_benchmark import run_benchmark

    results = run_benchmark(
        n=3, g=3, num_instances=1, num_shots=2, num_error_sets=2,
        batch_size=2, final_batch_size=3, fast=True, use_gpu=False,
    )
    assert len(results) == 1
    assert results[0]["gpu_device_name"] is None
    assert results[0]["gpu_memory_total_bytes"] is None


# ── Circuit generator: paper-exact gate set (Sec. IV-B) ─────────────────────

def test_generate_circuit_gate_set_matches_paper():
    from benchmarks import circuit_generator as cg

    fixed_1q = [cg._H, cg._X, cg._Y, cg._Z, cg._T]
    fixed_2q = list(cg._TWO_QUBIT_FIXED.values())

    def _is_rx(u):
        if not (abs(u[0, 0] - u[1, 1]) < 1e-9 and abs(u[0, 1] - u[1, 0]) < 1e-9):
            return False
        return abs(u[0, 0].imag) < 1e-9 and abs(u[0, 1].real) < 1e-9

    def _is_crx(u):
        if not np.allclose(u[:2, :2], np.eye(2)):
            return False
        return _is_rx(u[2:, 2:])

    circuit = cg.generate_circuit(n=6, g=100, seed=0)
    for gate in circuit["gates"]:
        u = gate["unitary"]
        nq = len(gate["qubits"])
        if nq == 1:
            assert u.shape == (2, 2)
            assert any(np.allclose(u, f) for f in fixed_1q) or _is_rx(u), \
                f"single-qubit gate not in {{H,X,Y,Z,T,Rx}}: {u}"
        else:
            assert nq == 2
            assert u.shape == (4, 4)
            q0, q1 = gate["qubits"]
            assert q1 == q0 + 1, "two-qubit gate must act on nearest-neighbor qubits"
            assert any(np.allclose(u, f) for f in fixed_2q) or _is_crx(u), \
                f"two-qubit gate not in {{CX,CY,CZ,CH,CRx}}: {u}"
        # every gate must be unitary regardless of which named gate it is
        d = u.shape[0]
        assert np.allclose(u @ u.conj().T, np.eye(d), atol=1e-8)


def test_generate_circuit_two_qubit_fraction_default():
    from benchmarks.circuit_generator import generate_circuit

    circuit = generate_circuit(n=10, g=2000, seed=1)
    frac = sum(1 for g in circuit["gates"] if len(g["qubits"]) == 2) / len(circuit["gates"])
    assert 0.15 < frac < 0.25, f"two-qubit fraction {frac:.3f} not near paper's 20%"


def test_generate_circuit_noise_matches_gate_arity():
    from benchmarks.circuit_generator import generate_circuit
    from tn_noise_sim.noise_model import ErrorType

    circuit = generate_circuit(n=6, g=50, seed=2)
    noise_model = circuit["noise_model"]
    for gate_idx, gate in enumerate(circuit["gates"]):
        spec = noise_model.get(gate_idx)
        assert spec is not None
        assert 0.02 <= spec.probability <= 0.20
        if len(gate["qubits"]) == 1:
            assert spec.error_type == ErrorType.PAULI
        else:
            assert spec.error_type == ErrorType.DEPOLARIZING


# ── Geometric-mean/std aggregation and success tracking ─────────────────────

def test_geo_stats_uses_geometric_not_arithmetic_mean():
    from benchmarks.run_benchmark import _geo_stats

    # Geometric mean of [1, 100] is 10, arithmetic mean would be 50.5
    stats = _geo_stats([1.0, 100.0])
    assert abs(stats["geomean"] - 10.0) < 1e-6


def test_run_benchmark_marks_failed_instance_and_excludes_from_geomean():
    from benchmarks.run_benchmark import run_benchmark, _successful

    # timeout_s=1e-9 forces every instance to exceed budget immediately
    results = run_benchmark(
        n=3, g=3, num_instances=2, num_shots=2, num_error_sets=2,
        batch_size=2, final_batch_size=3, fast=True, use_gpu=False,
        timeout_s=1e-9,
    )
    assert len(results) == 2
    assert all(r["success"] is False for r in results)
    assert _successful(results) == []


def test_run_benchmark_proportional_mode_plumbed_to_simulator():
    from benchmarks.run_benchmark import run_benchmark

    results = run_benchmark(
        n=3, g=3, num_instances=1, num_shots=4, num_error_sets=2,
        batch_size=2, final_batch_size=3, fast=True, use_gpu=False,
        mode="proportional",
    )
    assert results[0]["mode"] == "proportional"
    assert results[0]["success"] is True
    # proportional mode returns exactly num_shots bitstrings
    assert results[0]["num_bitstrings_optimized"] == 4
