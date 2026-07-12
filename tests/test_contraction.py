"""Tests for ContractionEngine and ContractionPathCache."""
import numpy as np
import pytest
from tn_noise_sim.tensor_network import TensorNetworkBuilder
from tn_noise_sim.contraction import (
    ContractionEngine,
    ContractionPathCache,
    HAS_CUPY,
    HAS_NETWORK_STATE,
)

requires_gpu = pytest.mark.skipif(
    not (HAS_CUPY and HAS_NETWORK_STATE), reason="requires cupy + cuquantum.tensornet.experimental"
)


def _h_gate() -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _cnot() -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)


def _bell_circuit() -> dict:
    """H on qubit 0, CNOT on (0,1) → Bell state."""
    return {
        "n_qubits": 2,
        "gates": [
            {"qubits": (0,), "unitary": _h_gate()},
            {"qubits": (0, 1), "unitary": _cnot()},
        ],
    }


def _single_qubit_circuit() -> dict:
    """H on one qubit — probabilities must be 0.5, 0.5."""
    return {
        "n_qubits": 1,
        "gates": [{"qubits": (0,), "unitary": _h_gate()}],
    }


@pytest.fixture(autouse=True)
def clear_cache():
    ContractionPathCache.clear()
    yield
    ContractionPathCache.clear()


# ── 4.1 Cache ─────────────────────────────────────────────────────────────────

def test_cache_hit():
    network = TensorNetworkBuilder.build(_bell_circuit())
    engine = ContractionEngine(batch_size=1, final_batch_size=2, use_gpu=False)
    path1 = engine.find_path(network)
    path2 = engine.find_path(network)
    assert path1 is path2  # same object returned from cache


def test_cache_miss_different_topology():
    c1 = {"n_qubits": 1, "gates": [{"qubits": (0,), "unitary": _h_gate()}]}
    c2 = {"n_qubits": 2, "gates": [{"qubits": (0,), "unitary": _h_gate()},
                                     {"qubits": (1,), "unitary": _h_gate()}]}
    n1 = TensorNetworkBuilder.build(c1)
    n2 = TensorNetworkBuilder.build(c2)
    engine = ContractionEngine(batch_size=1, final_batch_size=2, use_gpu=False)
    path1 = engine.find_path(n1)
    path2 = engine.find_path(n2)
    # Different topologies → different cache entries (paths may differ)
    h1 = ContractionPathCache.topology_hash(n1)
    h2 = ContractionPathCache.topology_hash(n2)
    assert h1 != h2


# ── 4.2/4.3 Contract batch — marginal sums to 1 ───────────────────────────────

def test_single_qubit_marginal_sums_to_one():
    circuit = _single_qubit_circuit()
    network = TensorNetworkBuilder.build(circuit)
    engine = ContractionEngine(batch_size=1, final_batch_size=1, use_gpu=False)
    path = engine.find_path(network)
    marginal = engine.contract_batch(network, path, batch_index=1, num_batches=1)
    assert marginal.shape == (2,), f"Expected (2,), got {marginal.shape}"
    assert abs(marginal.sum() - 1.0) < 1e-6, f"Sum={marginal.sum()}"


def test_single_qubit_h_probabilities():
    """H gate on |0> → 50/50."""
    circuit = _single_qubit_circuit()
    network = TensorNetworkBuilder.build(circuit)
    engine = ContractionEngine(batch_size=1, final_batch_size=1, use_gpu=False)
    path = engine.find_path(network)
    marginal = engine.contract_batch(network, path, batch_index=1, num_batches=1)
    assert abs(marginal[0] - 0.5) < 1e-6, f"P(0)={marginal[0]}"
    assert abs(marginal[1] - 0.5) < 1e-6, f"P(1)={marginal[1]}"


def test_bell_state_marginal():
    """Bell state: P(00)=0.5, P(01)=0, P(10)=0, P(11)=0.5."""
    circuit = _bell_circuit()
    network = TensorNetworkBuilder.build(circuit)
    engine = ContractionEngine(batch_size=2, final_batch_size=2, use_gpu=False)
    path = engine.find_path(network)
    marginal = engine.contract_batch(network, path, batch_index=1, num_batches=1)
    assert abs(marginal.sum() - 1.0) < 1e-6
    # marginal[0]=P(00), marginal[1]=P(01), marginal[2]=P(10), marginal[3]=P(11)
    assert abs(marginal[0] - 0.5) < 1e-5, f"P(00)={marginal[0]}"
    assert abs(marginal[3] - 0.5) < 1e-5, f"P(11)={marginal[3]}"
    assert abs(marginal[1]) < 1e-5, f"P(01)={marginal[1]}"
    assert abs(marginal[2]) < 1e-5, f"P(10)={marginal[2]}"


# ── 4.5 Conditional marginal ──────────────────────────────────────────────────

def test_conditional_marginal_sums_to_one():
    """After conditioning, marginal should still sum to ≤ 1."""
    circuit = {
        "n_qubits": 2,
        "gates": [{"qubits": (0,), "unitary": _h_gate()},
                  {"qubits": (1,), "unitary": _h_gate()}],
    }
    network = TensorNetworkBuilder.build(circuit)
    engine = ContractionEngine(batch_size=1, final_batch_size=1, use_gpu=False)
    path = engine.find_path(network)
    # Batch 1: get marginal over qubit 0
    m1 = engine.contract_batch(network, path, batch_index=1, num_batches=2)
    assert abs(m1.sum() - 1.0) < 1e-6

    # Batch 2: condition on qubit 0 = 0
    m2 = engine.contract_batch(network, path, batch_index=2, num_batches=2, prefix=(0,))
    assert m2.sum() <= 1.0 + 1e-6


def test_conditional_marginal_h_independent():
    """Two independent H gates: conditioning on qubit 0=0 shouldn't change P(qubit 1)."""
    circuit = {
        "n_qubits": 2,
        "gates": [{"qubits": (0,), "unitary": _h_gate()},
                  {"qubits": (1,), "unitary": _h_gate()}],
    }
    network = TensorNetworkBuilder.build(circuit)
    engine = ContractionEngine(batch_size=1, final_batch_size=1, use_gpu=False)
    path = engine.find_path(network)
    m2 = engine.contract_batch(network, path, batch_index=2, num_batches=2, prefix=(0,))
    # Qubit 1 is independent → still 50/50
    assert abs(m2[0] - 0.5) < 0.05, f"P(q1=0|q0=0)={m2[0]}"


# ── 4.4 Final batch size ──────────────────────────────────────────────────────

def test_num_batches_calculation():
    """n=4 qubits, batch_size=2, final_batch_size=2 → 2 batches."""
    circuit = {
        "n_qubits": 4,
        "gates": [{"qubits": (i,), "unitary": _h_gate()} for i in range(4)],
    }
    network = TensorNetworkBuilder.build(circuit)
    engine = ContractionEngine(batch_size=2, final_batch_size=2, use_gpu=False)
    assert engine._num_batches(network) == 2


# ── GPU-backed (NetworkState) bounded-memory contraction ───────────────────────

def _entangled_circuit_with_errors(n: int = 5, g: int = 10, seed: int = 0):
    """A small circuit with single- and two-qubit gates plus a fused error_set,
    small enough that the CPU dense-statevector path can also compute it, so
    GPU results can be checked against it directly."""
    rng = np.random.default_rng(seed)
    gates = []
    error_set = {}
    for gate_idx in range(g):
        if n >= 2 and rng.random() < 0.4:
            q0 = int(rng.integers(0, n - 1))
            z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
            u, _ = np.linalg.qr(z)
            gates.append({"qubits": (q0, q0 + 1), "unitary": u})
        else:
            q = int(rng.integers(0, n))
            z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
            u, _ = np.linalg.qr(z)
            gates.append({"qubits": (q,), "unitary": u})
        if rng.random() < 0.5:
            nq = len(gates[-1]["qubits"])
            d = 2 ** nq
            z = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
            k, _ = np.linalg.qr(z)
            error_set[gate_idx] = k
    return {"n_qubits": n, "gates": gates}, error_set


def _random_error_set_for_circuit(circuit: dict, seed: int, prob: float = 0.5):
    """Sample a fused error_set sized to match `circuit`'s own gate arities."""
    rng = np.random.default_rng(seed)
    error_set = {}
    for gate_idx, gate in enumerate(circuit["gates"]):
        if rng.random() < prob:
            nq = len(gate["qubits"])
            d = 2 ** nq
            z = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
            k, _ = np.linalg.qr(z)
            error_set[gate_idx] = k
    return error_set


@requires_gpu
def test_gpu_matches_cpu_unconditioned_marginal():
    n, b = 5, 3
    circuit, error_set = _entangled_circuit_with_errors(n=n, g=10, seed=1)

    cpu_engine = ContractionEngine(batch_size=b, final_batch_size=n - b, use_gpu=False)
    cpu_network = TensorNetworkBuilder.build(circuit, error_set=error_set, mode="fuse")
    cpu_path = cpu_engine.find_path(cpu_network)
    cpu_marginal = cpu_engine.contract_batch(cpu_network, cpu_path, batch_index=1, num_batches=2)

    gpu_engine = ContractionEngine(batch_size=b, final_batch_size=n - b, use_gpu=True)
    handle = gpu_engine.build_gpu_network(circuit, error_set=error_set, mode="fuse", num_hypersamples=1)
    try:
        gpu_marginal = gpu_engine.contract_batch(handle, handle, batch_index=1, num_batches=2)
    finally:
        handle.free()

    np.testing.assert_allclose(gpu_marginal, cpu_marginal, atol=1e-6)


@requires_gpu
def test_gpu_matches_cpu_conditioned_marginal():
    n, b = 5, 3
    circuit, error_set = _entangled_circuit_with_errors(n=n, g=10, seed=2)
    prefix = (1,)  # arbitrary fixed prefix value for batch 1 (b=3 qubits -> idx in [0,8))

    cpu_engine = ContractionEngine(batch_size=b, final_batch_size=n - b, use_gpu=False)
    cpu_network = TensorNetworkBuilder.build(circuit, error_set=error_set, mode="fuse")
    cpu_path = cpu_engine.find_path(cpu_network)
    cpu_marginal = cpu_engine.contract_batch(
        cpu_network, cpu_path, batch_index=2, num_batches=2, prefix=prefix
    )

    gpu_engine = ContractionEngine(batch_size=b, final_batch_size=n - b, use_gpu=True)
    handle = gpu_engine.build_gpu_network(circuit, error_set=error_set, mode="fuse", num_hypersamples=1)
    try:
        gpu_marginal = gpu_engine.contract_batch(
            handle, handle, batch_index=2, num_batches=2, prefix=prefix
        )
    finally:
        handle.free()

    np.testing.assert_allclose(gpu_marginal, cpu_marginal, atol=1e-6)


@requires_gpu
def test_gpu_upv_update_matches_fresh_build():
    """UPV: applying an error set via update_tensor_operator on a persistent
    NetworkState must match building a fresh fused NetworkState directly."""
    n, b = 5, 3
    circuit, error_set_a = _entangled_circuit_with_errors(n=n, g=10, seed=3)
    error_set_b = _random_error_set_for_circuit(circuit, seed=4)

    engine = ContractionEngine(batch_size=b, final_batch_size=n - b, use_gpu=True)

    noiseless_handle = engine.build_gpu_network(circuit, mode="noiseless", num_hypersamples=1)
    try:
        touched_a = engine.apply_error_set_gpu(noiseless_handle, error_set_a)
        marginal_a_via_update = engine.contract_batch(
            noiseless_handle, noiseless_handle, batch_index=1, num_batches=2
        )
        engine.revert_error_set_gpu(noiseless_handle, touched_a)

        touched_b = engine.apply_error_set_gpu(noiseless_handle, error_set_b)
        marginal_b_via_update = engine.contract_batch(
            noiseless_handle, noiseless_handle, batch_index=1, num_batches=2
        )
        engine.revert_error_set_gpu(noiseless_handle, touched_b)
    finally:
        noiseless_handle.free()

    fresh_a = engine.build_gpu_network(circuit, error_set=error_set_a, mode="fuse", num_hypersamples=1)
    try:
        marginal_a_fresh = engine.contract_batch(fresh_a, fresh_a, batch_index=1, num_batches=2)
    finally:
        fresh_a.free()

    fresh_b = engine.build_gpu_network(circuit, error_set=error_set_b, mode="fuse", num_hypersamples=1)
    try:
        marginal_b_fresh = engine.contract_batch(fresh_b, fresh_b, batch_index=1, num_batches=2)
    finally:
        fresh_b.free()

    np.testing.assert_allclose(marginal_a_via_update, marginal_a_fresh, atol=1e-6)
    np.testing.assert_allclose(marginal_b_via_update, marginal_b_fresh, atol=1e-6)


@requires_gpu
def test_gpu_bounded_memory_beyond_cpu_feasible_scale():
    """n=40 has no physical 2^40-entry dense array; the GPU path must still
    complete a bounded (2^b) batch contraction without attempting one."""
    n, b = 40, 10
    circuit, error_set = _entangled_circuit_with_errors(n=n, g=30, seed=5)

    engine = ContractionEngine(batch_size=b, final_batch_size=b, use_gpu=True)
    handle = engine.build_gpu_network(circuit, error_set=error_set, mode="fuse", num_hypersamples=1)
    try:
        marginal = engine.contract_batch(handle, handle, batch_index=1, num_batches=engine._num_batches(handle))
    finally:
        handle.free()

    assert marginal.shape == (2 ** b,)
    assert abs(marginal.sum() - 1.0) < 1e-4
