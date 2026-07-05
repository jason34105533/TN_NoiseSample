"""Tests for ContractionEngine and ContractionPathCache."""
import numpy as np
import pytest
from tn_noise_sim.tensor_network import TensorNetworkBuilder
from tn_noise_sim.contraction import ContractionEngine, ContractionPathCache


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
