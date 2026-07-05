"""Tests for TensorNetworkBuilder."""
import numpy as np
import pytest
from collections import Counter
from tn_noise_sim.tensor_network import TensorNetworkBuilder, TNNetwork
from tn_noise_sim.noise_model import NoiseModel
from tn_noise_sim.error_sampling import ErrorSampler, _I, _X

from tests.conftest import requires_qiskit


def _random_unitary(d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    u, _ = np.linalg.qr(z)
    return u


def _circuit_1q(n: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    gates = []
    for i in range(n):
        u = _random_unitary(2, seed=i)
        gates.append({"qubits": (i,), "unitary": u})
    return {"n_qubits": n, "gates": gates}


def _circuit_2q(n: int = 4, seed: int = 0) -> dict:
    u2 = _random_unitary(4, seed=seed)
    gates = [
        {"qubits": (0, 1), "unitary": u2},
        {"qubits": (2, 3), "unitary": u2},
    ]
    return {"n_qubits": n, "gates": gates}


def _index_graph(network: TNNetwork):
    """Return {index: count_of_tensors_using_it}."""
    counts = Counter()
    for t in network.tensors:
        for idx in t.indices:
            counts[idx] += 1
    return counts


# ── 3.1 Build from circuit dict ───────────────────────────────────────────────

def test_build_noiseless_tensor_count():
    """n ket tensors + g gate tensors."""
    circuit = _circuit_1q(n=3)
    net = TensorNetworkBuilder.build(circuit)
    # 3 kets + 3 gates
    assert len(net.tensors) == 6


def test_build_2qubit_gate_tensor_count():
    circuit = _circuit_2q(n=4)
    net = TensorNetworkBuilder.build(circuit)
    # 4 kets + 2 gates
    assert len(net.tensors) == 6


# ── 3.2 State tensors ────────────────────────────────────────────────────────

def test_ket_tensors_are_zero_state():
    circuit = _circuit_1q(n=2)
    net = TensorNetworkBuilder.build(circuit)
    kets = [t for t in net.tensors if t.name.startswith("ket_")]
    assert len(kets) == 2
    for ket in kets:
        assert ket.data.shape == (2,)
        assert np.allclose(ket.data, [1, 0])  # |0>


# ── 3.3 Insert mode ──────────────────────────────────────────────────────────

def test_insert_adds_error_nodes():
    circuit = _circuit_1q(n=3)
    # One error operator on gate 0, gate 2
    error_set = {0: _X.copy(), 2: _X.copy()}
    noiseless = TensorNetworkBuilder.build(circuit)
    noisy = TensorNetworkBuilder.build(circuit, error_set=error_set, mode="insert")
    # noisy should have 2 extra tensor nodes
    assert len(noisy.tensors) == len(noiseless.tensors) + 2


def test_insert_changes_topology():
    circuit = _circuit_1q(n=3)
    error_set = {0: _X.copy()}
    noiseless = TensorNetworkBuilder.build(circuit)
    noisy = TensorNetworkBuilder.build(circuit, error_set=error_set, mode="insert")
    assert noiseless.topology_hash() != noisy.topology_hash()


# ── 3.4 Fuse mode ────────────────────────────────────────────────────────────

def test_fuse_preserves_tensor_count():
    circuit = _circuit_1q(n=3)
    error_set = {0: _X.copy(), 1: _I.copy()}
    noiseless = TensorNetworkBuilder.build(circuit)
    fused = TensorNetworkBuilder.build(circuit, error_set=error_set, mode="fuse")
    assert len(fused.tensors) == len(noiseless.tensors)


def test_fuse_preserves_topology_hash():
    circuit = _circuit_1q(n=3)
    error_set = {0: _X.copy(), 1: _X.copy(), 2: _X.copy()}
    noiseless = TensorNetworkBuilder.build(circuit)
    fused = TensorNetworkBuilder.build(circuit, error_set=error_set, mode="fuse")
    assert noiseless.topology_hash() == fused.topology_hash()


def test_fuse_changes_gate_values():
    """Fusing X into a gate tensor must change the data."""
    rng = np.random.default_rng(42)
    u = np.linalg.qr(rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)))[0]
    circuit = {"n_qubits": 1, "gates": [{"qubits": (0,), "unitary": u}]}
    error_set = {0: _X.copy()}
    noiseless = TensorNetworkBuilder.build(circuit)
    fused = TensorNetworkBuilder.build(circuit, error_set=error_set, mode="fuse")
    gate_noiseless = next(t for t in noiseless.tensors if t.name == "gate_0")
    gate_fused = next(t for t in fused.tensors if t.name == "gate_0")
    assert not np.allclose(gate_noiseless.data, gate_fused.data)


# ── 3.5 Index graph structure ─────────────────────────────────────────────────

def test_index_graph_fused_equals_noiseless():
    circuit = _circuit_1q(n=3)
    error_set = {0: _X.copy(), 1: _X.copy(), 2: _X.copy()}
    noiseless = TensorNetworkBuilder.build(circuit)
    fused = TensorNetworkBuilder.build(circuit, error_set=error_set, mode="fuse")
    assert noiseless.index_graph() == fused.index_graph()


def test_insert_index_graph_differs():
    circuit = _circuit_1q(n=3)
    error_set = {0: _X.copy()}
    noiseless = TensorNetworkBuilder.build(circuit)
    inserted = TensorNetworkBuilder.build(circuit, error_set=error_set, mode="insert")
    assert noiseless.index_graph() != inserted.index_graph()


def test_single_qubit_gate_tensor_shape():
    u = _random_unitary(2)
    circuit = {"n_qubits": 1, "gates": [{"qubits": (0,), "unitary": u}]}
    net = TensorNetworkBuilder.build(circuit)
    gate = next(t for t in net.tensors if t.name == "gate_0")
    assert gate.shape == (2, 2), f"Expected (2,2), got {gate.shape}"


def test_two_qubit_gate_tensor_shape():
    u = _random_unitary(4)
    circuit = {"n_qubits": 2, "gates": [{"qubits": (0, 1), "unitary": u}]}
    net = TensorNetworkBuilder.build(circuit)
    gate = next(t for t in net.tensors if t.name == "gate_0")
    assert gate.shape == (2, 2, 2, 2), f"Expected (2,2,2,2), got {gate.shape}"


# ── 3.6 Qiskit IR integration (from_qiskit_circuit) ───────────────────────────
# This is the one code path that actually depends on the installed Qiskit
# version's circuit.data/find_bit/Operator surface (see design.md D2). Neither
# the rest of this suite nor the benchmark circuit generator exercises it, so
# it needs direct coverage to catch Qiskit version drift.

@requires_qiskit
def test_from_qiskit_circuit_tensor_and_gate_counts():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.3, 2)
    qc.cx(1, 2)

    net = TensorNetworkBuilder.from_qiskit_circuit(qc)
    assert net.n_qubits == 3
    assert net.n_gates == 4
    # 3 ket tensors + 4 gate tensors
    assert len(net.tensors) == 7


@requires_qiskit
def test_from_qiskit_circuit_gate_shapes_match_arity():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    net = TensorNetworkBuilder.from_qiskit_circuit(qc)
    gate_0 = next(t for t in net.tensors if t.name == "gate_0")
    gate_1 = next(t for t in net.tensors if t.name == "gate_1")
    assert gate_0.shape == (2, 2)
    assert gate_1.shape == (2, 2, 2, 2)


@requires_qiskit
def test_from_qiskit_circuit_matches_manual_dict_build():
    """A circuit built via Qiskit and the equivalent hand-built dict circuit
    should produce numerically identical tensor networks."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    net_from_qiskit = TensorNetworkBuilder.from_qiskit_circuit(qc)

    manual_circuit = {
        "n_qubits": 2,
        "gates": [
            {"qubits": (0,), "unitary": Operator(qc.data[0].operation).data},
            {"qubits": (0, 1), "unitary": Operator(qc.data[1].operation).data},
        ],
    }
    net_manual = TensorNetworkBuilder.build(manual_circuit)

    assert net_from_qiskit.topology_hash() == net_manual.topology_hash()
    for t1, t2 in zip(net_from_qiskit.tensors, net_manual.tensors):
        assert np.allclose(t1.data, t2.data)


@requires_qiskit
def test_from_qiskit_circuit_with_fused_error_set():
    """UPV fuse mode must work identically whether the noiseless network came
    from a Qiskit circuit or a manual dict circuit."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    error_set = {0: _X.copy()}
    noiseless = TensorNetworkBuilder.from_qiskit_circuit(qc)
    fused = TensorNetworkBuilder.from_qiskit_circuit(qc, error_set=error_set, mode="fuse")

    assert len(fused.tensors) == len(noiseless.tensors)
    assert noiseless.topology_hash() == fused.topology_hash()
    gate_noiseless = next(t for t in noiseless.tensors if t.name == "gate_0")
    gate_fused = next(t for t in fused.tensors if t.name == "gate_0")
    assert not np.allclose(gate_noiseless.data, gate_fused.data)
