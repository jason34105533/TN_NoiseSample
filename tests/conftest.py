import numpy as np
import pytest

try:
    from qiskit import QuantumCircuit
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    import cupy  # noqa: F401
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

requires_qiskit = pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="cupy/GPU not available")


def make_small_circuit(n_qubits=5, n_gates=10, seed=42):
    """Build a deterministic small test circuit without Qiskit."""
    rng = np.random.default_rng(seed)
    # Returns a plain dict representation for CPU-only tests
    gates = []
    for _ in range(n_gates):
        q = int(rng.integers(0, n_qubits))
        # Random unitary via QR decomposition
        z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        u, _ = np.linalg.qr(z)
        gates.append({"qubits": (q,), "unitary": u})
    return {"n_qubits": n_qubits, "gates": gates}


@pytest.fixture
def small_circuit_dict():
    return make_small_circuit(n_qubits=5, n_gates=10, seed=42)


@pytest.fixture
def tiny_circuit_dict():
    """3-qubit, 3-gate circuit for fast contraction tests."""
    return make_small_circuit(n_qubits=3, n_gates=3, seed=0)


@pytest.fixture
def simple_noise_model():
    from tn_noise_sim.noise_model import NoiseModel, GateNoiseSpec, ErrorType
    specs = {i: GateNoiseSpec(error_type=ErrorType.PAULI, probability=0.05)
             for i in range(10)}
    return NoiseModel(gate_specs=specs)
