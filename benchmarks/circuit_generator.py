"""Random circuit generator for PTSBE benchmarks."""
from __future__ import annotations

from typing import List
import numpy as np

from tn_noise_sim.noise_model import NoiseModel, GateNoiseSpec, ErrorType


def random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    u, _ = np.linalg.qr(z)
    return u


def generate_circuit(
    n: int,
    g: int,
    seed: int = 0,
    two_qubit_fraction: float = 0.3,
) -> dict:
    """
    Generate a random n-qubit, g-gate circuit.

    Gates are random unitaries; ~(1-two_qubit_fraction) are single-qubit and
    ~two_qubit_fraction are two-qubit gates (placed on adjacent qubits).
    Error probabilities are drawn uniformly from [0.02, 0.20] per gate,
    matching the paper's experimental setup.

    Returns circuit dict: {"n_qubits": n, "gates": [...], "noise_model": NoiseModel}
    """
    rng = np.random.default_rng(seed)
    gates = []
    gate_specs = {}

    for gate_idx in range(g):
        use_2q = (n >= 2) and (rng.random() < two_qubit_fraction)
        error_prob = float(rng.uniform(0.02, 0.20))

        if use_2q:
            q0 = int(rng.integers(0, n - 1))
            q1 = q0 + 1
            u = random_unitary(4, rng)
            gates.append({"qubits": (q0, q1), "unitary": u})
            gate_specs[gate_idx] = GateNoiseSpec(ErrorType.DEPOLARIZING, error_prob)
        else:
            q = int(rng.integers(0, n))
            u = random_unitary(2, rng)
            gates.append({"qubits": (q,), "unitary": u})
            gate_specs[gate_idx] = GateNoiseSpec(ErrorType.PAULI, error_prob)

    noise_model = NoiseModel(gate_specs=gate_specs)
    return {"n_qubits": n, "gates": gates, "noise_model": noise_model}


def generate_ensemble(
    n: int,
    g: int,
    num_instances: int,
    base_seed: int = 0,
) -> List[dict]:
    """Generate `num_instances` random circuits with reproducible seeds."""
    return [generate_circuit(n, g, seed=base_seed + i) for i in range(num_instances)]
