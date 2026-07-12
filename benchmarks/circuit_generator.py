"""Random circuit generator for PTSBE benchmarks.

Matches the reference paper's circuit-generation procedure (arXiv:2604.08467,
Sec. IV-B): single-qubit gates drawn from {H, X, Y, Z, T, Rx} and two-qubit
nearest-neighbor gates drawn from {CX, CY, CZ, CH, CRx}, with 20% of gates
being two-qubit by default. Noise channels are Pauli (X/Y/Z) for single-qubit
gates and two-qubit depolarizing for two-qubit gates, with per-gate error
probability drawn uniformly from [0.02, 0.20].
"""
from __future__ import annotations

from typing import List
import numpy as np

from tn_noise_sim.noise_model import NoiseModel, GateNoiseSpec, ErrorType

_I2 = np.eye(2, dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def _rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _controlled(u: np.ndarray) -> np.ndarray:
    """Build a 4x4 controlled-U (control=first qubit) in |q0 q1> basis order."""
    c = np.eye(4, dtype=complex)
    c[2:4, 2:4] = u
    return c


_SINGLE_QUBIT_FIXED = {"H": _H, "X": _X, "Y": _Y, "Z": _Z, "T": _T}
_SINGLE_QUBIT_NAMES = ["H", "X", "Y", "Z", "T", "Rx"]
_TWO_QUBIT_FIXED = {"CX": _controlled(_X), "CY": _controlled(_Y), "CZ": _controlled(_Z), "CH": _controlled(_H)}
_TWO_QUBIT_NAMES = ["CX", "CY", "CZ", "CH", "CRx"]


def _random_single_qubit_gate(rng: np.random.Generator) -> np.ndarray:
    name = rng.choice(_SINGLE_QUBIT_NAMES)
    if name == "Rx":
        theta = float(rng.uniform(0, 2 * np.pi))
        return _rx(theta)
    return _SINGLE_QUBIT_FIXED[name].copy()


def _random_two_qubit_gate(rng: np.random.Generator) -> np.ndarray:
    name = rng.choice(_TWO_QUBIT_NAMES)
    if name == "CRx":
        theta = float(rng.uniform(0, 2 * np.pi))
        return _controlled(_rx(theta))
    return _TWO_QUBIT_FIXED[name].copy()


def generate_circuit(
    n: int,
    g: int,
    seed: int = 0,
    two_qubit_fraction: float = 0.2,
) -> dict:
    """
    Generate a random n-qubit, g-gate circuit matching the paper's setup.

    Single-qubit gates are drawn from {H, X, Y, Z, T, Rx}; two-qubit gates are
    drawn from {CX, CY, CZ, CH, CRx} and act on nearest-neighbor qubit pairs.
    ~two_qubit_fraction of gates are two-qubit (default 20%, per the paper).
    Error probabilities are drawn uniformly from [0.02, 0.20] per gate.

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
            u = _random_two_qubit_gate(rng)
            gates.append({"qubits": (q0, q1), "unitary": u})
            gate_specs[gate_idx] = GateNoiseSpec(ErrorType.DEPOLARIZING, error_prob)
        else:
            q = int(rng.integers(0, n))
            u = _random_single_qubit_gate(rng)
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
