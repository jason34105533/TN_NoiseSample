"""ErrorSampler: pre-trajectory error set sampling (proportional and non-proportional)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .noise_model import ErrorType, GateNoiseSpec, NoiseModel

# ----- Pauli matrices -----
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI_1Q = [_I, _X, _Y, _Z]  # indices 0=I, 1=X, 2=Y, 3=Z

# All 15 non-identity two-qubit Paulis (tensor products, excluding II)
_PAULI_2Q_NONID: List[np.ndarray] = []
for _a in [_I, _X, _Y, _Z]:
    for _b in [_I, _X, _Y, _Z]:
        _op = np.kron(_a, _b)
        if not np.allclose(_op, np.eye(4)):
            _PAULI_2Q_NONID.append(_op)
assert len(_PAULI_2Q_NONID) == 15


def _sample_pauli_1q(spec: GateNoiseSpec, rng: np.random.Generator) -> np.ndarray:
    p = spec.probability
    # p split equally among X, Y, Z; identity gets 1-p
    probs = [1.0 - p, p / 3, p / 3, p / 3]
    idx = rng.choice(4, p=probs)
    return _PAULI_1Q[idx].copy()


def _sample_depolarizing_2q(spec: GateNoiseSpec, rng: np.random.Generator) -> np.ndarray:
    p = spec.probability
    # identity with prob (1-p), each of 15 non-id Paulis with prob p/15
    if rng.random() < p:
        idx = rng.integers(0, 15)
        return _PAULI_2Q_NONID[idx].copy()
    return np.eye(4, dtype=complex)


def _gate_weight(spec: GateNoiseSpec) -> float:
    """Return the probability that this gate is in error (for proportional sampling weight)."""
    return spec.probability


# Type alias: an error set is {gate_idx: error_operator_ndarray}
ErrorSet = Dict[int, np.ndarray]


class ErrorSampler:
    """
    Pre-samples E error sets for PTSBE trajectory simulation.

    Parameters
    ----------
    noise_model: NoiseModel
    num_gates: int — total number of gates in the circuit
    rng_seed: optional seed for reproducibility
    """

    def __init__(
        self,
        noise_model: NoiseModel,
        num_gates: int,
        rng_seed: Optional[int] = None,
    ):
        self.noise_model = noise_model
        self.num_gates = num_gates
        self.rng = np.random.default_rng(rng_seed)

        self._error_sets: List[ErrorSet] = []
        self._shot_counts: List[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        num_error_sets: int,
        total_shots: int,
        mode: str = "proportional",
    ) -> "ErrorSampler":
        """
        Pre-sample `num_error_sets` error sets and assign shot counts.

        Parameters
        ----------
        num_error_sets: E
        total_shots: m — total shots to distribute
        mode: "proportional" | "non_proportional"

        Returns self for chaining.
        """
        if mode == "proportional":
            self._sample_proportional(num_error_sets, total_shots)
        elif mode == "non_proportional":
            self._sample_non_proportional(num_error_sets, total_shots)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
        return self

    def to_list(self) -> List[Tuple[ErrorSet, int]]:
        """Return [(error_set_dict, shot_count), ...] for all E error sets."""
        return list(zip(self._error_sets, self._shot_counts))

    @property
    def error_sets(self) -> List[ErrorSet]:
        return self._error_sets

    @property
    def shot_counts(self) -> List[int]:
        return self._shot_counts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_one_error_set(self) -> ErrorSet:
        """Draw one error set: independently sample each gate's error operator."""
        error_set: ErrorSet = {}
        for gate_idx in range(self.num_gates):
            spec = self.noise_model.get(gate_idx)
            if spec is None:
                continue
            if spec.error_type == ErrorType.PAULI:
                op = _sample_pauli_1q(spec, self.rng)
            elif spec.error_type == ErrorType.DEPOLARIZING:
                op = _sample_depolarizing_2q(spec, self.rng)
            else:
                raise ValueError(f"Unknown error type: {spec.error_type}")
            error_set[gate_idx] = op
        return error_set

    def _error_set_weight(self, error_set: ErrorSet) -> float:
        """
        Compute the sampling weight of an error set as the product of per-gate
        probabilities (identity gates contribute 1 - p, error gates contribute p/k).
        """
        weight = 1.0
        for gate_idx in range(self.num_gates):
            spec = self.noise_model.get(gate_idx)
            if spec is None:
                continue
            op = error_set.get(gate_idx)
            if op is None:
                continue
            if spec.error_type == ErrorType.PAULI:
                if np.allclose(op, _I):
                    weight *= 1.0 - spec.probability
                else:
                    weight *= spec.probability / 3.0
            elif spec.error_type == ErrorType.DEPOLARIZING:
                if np.allclose(op, np.eye(4)):
                    weight *= 1.0 - spec.probability
                else:
                    weight *= spec.probability / 15.0
        return weight

    def _sample_proportional(self, num_error_sets: int, total_shots: int):
        sets = [self._draw_one_error_set() for _ in range(num_error_sets)]
        weights = np.array([self._error_set_weight(s) for s in sets], dtype=float)
        weight_sum = weights.sum()
        if weight_sum == 0:
            weights = np.ones(num_error_sets, dtype=float) / num_error_sets
        else:
            weights /= weight_sum

        # Assign shot counts proportional to weights, preserving total
        raw = weights * total_shots
        counts = np.floor(raw).astype(int)
        remainder = total_shots - counts.sum()
        # Distribute remaining shots to sets with largest fractional parts
        fracs = raw - counts
        top_indices = np.argsort(-fracs)[:remainder]
        counts[top_indices] += 1

        self._error_sets = sets
        self._shot_counts = counts.tolist()

    def _sample_non_proportional(self, num_error_sets: int, total_shots: int):
        sets = [self._draw_one_error_set() for _ in range(num_error_sets)]
        base = total_shots // num_error_sets
        extra = total_shots % num_error_sets
        counts = [base + (1 if i < extra else 0) for i in range(num_error_sets)]

        self._error_sets = sets
        self._shot_counts = counts
