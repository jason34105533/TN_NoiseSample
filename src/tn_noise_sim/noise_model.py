"""NoiseModel: per-gate error channel specifications."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Optional


class ErrorType(enum.Enum):
    PAULI = "pauli"
    DEPOLARIZING = "depolarizing"


@dataclass
class GateNoiseSpec:
    error_type: ErrorType
    probability: float  # total error probability in [0, 1]

    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0,1], got {self.probability}")


@dataclass
class NoiseModel:
    """Maps gate index -> GateNoiseSpec for a quantum circuit."""
    gate_specs: Dict[int, GateNoiseSpec] = field(default_factory=dict)
    default_spec: Optional[GateNoiseSpec] = None

    def get(self, gate_idx: int) -> Optional[GateNoiseSpec]:
        return self.gate_specs.get(gate_idx, self.default_spec)

    @classmethod
    def uniform_pauli(cls, num_gates: int, probability: float) -> "NoiseModel":
        specs = {i: GateNoiseSpec(ErrorType.PAULI, probability) for i in range(num_gates)}
        return cls(gate_specs=specs)

    @classmethod
    def uniform_depolarizing(cls, num_gates: int, probability: float) -> "NoiseModel":
        specs = {i: GateNoiseSpec(ErrorType.DEPOLARIZING, probability) for i in range(num_gates)}
        return cls(gate_specs=specs)
