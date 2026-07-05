"""BaseTNSimulator abstract interface."""
from __future__ import annotations

import abc
from typing import List, Tuple


class BaseTNSimulator(abc.ABC):
    """
    Abstract base for all three trajectory simulator phases.

    `sample()` returns a list of (bitstring, error_set_id) tuples.
    - bitstring: length-n string of '0'/'1' characters
    - error_set_id: integer index of the error set that produced this shot
    """

    @abc.abstractmethod
    def sample(
        self,
        circuit: dict,
        noise_model,
        num_shots: int,
    ) -> List[Tuple[str, int]]:
        """Simulate the noisy circuit and return `num_shots` labeled bitstrings."""
        ...
