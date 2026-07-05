"""Phase 2: Unoptimized PTSBE Simulator."""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from .base import BaseTNSimulator
from ..noise_model import NoiseModel
from ..error_sampling import ErrorSampler
from ..tensor_network import TensorNetworkBuilder
from ..contraction import ContractionEngine, ContractionPathCache


class UnoptimizedPTSBESimulator(BaseTNSimulator):
    """
    Phase 2: Pre-samples E error sets upfront, finds contraction path once per
    error set (not per shot), still extracts shots sequentially one at a time.
    """

    def __init__(
        self,
        batch_size: int = 24,
        num_error_sets: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.num_error_sets = num_error_sets
        self.rng_seed = rng_seed
        self._path_find_count = 0

    def sample(
        self,
        circuit: dict,
        noise_model: NoiseModel,
        num_shots: int,
    ) -> List[Tuple[str, int]]:
        n = circuit["n_qubits"]
        num_gates = len(circuit["gates"])

        # Default E = total_shots (one error set per shot — matches traditional behavior)
        E = self.num_error_sets if self.num_error_sets is not None else num_shots

        # Pre-sample all E error sets
        sampler = ErrorSampler(noise_model, num_gates, rng_seed=self.rng_seed)
        sampler.sample(num_error_sets=E, total_shots=num_shots, mode="non_proportional")
        error_sets_with_counts = sampler.to_list()

        engine = ContractionEngine(
            batch_size=self.batch_size,
            final_batch_size=self.batch_size,
            use_gpu=False,
        )
        rng = np.random.default_rng(self.rng_seed)
        results: List[Tuple[str, int]] = []
        self._path_find_count = 0

        for set_idx, (error_set, shot_count) in enumerate(error_sets_with_counts):
            # Build network with errors inserted (changes topology per error set)
            network = TensorNetworkBuilder.build(circuit, error_set=error_set, mode="insert")

            # Find path ONCE per error set (not per shot)
            path = engine.find_path(network, num_hypersamples=1)
            self._path_find_count += 1

            # Extract shot_count shots sequentially, one per contraction pass
            for _ in range(shot_count):
                bitstring = self._sample_one_bitstring(engine, network, path, n, rng)
                results.append((bitstring, set_idx))

        return results

    def _sample_one_bitstring(self, engine, network, path, n, rng) -> str:
        num_batches = engine._num_batches(network)
        bits: List[str] = []
        prefix: Tuple[int, ...] = ()

        for batch_idx in range(1, num_batches + 1):
            marginal = engine.contract_batch(network, path, batch_idx, num_batches, prefix=prefix)
            marginal = np.maximum(marginal, 0)
            s = marginal.sum()
            probs = marginal / s if s > 0 else np.ones(len(marginal)) / len(marginal)

            chosen = int(rng.choice(len(probs), p=probs))
            b = engine.batch_size if batch_idx < num_batches else engine.final_batch_size
            b = min(b, n - len(bits))
            bits.append(format(chosen, f"0{b}b"))
            prefix = prefix + (chosen,)

        return "".join(bits)[:n]
