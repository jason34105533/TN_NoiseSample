"""Phase 1: Traditional Trajectory Simulator."""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from .base import BaseTNSimulator
from ..noise_model import NoiseModel
from ..error_sampling import ErrorSampler
from ..tensor_network import TensorNetworkBuilder
from ..contraction import ContractionEngine, ContractionPathCache


class TraditionalTrajectorySimulator(BaseTNSimulator):
    """
    Baseline: per-shot CPU path finding, sequential single-shot extraction,
    fixed batch size (default 24), no path caching.
    """

    def __init__(self, batch_size: int = 24, rng_seed: Optional[int] = None):
        self.batch_size = batch_size
        self.rng_seed = rng_seed
        self._path_find_count = 0  # for test verification

    def sample(
        self,
        circuit: dict,
        noise_model: NoiseModel,
        num_shots: int,
    ) -> List[Tuple[str, int]]:
        n = circuit["n_qubits"]
        num_gates = len(circuit["gates"])
        results: List[Tuple[str, int]] = []
        self._path_find_count = 0

        engine = ContractionEngine(
            batch_size=self.batch_size,
            final_batch_size=self.batch_size,
            use_gpu=False,
        )
        rng = np.random.default_rng(self.rng_seed)

        for shot_idx in range(num_shots):
            # Sample one error set per shot
            sampler = ErrorSampler(noise_model, num_gates, rng_seed=int(rng.integers(0, 2**31)))
            sampler.sample(num_error_sets=1, total_shots=1, mode="non_proportional")
            error_set, _ = sampler.to_list()[0]

            # Build noisy network with error operators inserted
            network = TensorNetworkBuilder.build(circuit, error_set=error_set, mode="insert")

            # Find path (no caching — fresh every shot)
            ContractionPathCache.clear()
            path = engine.find_path(network, num_hypersamples=1)
            ContractionPathCache.clear()
            self._path_find_count += 1

            # Contract all batches and sample one bitstring
            bitstring = self._sample_one_bitstring(engine, network, path, n, rng)
            results.append((bitstring, shot_idx))

        return results

    def _sample_one_bitstring(
        self,
        engine: ContractionEngine,
        network,
        path,
        n: int,
        rng: np.random.Generator,
    ) -> str:
        num_batches = engine._num_batches(network)
        bits: List[str] = []
        prefix: Tuple[int, ...] = ()

        for batch_idx in range(1, num_batches + 1):
            marginal = engine.contract_batch(network, path, batch_idx, num_batches, prefix=prefix)
            marginal = np.maximum(marginal, 0)
            s = marginal.sum()
            if s <= 0:
                probs = np.ones(len(marginal)) / len(marginal)
            else:
                probs = marginal / s

            chosen = int(rng.choice(len(probs), p=probs))
            b = engine.batch_size if batch_idx < num_batches else engine.final_batch_size
            b = min(b, n - len(bits))
            bits.append(format(chosen, f"0{b}b"))
            prefix = prefix + (chosen,)

        return "".join(bits)[:n]
