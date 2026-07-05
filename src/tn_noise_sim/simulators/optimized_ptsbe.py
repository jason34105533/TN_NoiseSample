"""Phase 3: Optimized TN PTSBE Simulator — UPV + NBS."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from .base import BaseTNSimulator
from ..noise_model import NoiseModel
from ..error_sampling import ErrorSampler
from ..tensor_network import TensorNetworkBuilder, TNNetwork
from ..contraction import ContractionEngine, ContractionPathCache


class OptimizedPTSBESimulator(BaseTNSimulator):
    """
    Phase 3: Unified Path Variations (UPV) + Non-Degenerate Batched Sampling (NBS).

    Key innovations vs Phase 2:
    - One path-find on the noiseless network, cached and reused for all E error sets.
    - Non-proportional mode: exhaustive final-batch sampling for maximum throughput.
    - Proportional mode: conditional marginal branching to match Born rule statistics.
    - Flexible per-batch and final-batch size parameters.
    """

    def __init__(
        self,
        batch_size: int = 10,
        final_batch_size: int = 28,
        num_hypersamples: int = 100,
        num_error_sets: Optional[int] = None,
        min_prob: float = 0.0,
        rng_seed: Optional[int] = None,
    ):
        if final_batch_size < 1:
            raise ValueError("final_batch_size must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = batch_size
        self.final_batch_size = final_batch_size
        self.num_hypersamples = num_hypersamples
        self.num_error_sets = num_error_sets
        self.min_prob = min_prob
        self.rng_seed = rng_seed
        self._path_find_count = 0

    def sample(
        self,
        circuit: dict,
        noise_model: NoiseModel,
        num_shots: int,
        mode: str = "non_proportional",
    ) -> List[Tuple[str, int]]:
        n = circuit["n_qubits"]
        if self.final_batch_size > n:
            raise ValueError(
                f"final_batch_size ({self.final_batch_size}) > n_qubits ({n})"
            )

        num_gates = len(circuit["gates"])
        E = self.num_error_sets if self.num_error_sets is not None else max(1, num_shots // 10)

        sampler = ErrorSampler(noise_model, num_gates, rng_seed=self.rng_seed)
        sampler.sample(
            num_error_sets=E,
            total_shots=num_shots,
            mode="proportional" if mode == "proportional" else "non_proportional",
        )
        error_sets_with_counts = sampler.to_list()

        engine = ContractionEngine(
            batch_size=self.batch_size,
            final_batch_size=self.final_batch_size,
            use_gpu=False,
        )

        # ── UPV: find path once on the NOISELESS network ──────────────────────
        noiseless_network = TensorNetworkBuilder.build(circuit, mode="noiseless")
        self._path_find_count = 0
        path = engine.find_path(noiseless_network, num_hypersamples=self.num_hypersamples)
        self._path_find_count = 1

        rng = np.random.default_rng(self.rng_seed)
        results: List[Tuple[str, int]] = []

        for set_idx, (error_set, shot_count) in enumerate(error_sets_with_counts):
            # Fuse error operators into gate tensors — topology preserved, path reused
            fused_network = TensorNetworkBuilder.build(
                circuit, error_set=error_set, mode="fuse"
            )

            if mode == "non_proportional":
                shots = self._sample_non_proportional(
                    engine, fused_network, path, n, shot_count, rng
                )
            else:
                shots = self._sample_proportional(
                    engine, fused_network, path, n, shot_count, rng
                )

            results.extend((bs, set_idx) for bs in shots)

        return results

    # ── Non-proportional NBS ──────────────────────────────────────────────────

    def _sample_non_proportional(
        self,
        engine: ContractionEngine,
        network: TNNetwork,
        path,
        n: int,
        shot_count: int,
        rng: np.random.Generator,
    ) -> List[str]:
        num_batches = engine._num_batches(network)
        # Start with one empty prefix
        prefixes: List[Tuple[int, ...]] = [()]
        all_bitstrings: List[str] = []

        for batch_idx in range(1, num_batches + 1):
            is_final = (batch_idx == num_batches)
            next_prefixes: List[Tuple[int, ...]] = []

            for prefix in prefixes:
                marginal = engine.contract_batch(
                    network, path, batch_idx, num_batches, prefix=prefix
                )
                marginal = np.maximum(marginal, 0)
                s = marginal.sum()
                probs = marginal / s if s > 0 else np.ones(len(marginal)) / len(marginal)

                if is_final:
                    # Exhaustive: harvest all entries above min_prob
                    b = engine.final_batch_size
                    prefix_str = self._prefix_to_bits(prefix, batch_idx - 1, engine)
                    for idx, prob in enumerate(probs):
                        if float(prob) >= self.min_prob:
                            suffix = format(idx, f"0{b}b")
                            full = (prefix_str + suffix)[:n]
                            all_bitstrings.append(full)
                else:
                    # Sample one branch per prefix (can be extended to k > 1)
                    chosen = int(rng.choice(len(probs), p=probs))
                    next_prefixes.append(prefix + (chosen,))

            if not is_final:
                prefixes = next_prefixes

        # If exhaustive yielded nothing, fall back to at least shot_count random draws
        if not all_bitstrings:
            for _ in range(shot_count):
                bits = "".join(rng.choice(["0", "1"]) for _ in range(n))
                all_bitstrings.append(bits)

        return all_bitstrings

    # ── Proportional NBS ─────────────────────────────────────────────────────

    def _sample_proportional(
        self,
        engine: ContractionEngine,
        network: TNNetwork,
        path,
        n: int,
        shot_count: int,
        rng: np.random.Generator,
    ) -> List[str]:
        num_batches = engine._num_batches(network)

        # Each element: (prefix_tuple, count_of_shots_with_this_prefix)
        prefix_counts: Dict[Tuple[int, ...], int] = {(): shot_count}
        final_bitstrings: List[str] = []

        for batch_idx in range(1, num_batches + 1):
            is_final = (batch_idx == num_batches)
            new_prefix_counts: Dict[Tuple[int, ...], int] = defaultdict(int)

            for prefix, count in prefix_counts.items():
                if count == 0:
                    continue
                marginal = engine.contract_batch(
                    network, path, batch_idx, num_batches, prefix=prefix
                )
                marginal = np.maximum(marginal, 0)
                s = marginal.sum()
                probs = marginal / s if s > 0 else np.ones(len(marginal)) / len(marginal)

                # Sample `count` continuations from this marginal
                choices = rng.choice(len(probs), size=count, p=probs)

                if is_final:
                    b = engine.final_batch_size
                    prefix_str = self._prefix_to_bits(prefix, batch_idx - 1, engine)
                    for chosen in choices:
                        suffix = format(int(chosen), f"0{b}b")
                        full = (prefix_str + suffix)[:n]
                        final_bitstrings.append(full)
                else:
                    for chosen in choices:
                        new_prefix_counts[prefix + (int(chosen),)] += 1

            if not is_final:
                prefix_counts = dict(new_prefix_counts)

        return final_bitstrings

    def _prefix_to_bits(
        self,
        prefix: Tuple[int, ...],
        num_prev_batches: int,
        engine: ContractionEngine,
    ) -> str:
        parts = []
        for i, chosen in enumerate(prefix):
            b = engine.batch_size  # non-final batches all use batch_size
            parts.append(format(chosen, f"0{b}b"))
        return "".join(parts)
