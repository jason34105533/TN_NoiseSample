"""ContractionEngine: path finding and batched marginal contraction."""
from __future__ import annotations

import hashlib
import itertools
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .tensor_network import TNNetwork, Tensor

# Optional GPU backend
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

# Optional cuQuantum backend
try:
    import cuquantum.cutensornet as cutn
    HAS_CUTENSORNET = True
except ImportError:
    cutn = None
    HAS_CUTENSORNET = False

# Optional cuQuantum NetworkState backend (real bounded-memory GPU contraction)
try:
    from cuquantum.tensornet.experimental import NetworkState as _NetworkState
    HAS_NETWORK_STATE = True
except ImportError:
    _NetworkState = None
    HAS_NETWORK_STATE = False

# CPU einsum fallback via opt_einsum
try:
    import opt_einsum as oe
    HAS_OPT_EINSUM = True
except ImportError:
    oe = None
    HAS_OPT_EINSUM = False


# ─────────────────────────────────────────────────────────────────────────────
# Path cache
# ─────────────────────────────────────────────────────────────────────────────

class ContractionPathCache:
    """Process-lifetime cache mapping topology_hash -> contraction path."""

    _store: Dict[str, Any] = {}

    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        return cls._store.get(key)

    @classmethod
    def put(cls, key: str, path: Any) -> None:
        cls._store[key] = path

    @classmethod
    def clear(cls) -> None:
        cls._store.clear()

    @classmethod
    def topology_hash(cls, network: TNNetwork) -> str:
        h = hashlib.sha256(network.topology_hash().encode()).hexdigest()
        return h

    @classmethod
    def circuit_topology_hash(cls, circuit: Dict[str, Any]) -> str:
        """Topology hash for a circuit dict, for GPU NetworkState handle caching."""
        parts = [str(circuit["n_qubits"])]
        for gate in circuit["gates"]:
            parts.append(f"{tuple(gate['qubits'])}:{gate['unitary'].shape}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()


class GPUNetworkHandle:
    """A persistent NetworkState plus bookkeeping needed for UPV-style reuse."""

    __slots__ = ("state", "tensor_ids", "coherent_operands", "n_qubits")

    def __init__(self, state, tensor_ids: Dict[int, Any], coherent_operands: Dict[int, np.ndarray], n_qubits: int):
        self.state = state
        self.tensor_ids = tensor_ids
        self.coherent_operands = coherent_operands
        self.n_qubits = n_qubits

    def free(self) -> None:
        self.state.free()


# ─────────────────────────────────────────────────────────────────────────────
# Einsum path (CPU fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _build_einsum_args(tensors: List[Tensor]) -> Tuple[str, List[np.ndarray]]:
    """Build einsum string + operand list for the given tensors."""
    all_indices = {}
    counter = [0]

    def _char(label: str) -> str:
        if label not in all_indices:
            # Use multi-character einsum format via opt_einsum
            all_indices[label] = counter[0]
            counter[0] += 1
        return all_indices[label]

    operands = []
    subscripts = []
    for t in tensors:
        chars = [_char(idx) for idx in t.indices]
        subscripts.append(chars)
        operands.append(t.data)

    # Build output indices: dangling (appear exactly once)
    from collections import Counter
    flat = [c for sub in subscripts for c in sub]
    counts = Counter(flat)
    out_chars = sorted([c for c, n in counts.items() if n == 1])

    # Convert to opt_einsum interleaved format
    # interleaved: [op1, [idx1, idx2, ...], op2, [idx3, ...], ..., [out_indices]]
    interleaved = []
    for op, chars in zip(operands, subscripts):
        interleaved.append(op)
        interleaved.append(chars)
    interleaved.append(out_chars)
    return interleaved, out_chars, all_indices


def _contract_network_cpu(tensors: List[Tensor]) -> np.ndarray:
    """Contract a list of tensors on CPU using opt_einsum or numpy einsum."""
    if not tensors:
        return np.array(1.0 + 0j)

    # Build einsum subscript string
    from collections import Counter
    all_labels = []
    for t in tensors:
        all_labels.extend(t.indices)
    unique_labels = list(dict.fromkeys(all_labels))
    label_to_char = {l: chr(ord('a') + i) for i, l in enumerate(unique_labels)}

    # Verify we don't exceed alphabet
    if len(unique_labels) > 52:
        # Use opt_einsum interleaved format
        return _contract_interleaved(tensors)

    subscript_parts = []
    for t in tensors:
        subscript_parts.append("".join(label_to_char[l] for l in t.indices))

    counts = Counter(all_labels)
    output_labels = [l for l in unique_labels if counts[l] == 1]
    output_chars = "".join(label_to_char[l] for l in output_labels)

    einsum_str = ",".join(subscript_parts) + "->" + output_chars

    if HAS_OPT_EINSUM:
        result = oe.contract(einsum_str, *[t.data for t in tensors])
    else:
        result = np.einsum(einsum_str, *[t.data for t in tensors])

    return np.asarray(result)


def _contract_interleaved(tensors: List[Tensor]) -> np.ndarray:
    """opt_einsum interleaved format for large index sets."""
    from collections import Counter
    all_labels = []
    for t in tensors:
        all_labels.extend(t.indices)
    counts = Counter(all_labels)

    interleaved = []
    for t in tensors:
        interleaved.append(t.data)
        interleaved.append(list(t.indices))
    out_indices = [l for l in dict.fromkeys(all_labels) if counts[l] == 1]
    interleaved.append(out_indices)

    return np.asarray(oe.contract(*interleaved))


# ─────────────────────────────────────────────────────────────────────────────
# Contraction Engine
# ─────────────────────────────────────────────────────────────────────────────

class ContractionEngine:
    """
    Finds contraction paths and executes batched marginal contractions.

    Parameters
    ----------
    batch_size: qubit batch size for non-final batches (b_j)
    final_batch_size: qubit batch size for the final batch (b_f)
    use_gpu: if True and cupy is available, tensors are moved to GPU
    """

    def __init__(
        self,
        batch_size: int = 10,
        final_batch_size: int = 28,
        use_gpu: bool = True,
    ):
        self.batch_size = batch_size
        self.final_batch_size = final_batch_size
        if use_gpu and not (HAS_CUPY and HAS_NETWORK_STATE):
            warnings.warn(
                "use_gpu=True was requested but cupy and/or "
                "cuquantum.tensornet.experimental.NetworkState are not importable; "
                "falling back to the CPU dense-statevector contraction path.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.use_gpu = use_gpu and HAS_CUPY and HAS_NETWORK_STATE
        self.cache = ContractionPathCache()
        # Per-call GPU timing log: (batch_index, is_fixed_pattern_first_seen, elapsed_s).
        # Used by the benchmarking harness to approximate the paper's Fig. 6
        # path-finding-vs-contraction split (see design.md D3): the first
        # compute_reduced_density_matrix() call for a given batch_index pays
        # cuTensorNet's lazy path-finding cost; later calls for the same
        # batch_index reuse the cached plan.
        self.call_log: List[Tuple[int, bool, float]] = []
        self._seen_batch_indices: set = set()

    # ── GPU (NetworkState-backed) construction ─────────────────────────────

    def build_gpu_network(
        self,
        circuit: Dict[str, Any],
        error_set: Optional[Dict[int, np.ndarray]] = None,
        mode: str = "fuse",
        num_hypersamples: int = 1,
        cache: bool = False,
    ) -> GPUNetworkHandle:
        """
        Build a GPU-resident NetworkState for `circuit`.

        mode="noiseless": build once, then use apply_error_set_gpu()/revert_error_set_gpu()
        to reuse this same handle (and its cached cuTensorNet contraction plan) across many
        error sets — this is UPV. mode="fuse": build with `error_set` already fused in, for
        one-shot use (Traditional/Unoptimized). `cache=True` looks up/stores the handle in
        ContractionPathCache keyed by circuit topology (only meaningful for mode="noiseless",
        where different calls for the same circuit topology can share one handle).
        """
        if not HAS_NETWORK_STATE:
            raise RuntimeError("cuquantum.tensornet.experimental.NetworkState is not available")

        key = None
        if cache:
            key = "gpu:" + ContractionPathCache.circuit_topology_hash(circuit)
            cached = ContractionPathCache.get(key)
            if cached is not None:
                return cached

        from .tensor_network import TensorNetworkBuilder

        state, tensor_ids, coherent_operands = TensorNetworkBuilder.build_network_state(
            circuit,
            error_set=error_set,
            mode=mode,
            num_hypersamples=num_hypersamples,
        )
        handle = GPUNetworkHandle(state, tensor_ids, coherent_operands, circuit["n_qubits"])

        if cache and key is not None:
            ContractionPathCache.put(key, handle)

        return handle

    def apply_error_set_gpu(self, handle: GPUNetworkHandle, error_set: Dict[int, np.ndarray]) -> List[int]:
        """Fuse `error_set` into `handle` in place via update_tensor_operator (UPV)."""
        from .tensor_network import TensorNetworkBuilder

        touched: List[int] = []
        for gate_idx, error_op in error_set.items():
            if gate_idx not in handle.tensor_ids:
                continue
            coherent = handle.coherent_operands[gate_idx]
            nq = coherent.ndim // 2
            # Recover the original (2^nq, 2^nq) unitary matrix to re-fuse against.
            d = 2 ** nq
            U = coherent.reshape(d, d)
            fused = (error_op.astype(complex) @ U).reshape(coherent.shape)
            handle.state.update_tensor_operator(handle.tensor_ids[gate_idx], fused, unitary=False)
            touched.append(gate_idx)
        return touched

    def revert_error_set_gpu(self, handle: GPUNetworkHandle, touched_gate_idxs: List[int]) -> None:
        """Restore `handle`'s gates listed in `touched_gate_idxs` to their coherent values."""
        for gate_idx in touched_gate_idxs:
            handle.state.update_tensor_operator(
                handle.tensor_ids[gate_idx], handle.coherent_operands[gate_idx], unitary=False
            )

    def contract_batch_gpu(
        self,
        handle: GPUNetworkHandle,
        batch_index: int,
        num_batches: int,
        prefix: Tuple[int, ...] = (),
    ) -> np.ndarray:
        """GPU-backed equivalent of contract_batch(), bounded to 2**b memory."""
        import time as _time

        is_final = (batch_index == num_batches)
        b = self.final_batch_size if is_final else self.batch_size

        batch_qubit_start = (batch_index - 1) * self.batch_size
        batch_qubits = list(range(batch_qubit_start, min(batch_qubit_start + b, handle.n_qubits)))

        fixed: Dict[int, int] = {}
        for j, marginal_idx in enumerate(prefix):
            start = j * self.batch_size
            qubits_j = list(range(start, start + self.batch_size))
            bits = format(marginal_idx, f"0{len(qubits_j)}b")
            for q, bit_ch in zip(qubits_j, bits):
                fixed[q] = int(bit_ch)

        is_first_for_pattern = batch_index not in self._seen_batch_indices
        self._seen_batch_indices.add(batch_index)

        t0 = _time.perf_counter()
        rdm = handle.state.compute_reduced_density_matrix(
            tuple(batch_qubits), fixed=fixed, diagonal=True
        )
        marginal = np.asarray(cp.asnumpy(rdm) if HAS_CUPY else rdm).real.reshape(-1)
        elapsed = _time.perf_counter() - t0
        self.call_log.append((batch_index, is_first_for_pattern, elapsed))

        norm = marginal.sum()
        if norm > 0:
            marginal = marginal / norm
        return marginal

    def find_path(self, network: TNNetwork, num_hypersamples: int = 1) -> Any:
        """Find (or retrieve cached) contraction path for the network."""
        key = ContractionPathCache.topology_hash(network)
        cached = ContractionPathCache.get(key)
        if cached is not None:
            return cached

        if HAS_CUTENSORNET and self.use_gpu:
            path = self._find_path_cutensornet(network, num_hypersamples)
        else:
            path = self._find_path_opt_einsum(network)

        ContractionPathCache.put(key, path)
        return path

    def contract_batch(
        self,
        network: TNNetwork,
        path: Any,
        batch_index: int,
        num_batches: int,
        prefix: Tuple[int, ...] = (),
    ) -> np.ndarray:
        """
        Contract `batch_size` qubits at `batch_index`, conditioned on `prefix`.

        Parameters
        ----------
        network: the tensor network
        path: contraction path (from find_path)
        batch_index: 1-based batch number (1..num_batches)
        num_batches: total number of batches f
        prefix: tuple of already-sampled bit values (s_1, ..., s_{j-1})

        Returns
        -------
        marginal: real numpy array of shape (2**b,) summing to ≤ 1.0
        """
        if isinstance(network, GPUNetworkHandle):
            return self.contract_batch_gpu(network, batch_index, num_batches, prefix=prefix)

        is_final = (batch_index == num_batches)
        b = self.final_batch_size if is_final else self.batch_size

        # Determine which qubits are in this batch
        batch_start = sum(
            self.batch_size if i < num_batches - 1 else self.final_batch_size
            for i in range(batch_index - 1)
        )
        # Simpler: just track via batch_index
        batch_qubit_start = (batch_index - 1) * self.batch_size
        batch_qubits = list(range(batch_qubit_start, min(batch_qubit_start + b, network.n_qubits)))

        # Build the full amplitude tensor by contracting all tensors
        # then marginalise over the batch qubits conditioned on prefix
        marginal = self._compute_marginal(network, batch_qubits, prefix)
        return marginal

    def _compute_marginal(
        self,
        network: TNNetwork,
        batch_qubits: List[int],
        prefix: Tuple[int, ...],
    ) -> np.ndarray:
        """
        Compute the marginal probability over `batch_qubits` conditioned on `prefix`.

        This CPU reference implementation contracts the full amplitude tensor then
        slices and marginalises. On GPU, cuTensorNet contractions replace this.

        The amplitude tensor has one axis per qubit in sorted qubit order.
        `prefix` is a tuple of marginal *indices* from previous batches, where
        each index encodes the joint bit value of that batch's qubits.
        `batch_qubits` lists which qubit indices belong to the current batch.
        """
        from collections import Counter

        # 1. Contract network → amplitude tensor
        amp = _contract_network_cpu(network.tensors)
        if amp.ndim == 0:
            return np.array([float(abs(amp) ** 2)])

        # 2. Determine axis ordering: sort open wire labels by qubit index
        all_idx = []
        for t in network.tensors:
            all_idx.extend(t.indices)
        counts = Counter(all_idx)
        open_indices = [l for l in dict.fromkeys(all_idx) if counts[l] == 1]

        def _qubit_of(label: str) -> int:
            return int(label.split("_")[1])

        open_qubits = [_qubit_of(l) for l in open_indices]
        # Sort axes so axis i corresponds to the i-th smallest open qubit
        sorted_order = sorted(range(len(open_qubits)), key=lambda i: open_qubits[i])
        sorted_qubits = [open_qubits[i] for i in sorted_order]
        amp = amp.transpose(sorted_order)

        # 3. prob[b0, b1, ..., b_{n-1}] = |amplitude|^2
        prob = (amp * amp.conj()).real

        # 4. Apply prefix conditioning by fixing already-measured qubits.
        #    prefix[j] is the integer index into the 2^(batch_size) marginal of batch j+1.
        #    Decode each prefix element into individual bit values, then slice those axes.
        axis_to_fix: Dict[int, int] = {}  # {qubit_idx: bit_value}
        prev_qubits: List[int] = []
        for j, marginal_idx in enumerate(prefix):
            # Qubits in batch j (0-based):
            start = j * self.batch_size
            end = start + self.batch_size
            b_qubits = [q for q in sorted_qubits if start <= sorted_qubits.index(q) < end
                        if q not in batch_qubits]
            # Simpler: enumerate which qubits were covered by prefix batch j
            batch_j_qubits = sorted_qubits[j * self.batch_size: (j + 1) * self.batch_size]
            bits = format(marginal_idx, f"0{len(batch_j_qubits)}b")
            for q, bit_ch in zip(batch_j_qubits, bits):
                axis_to_fix[q] = int(bit_ch)

        # Build slice tuple (qubit axis order)
        slices = []
        for q in sorted_qubits:
            if q in axis_to_fix:
                slices.append(axis_to_fix[q])
            else:
                slices.append(slice(None))
        prob = prob[tuple(slices)]

        if prob.ndim == 0:
            return np.array([float(prob)])

        # 5. After slicing, the remaining axes correspond to unfixed qubits.
        #    Map batch_qubits to their post-slice axis positions.
        remaining_qubits = [q for q in sorted_qubits if q not in axis_to_fix]
        batch_axes_after_slice = [
            remaining_qubits.index(q) for q in batch_qubits if q in remaining_qubits
        ]

        # 6. Sum out all axes not in batch_axes_after_slice
        all_axes = list(range(prob.ndim))
        axes_to_sum = sorted(
            [ax for ax in all_axes if ax not in batch_axes_after_slice], reverse=True
        )
        for ax in axes_to_sum:
            prob = prob.sum(axis=ax)

        marginal = np.asarray(prob).flatten().real
        norm = marginal.sum()
        if norm > 0:
            marginal = marginal / norm
        return marginal

    def exhaustive_final_batch(
        self,
        network: TNNetwork,
        path: Any,
        prefix: Tuple[int, ...],
        min_prob: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Contract the final batch and return all bitstrings with prob >= min_prob.

        Returns list of (bitstring_suffix, probability) tuples.
        """
        b = self.final_batch_size
        batch_index = self._num_batches(network)
        marginal = self.contract_batch(
            network, path, batch_index, batch_index, prefix=prefix
        )

        results = []
        for idx, prob in enumerate(marginal):
            if float(prob) >= min_prob:
                bits = format(idx, f"0{b}b")[: len(marginal).bit_length() - 1]
                bits = format(idx, f"0{max(1, len(marginal) - 1)}b")
                results.append((bits, float(prob)))
        return results

    def _num_batches(self, network: TNNetwork) -> int:
        n = network.n_qubits
        non_final = (n - self.final_batch_size + self.batch_size - 1) // self.batch_size
        return max(1, non_final + 1)

    def _find_path_opt_einsum(self, network: TNNetwork) -> Any:
        if not HAS_OPT_EINSUM:
            return None  # numpy.einsum will be used directly
        from collections import Counter
        all_labels = []
        for t in network.tensors:
            all_labels.extend(t.indices)
        counts = Counter(all_labels)

        interleaved = []
        for t in network.tensors:
            interleaved.append(t.data)
            interleaved.append(list(t.indices))
        out_indices = [l for l in dict.fromkeys(all_labels) if counts[l] == 1]
        interleaved.append(out_indices)

        try:
            path, _ = oe.contract_path(*interleaved, optimize="greedy")
        except Exception:
            path = None
        return path

    def _find_path_cutensornet(self, network: TNNetwork, num_hypersamples: int) -> Any:
        # cuTensorNet path finding — requires GPU tensors
        # Stub: falls back to opt_einsum path for now
        return self._find_path_opt_einsum(network)
