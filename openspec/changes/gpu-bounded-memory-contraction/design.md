## Context

Current state (`src/tn_noise_sim/contraction.py`):
- `ContractionEngine._find_path_cutensornet()` is a stub that falls back to CPU `opt_einsum` path-finding.
- `_compute_marginal()` contracts the network's *entire* tensor list into a dense `2^n`-entry amplitude array (via `opt_einsum`/`numpy.einsum` on host memory), then slices/sums it down to the requested `2^b` marginal. Peak memory is `O(2^n)`, not `O(2^b)`.
- `TensorNetworkBuilder.build(..., mode="fuse")` already produces per-gate tensors with the error operator matrix-multiplied in (`K @ U`), in application order, with a documented index convention: single-qubit gate tensors are shape `(2,2)` = `[out, in]`; two-qubit gate tensors are shape `(2,2,2,2)` = `[out0, out1, in0, in1]`. This convention was chosen independently of any particular contraction backend.
- `OptimizedPTSBESimulator` (UPV) already: (1) finds one path on the noiseless network, (2) per error set, builds a **new** fused `TNNetwork` via `TensorNetworkBuilder.build(..., mode="fuse")` and calls `engine.contract_batch()` against the *same* cached path object. Nothing about the CPU path actually reuses computation across error sets beyond the (currently meaningless, since path-finding is a stub) cached path — every error set still gets a fresh `_contract_network_cpu()` call over its own fused tensor list.

This machine now has a single NVIDIA H100 80GB HBM3 (driver 550.127.08), matching the paper's hardware class. A dedicated `tn-noise-sim` conda env (Python 3.11) was created and the package installed via `pip install -e ".[dev,gpu]"` (`cupy-cuda12x==14.1.1`, `cuquantum-python-cu12==26.6.0`/cuTensorNet 2.13.0 — newer than the paper's 26.01.0/2.11.00, no API concerns found). The CUDA runtime shared libraries (`libcublas`, `libcusolver`, etc.) were missing from the base environment and had to be added explicitly via `pip install nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-nvjitlink-cu12` — the `cupy-cuda12x`/`cuquantum-python-cu12` wheels do not vendor them and this host has no system CUDA toolkit on `PATH`/`LD_LIBRARY_PATH`.

Before committing to an implementation approach, the exact `cuquantum.tensornet.experimental.NetworkState` API (installed version) was inspected and smoke-tested live on the H100:

```python
from cuquantum.tensornet.experimental import NetworkState, TNConfig
state = NetworkState((2,2,2), dtype='complex128', config=TNConfig(num_hyper_samples=2))
tid_x = state.apply_tensor_operator((0,), X, unitary=True)          # returns an int tensor_id
...
rdm = state.compute_reduced_density_matrix((2,), fixed={1: 0}, diagonal=True)  # bounded, unnormalized marginal
state.update_tensor_operator(tid_x, new_operand, unitary=True)       # swap values, keep structure
```

Confirmed empirically (3-qubit X/H/CX circuit): `compute_reduced_density_matrix(where, fixed=..., diagonal=True)` returns the correct (unnormalized) marginal probability tensor of shape `(2,)*len(where)` — i.e. bounded to `2^{|where|}`, **not** `2^n` — and `update_tensor_operator` correctly swaps a gate's value in place while leaving the network structure untouched.

## Goals / Non-Goals

**Goals:**
- Peak GPU memory for any single batch contraction is `O(2^b)` (the requested batch/final-batch size), independent of `n`.
- UPV's "one path, reused across all E error sets" guarantee is realized with a real backend: path-finding cost is paid once (or once per distinct batch/fixed-mode *pattern*, not once per error set or per shot).
- All three simulator phases (Traditional, Unoptimized PTSBE, Optimized PTSBE) route through the same real GPU contraction primitive; only their *calling pattern* (fresh network per shot vs. per error set vs. once-with-updates) differs, matching Fig. 1/2 of the paper.
- Existing CPU dense-statevector path (`_compute_marginal`) is kept as the no-GPU fallback (`use_gpu=False` or CuPy/cuQuantum unavailable) — it is already correct at small `n`, just not scalable.

**Non-Goals:**
- Multi-GPU / intra-error-set parallelism (paper Sec. VI future work; out of scope here).
- Lightcone simplification (paper explicitly flags this as incompatible with UPV in its current form, Sec. VI future work).
- MPS-based simulation (`MPSConfig`) — the paper uses plain contraction-based `TNConfig`; we match that.
- Changing NBS's branching/exhaustive-harvest *decision logic* in `optimized_ptsbe.py`/`unoptimized_ptsbe.py` — only how it obtains a marginal changes.

## Decisions

### D1: Replace the custom `TNNetwork`/`opt_einsum` GPU path with `cuquantum.tensornet.experimental.NetworkState`

**Why**: `NetworkState.compute_reduced_density_matrix(where, fixed=..., diagonal=True)` is a purpose-built cuTensorNet primitive (backed by `cutensornetCreateMarginalDiagonal`) that computes *exactly* "marginal probability over `where` qubits, conditioned on `fixed` qubits, with all other qubits traced out" — this is NBS's core operation, expressed natively instead of being hand-rolled via dense contraction + slicing. Its output is bounded to `2^{|where|}`.

**Alternative considered**: keep the existing `TNNetwork`/`Tensor` custom graph and lower it to the low-level `cuquantum.tensornet.contract`/`Network` einsum API (which does expose an explicit `contract_path()`/`contract()` split). Rejected because that API operates in amplitude space — computing a partial-trace marginal with some qubits fixed and others summed out would require manually building a conjugate-paired "bra network" and threading projector/identity insertions through `opt_einsum`-style indices ourselves, reproducing (worse) what `NetworkState` already does natively and correctly.

**Consequence**: `TensorNetworkBuilder`'s existing wire-indexed `TNNetwork`/`Tensor` graph remains the CPU-fallback representation unchanged; a new build path constructs a `NetworkState` by calling `apply_tensor_operator(modes, gate_tensor)` once per gate **in circuit order**, reusing the exact same `(2,2)`/`(2,2,2,2)` `[out(s), in(s)]` tensor convention `TensorNetworkBuilder.build()` already produces — verified to match `NetworkState`'s expected `ABC...abc...` (output-then-input) mode ordering with no reshaping needed.

### D2: UPV reuse = one persistent `NetworkState` per noiseless circuit topology, updated via `update_tensor_operator` per error set

**Why**: The paper's Fig. 2c describes UPV as "fuse error into a copy of the coherent gate, same topology, reuse the path." `NetworkState.update_tensor_operator(tensor_id, new_operand)` is exactly this: it changes a tensor's *value* while leaving cuTensorNet's internally-cached contraction structure/path intact. So instead of rebuilding a fresh `NetworkState` per error set (which would force a fresh path-find each time — no better than Unoptimized PTSBE), `OptimizedPTSBESimulator` builds **one** `NetworkState` on the noiseless circuit, records the `tensor_id` returned by each `apply_tensor_operator` call (indexed by gate index), and for each error set calls `update_tensor_operator(tensor_id, fused_operand)` only for the gates that error set touches, leaving all other gates alone.

**Alternative considered**: build a fresh `NetworkState` per error set (mirroring the existing CPU code's `TensorNetworkBuilder.build(..., mode="fuse")` per error set). Rejected — this is architecturally identical to *Unoptimized* PTSBE (fresh network ⇒ cuTensorNet must re-derive/re-verify its internal contraction plan per error set), defeating UPV's entire premise.

**Consequence**: `ContractionEngine` needs a new stateful mode that owns a `NetworkState` across multiple `contract_batch()` calls for different error sets, rather than being handed a fresh `TNNetwork` each time. `Unoptimized PTSBE` and `Traditional`, by contrast, legitimately rebuild `NetworkState` per error set / per shot respectively (matching Fig. 1 center/left) — for them, one fresh `NetworkState` per error set (Unoptimized) or per shot (Traditional) is *correct*, not a shortcut we're avoiding.

### D3: Path-finding vs. contraction timing is inferred from cold-vs-warm calls, not an explicit separate call

**Why**: Unlike the low-level `Network` API, `NetworkState` doesn't expose a public `find_path()`/`contract()` split — path-finding happens lazily, internally, on the first `compute_reduced_density_matrix()` call for a given `(where, fixed-keys-pattern)` shape, and is cached internally by cuTensorNet for subsequent calls with the same shape (values of `fixed` may differ; verified empirically that changing only `fixed`'s *values* — not which modes are fixed — does not require rediscovering the tree, consistent with contraction paths depending on tensor topology/shape, not data values).

**Consequence**: the benchmarking harness measures path-finding cost as the (first call − mean of subsequent calls) delta for a given batch index, rather than calling a separate `find_path()`. This reproduces the paper's Fig. 6 "path-finding time" vs. "contraction time per shot" split without needing an API that doesn't exist at this abstraction level. `num_hyper_samples` (paper's "hypersamples") is set via `TNConfig(num_hyper_samples=...)` at `NetworkState` construction time — confirmed to exist and match the paper's terminology exactly.

### D4: `ContractionEngine` public interface stays batch/prefix-shaped; only its internals change

`find_path()` and `contract_batch()` keep their existing signatures and semantics from the simulators' point of view (batch index, `prefix` tuple of previously-sampled marginal indices) — `optimized_ptsbe.py`/`unoptimized_ptsbe.py`/`traditional.py` do not need algorithmic changes, only construction-time wiring (`use_gpu=True` by default; `OptimizedPTSBESimulator` passes error sets through a persistent engine/state instead of rebuilding a `TNNetwork` per error set per D2). `contract_batch()` translates `prefix` (indices into prior batches' marginals) into a `fixed={qubit_idx: bit_value}` dict for `compute_reduced_density_matrix`, and `batch_qubits` into `where`.

### D5: CPU fallback path is preserved verbatim

When `use_gpu=False` or CuPy/cuQuantum aren't importable, `ContractionEngine` keeps using `_compute_marginal()`'s dense-statevector CPU path exactly as today. It's correct (validated by 48/48 existing tests) and remains useful for small-`n` CI/dev-machine testing without a GPU. No behavior change for that path.

## Risks / Trade-offs

- **[Risk] `NetworkState`'s internal path cache lifetime/keying is undocumented at the Python level** (we only observed it empirically, not from a spec). If path caching turns out to be keyed more broadly or narrowly than the `(where, fixed-keys)` pattern we assumed, UPV's "found once" cost claim in our benchmark numbers could be wrong. → **Mitigation**: task list includes an explicit timing micro-benchmark (first call vs. Nth call latency, across several error sets with the same batch/fixed pattern) before trusting the split in real figures; if cold-call cost recurs unexpectedly, report that honestly in `validation_notes.md` rather than assuming UPV is free.
- **[Risk] `compute_reduced_density_matrix` cost may itself scale worse than expected with `n` for large, deep circuits** (the underlying contraction tree still spans the whole network even though the *output* is bounded to `2^b`) — this is inherent to tensor-network contraction generally (paper's own Fig. 6 shows contraction time growing with both `n` and `g`), not a bug. → **Mitigation**: this matches the paper's own reported behavior (some `(n,g)` configs hit "<80% success" from excessive time/memory); the harness must track and report per-config success rate rather than assuming every point in the grid completes.
- **[Risk] Two parallel `TNNetwork`-building code paths** (existing CPU wire-graph builder, new `NetworkState` builder) could drift if only one is updated later. → **Mitigation**: both share the same gate-tensor construction (`K @ U` fusion, `[out,in]`/`[out0,out1,in0,in1]` convention) from `TensorNetworkBuilder`; the new path is additive (a new method), not a rewrite of the existing one, and the GPU-vs-CPU distributional regression test (task list) is the ongoing drift detector.
- **[Trade-off] `final_batch_size` is still capped at 28** (paper's own cap, "largest population vector that fits on a single H100"), even though our H100 has 80GB (same as the paper's). We keep 28 as the default to match the paper rather than push higher, since going beyond it changes the reproduction target rather than the contraction engine itself; a larger `bf` could be explored as a follow-up but is out of scope here.

## Migration Plan

1. Implement `NetworkState`-backed contraction behind `ContractionEngine`, keep `use_gpu=False` as the default until the GPU-vs-CPU distributional regression test passes at n>26 (a scale the old CPU path cannot reach at all, so this is a new capability, not a regression risk for existing behavior).
2. Flip `use_gpu=True` default in the three simulator classes once step 1's regression test passes.
3. Existing CPU-path tests (48/48 from the prior validation pass) must continue passing unmodified throughout — they exercise `use_gpu=False`, which is untouched.
4. No rollback complexity: this is a library, not a deployed service; reverting is a normal git revert if the GPU path proves unreliable.

## Open Questions

- None blocking implementation — the API-level unknowns identified in earlier planning (exact `cuquantum` primitive, index convention compatibility, UPV reuse mechanism) were resolved empirically against the real installed version before writing this document (see Context).
