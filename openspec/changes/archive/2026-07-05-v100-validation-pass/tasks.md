## 1. Environment setup

- [x] 1.1 Create a dedicated conda environment for this project (e.g. `tn-noise-sim`, Python >=3.10), sibling to the existing `cvenv`/`nlpenv` environments.
- [x] 1.2 In the new environment, `pip install -e ".[dev,gpu]"`; confirm `cupy` and `cuquantum` import correctly inside the env (not just system-wide).
- [x] 1.3 Confirm `pytest` and other dev tools are available in the new environment.

## 2. CPU test suite validation against real dependency versions

- [x] 2.1 Run `pytest tests/` in the new environment and record all failures. All 44 pre-existing tests passed with no changes; however this revealed the suite (and the benchmark circuit generator) never actually exercises `TensorNetworkBuilder.from_qiskit_circuit` — the one real Qiskit-IR-dependent code path (design.md D2). Added 4 direct tests for it (`tests/test_tensor_network.py`, section 3.6) using a real `qiskit.QuantumCircuit` against installed Qiskit 2.5.0; all pass (48/48 total).
- [x] 2.2 Triage failures: separate Qiskit 2.x / NumPy 2.x API incompatibilities from genuine logic bugs. No failures occurred; nothing to triage.
- [x] 2.3 Fix forward each Qiskit/NumPy compatibility issue found. None found — `instruction.operation`/`instruction.qubits`/`qc.find_bit`/`Operator(op).data` all still work unchanged in Qiskit 2.5.0.
- [x] 2.4 No documented-behavior changes were needed.
- [x] 2.5 Suite is green (48/48) with no deferred failures.
- [x] 2.6 Not applicable — Qiskit 2.x proved fully compatible with the code paths this project uses. No downgrade to 1.x warranted.

## 3. GPU device metadata in benchmarking harness

- [x] 3.1 Add GPU device name and total device memory capture (`benchmarks/timing.py::gpu_device_info`, via `cupy.cuda.runtime.getDeviceProperties`/`memGetInfo`) to the benchmarking harness's results JSON output (`benchmarks/run_benchmark.py`).
- [x] 3.2 Include the GPU device name in the stdout summary table.
- [x] 3.3 Added `tests/test_benchmark_harness.py` covering `gpu_device_info()` (both with and without GPU) and the `gpu_device_name`/`gpu_memory_total_bytes` fields on `run_benchmark()` records.

## 4. Smoke-scale GPU execution

- [x] 4.1 **Scope revised after checking the reference paper (arXiv:2604.08467) against the actual code.** Before running anything on the V100, I checked what `use_gpu=True` actually does: `ContractionEngine._find_path_cutensornet()` is a stub that silently falls back to CPU `opt_einsum`, and `_compute_marginal()` always contracts on host numpy arrays — `cupy`/`cuquantum` are only used to drive `benchmarks/timing.py`'s CUDA-event `Timer`, never real computation. Worse: `_compute_marginal()`'s own docstring admits it's a "CPU reference implementation" that contracts the **full `2^n`-entry statevector** before slicing to get a batch marginal — the paper's Fig. 1 shows contraction happening on GPU with memory bounded to `2^b` (the batch size), for *all three* simulator phases, not just the optimized one. A dense `2^n` statevector is infeasible at the paper's n=100-200 regardless of hardware (2^100 has no physical realization). So real cuTensorNet contraction isn't a missing optimization to bolt on — it's unbuilt core functionality, and completing it (properly, via `cuquantum.tensornet.experimental.NetworkState`, confirmed available and Qiskit-circuit-native in the installed `cuquantum-python-cu12 26.6.0`) is a separate, larger engineering effort deserving its own change, not a task inside a "validation pass." Ran the existing dense-statevector CPU path instead, at n=20/g=50 (well past the old n=6 toy, still within the ~16MB memory bound for 2^20) — all three phases execute successfully; results in `benchmarks/validation_notes.md`.
- [x] 4.2 No execution bugs surfaced at n=20 CPU scale. (cuTensorNet-specific bugs are moot until real GPU contraction is implemented — see 4.1.)
- [x] 4.3 Cross-checked Phase 1 vs. Phase 3 (proportional mode) bitstring distributions on a real (nonzero-noise) n=5 circuit at 20,000 shots. Found total variation distance of 0.21 — an order of magnitude above the ~0.02-0.04 expected from sampling noise alone, confirming a **real correctness bug**: `ErrorSampler._sample_proportional()` (`src/tn_noise_sim/error_sampling.py`) drew error sets i.i.d. from their true probability, then re-weighted shot counts by that same probability again (double-applying it, a self-normalized-importance-sampling bug). Fixed by allocating shots uniformly across the i.i.d. draws (matching `_sample_non_proportional`'s existing logic) — TVD dropped to 0.036, consistent with sampling noise. Added a regression test (`tests/test_simulators.py::test_proportional_matches_traditional_distribution`); the existing `test_proportional_born_rule` never caught this because it uses zero noise, where the bug is dormant (all weights are identically 1).

## 5. Scale-up benchmarking toward paper regime

- [x] 5.1 **Descoped, not attempted, for a fundamental reason (not a hardware limit).** Per 4.1's finding, `_compute_marginal()` materializes a dense `2^n`-entry statevector before slicing. Confirmed empirically: at n=26, g=60, a 20-shot Traditional run (fresh path-find + full contraction per shot) took several minutes and grew past 4GB resident memory. The paper's n=100-200 regime would require materializing `2^100`-`2^200`-entry arrays — not a matter of "more time" or "a GPU," physically impossible on any existing computer. Scaling toward paper regime requires the bounded-memory GPU contraction from 4.1's recommendation first.
- [x] 5.2 Not applicable — never reached a scale where `bf` reduction was the binding constraint; the dense-statevector materialization is the binding constraint at any n much above ~26-28, independent of `bf`.
- [x] 5.3 Collected honest numbers within the regime the current architecture actually supports (n=20, not n=100-200) — see `benchmarks/validation_notes.md`. Did not collect numbers at paper scale; recorded why rather than guessing or faking a number.

## 6. Results documentation

- [x] 6.1 Updated `benchmarks/validation_notes.md`. No GPU numbers to record (no GPU contraction exists to run — see 4.1); recorded honest CPU-only numbers at n=20 instead, and explicitly noted no `bf` reduction was ever needed since the dense-statevector wall binds well before `bf` does.
- [x] 6.2 Documented why V100/current-architecture numbers are not comparable to the paper's Table I / Figs. 4-7 at all yet (different regime, no real GPU contraction, no lightcone simplification) rather than presenting a misleading side-by-side comparison.
- [x] 6.3 Recorded two follow-up items in `validation_notes.md`: (1) implement real bounded-memory GPU contraction via `cuquantum.tensornet.experimental.NetworkState` — prerequisite for any paper-scale or true V100/H100 benchmarking; (2) no deferred Qiskit/NumPy compat issues from section 2 (none were found).
