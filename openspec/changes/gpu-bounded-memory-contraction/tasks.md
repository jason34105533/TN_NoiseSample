## 1. GPU-backed contraction engine

- [x] 1.1 Add a `NetworkState`-based build path (`TensorNetworkBuilder` or a new helper) that constructs a `cuquantum.tensornet.experimental.NetworkState` from a circuit dict by calling `apply_tensor_operator(modes, gate_tensor)` per gate in circuit order, reusing the existing `[out,in]`/`[out0,out1,in0,in1]` gate tensor convention. Return the `NetworkState` plus a `{gate_idx: tensor_id}` mapping.
- [x] 1.2 Rewrite `ContractionEngine` to support a GPU mode backed by `NetworkState`: `find_path()`/network construction creates (or retrieves from `ContractionPathCache`) a `NetworkState` configured with `TNConfig(num_hyper_samples=...)`; `contract_batch(batch_index, prefix)` translates `prefix` into a `fixed={qubit: bit}` dict and calls `compute_reduced_density_matrix(where=batch_qubits, fixed=fixed_dict, diagonal=True)`, flattening/normalizing the result to match the existing `np.ndarray` shape `(2**b,)` contract that `optimized_ptsbe.py`/`unoptimized_ptsbe.py`/`traditional.py` already depend on.
- [x] 1.3 Add a persistent-`NetworkState` UPV mode: given a noiseless circuit's `NetworkState` (built once) and its `{gate_idx: tensor_id}` mapping, apply an error set via `update_tensor_operator(tensor_id, fused_operand)` for only the touched gates, without rebuilding the network. Wire `ContractionPathCache` to store the `NetworkState`/mapping instead of an opaque path object for the GPU backend.
- [x] 1.4 Keep the existing CPU dense-statevector `_compute_marginal()` path fully intact as the `use_gpu=False` / no-CuPy fallback; add a runtime check that falls back gracefully (with a warning, not a silent behavior change) if `cupy`/`cuquantum` import fails despite `use_gpu=True`.
- [x] 1.5 Unit tests in `tests/test_contraction.py`: GPU-backed `contract_batch()` matches CPU-backed `contract_batch()` on the same small circuit (n≤10) for both unconditioned and prefix-conditioned marginals, within floating-point tolerance.
- [x] 1.6 Unit test confirming `update_tensor_operator`-based UPV reuse: construct one `NetworkState`, apply two different error sets via `update_tensor_operator`, and confirm the resulting marginals match what a from-scratch `NetworkState` per error set would produce.

## 2. Simulator wiring

- [x] 2.1 `OptimizedPTSBESimulator`: replace the per-error-set `TensorNetworkBuilder.build(..., mode="fuse")` + fresh contraction with the persistent-`NetworkState`/`update_tensor_operator` UPV path from 1.3, keeping the existing NBS branching/exhaustive-harvest logic (`_sample_non_proportional`, `_sample_proportional`) unchanged apart from how a marginal is obtained.
- [x] 2.2 `UnoptimizedPTSBESimulator`: wire to build one `NetworkState` per error set (matching Fig. 1 center — fresh network per error set, reused across that error set's `m_i` shots), via the GPU contraction engine.
- [x] 2.3 `TraditionalTrajectorySimulator`: wire to build one `NetworkState` per shot (matching Fig. 1 left — fresh path per shot), via the GPU contraction engine.
- [x] 2.4 Flip `use_gpu=True` as the default in all three simulator constructors (or wherever they construct `ContractionEngine`), while leaving an explicit `use_gpu=False` override available.
- [x] 2.5 Add a GPU-vs-CPU distributional regression test (style of `tests/test_simulators.py::test_proportional_matches_traditional_distribution`) at a scale beyond n=26 (where the old CPU path could not run at all, e.g. n=40), confirming non-proportional and proportional bitstring distributions are statistically consistent between the two backends where both can run (small n), and that the GPU path alone completes correctly beyond CPU's reach.
- [x] 2.6 Run the full existing `pytest` suite (all 48+ tests) against the new default and confirm no regressions; existing CPU-only tests must keep passing since `use_gpu=False` behavior is untouched.

## 3. Circuit generation and benchmark harness alignment

- [x] 3.1 Rewrite `benchmarks/circuit_generator.py::generate_circuit` to draw single-qubit gates from {H, X, Y, Z, T, Rx} and two-qubit nearest-neighbor gates from {CX, CY, CZ, CH, CRx} (as fixed unitary matrices, not Haar-random), with a default `two_qubit_fraction=0.2`.
- [x] 3.2 Confirm/adjust noise coupling: Pauli (X/Y/Z) errors for single-qubit gates, two-qubit depolarizing for two-qubit gates, probability ~U[0.02, 0.20] (already matches; add a direct test if none exists).
- [x] 3.3 Update `benchmarks/run_benchmark.py` defaults to the paper's Sec. IV-C values: `batch_size=10`, `final_batch_size=28`, `num_hypersamples=100` for PTSBE simulators; Traditional simulator fixed at `batch_size=24`, `num_hypersamples=1` (confirm `TraditionalTrajectorySimulator` already enforces this per its spec).
- [x] 3.4 Switch speedup aggregation in `run_benchmark.py`/`_print_summary` from arithmetic mean/std to geometric mean/std (`scipy.stats.gmean` and log-space std), matching Sec. IV-D.
- [x] 3.5 Add per-instance success/failure tracking: a configurable time/memory budget per instance; instances exceeding it are recorded with `success=False` and excluded from the geometric-mean computation but retained in JSON output. Compute and report per-(n,g) success rate.
- [x] 3.6 Add `mode` parameter (proportional/non-proportional) plumbing through `run_benchmark()` to `OptimizedPTSBESimulator.sample()`.
- [x] 3.7 Add a batch-size-sweep entry point (`run_batch_size_sweep` or similar) that fixes (n, g) and sweeps `bj` over a caller-specified list, recording per-batch contraction+sampling time.
- [x] 3.8 Add cold/warm timing capture per the design's path-finding-cost approach (first call vs. subsequent calls for the same batch/fixed pattern) so Fig. 6-style path-finding-vs-contraction splits can be computed from results.
- [x] 3.9 Update/add tests in `tests/test_benchmark_harness.py` covering the new gate set, geometric stats, success tracking, and mode/batch-sweep plumbing.

## 4. Paper figure reproduction

- [x] 4.1 Add `benchmarks/plots.py` (or similar) with functions to produce Fig. 3 (non-proportional speedup vs g per n), Fig. 4 (throughput vs g vs bf), Fig. 5 (proportional speedup vs mi, two configs), Fig. 6 (contraction/path-finding time and ratio vs g per n), and Fig. 7 (per-batch cost vs batch size for n=100,g=600) from results JSON, using matplotlib, log-scale axes, and hollow markers for <80%-success configurations. Fig. 3 and Fig. 6 generated from real data; Fig. 4/5/7 implemented and callable but not yet run against real sweep data (see 5.x).
- [x] 4.2 Each plot function SHALL report actual measured values without adjustment; docstring in `plots.py` states this explicitly. `validation_notes.md` reports the measured 10⁶×-range speedups plainly alongside the paper's ~10⁸× claim rather than adjusting them.

## 5. Full-scale H100 run and validation writeup

**Scope reduced mid-session to a ~30-minute GPU time budget** (explicit request), so this group ran a small curated subset rather than the full grid/sweeps below. Sub-items are marked with what actually happened.

- [x] 5.1 Run a small correctness/smoke pass first (e.g. n=10-20) — done via `tests/test_contraction.py`'s n=40 bounded-memory test and `tests/test_simulators.py`'s n=40 GPU-completion test, plus the first curbed-sweep config (n=50,g=200) before scaling up.
- [~] 5.2 Run the non-proportional grid — **partial**: 4 of the paper's 25 (n,g) cells run (n∈{50,100} × g∈{200,600}), 1 instance each (not 10), E=5 (not the paper's implicit larger E), 10 shots. All 4 succeeded; speedups in the 10⁶× range. Fig. 3 generated from these 4 points (`benchmarks/figures/fig3_nonproportional_speedup.png`). Remaining 21 cells and the 10-instances-per-cell statistics were **not run** — cut short by the time budget mid-way through a 5th config (n=200,g=600).
- [ ] 5.3 Final-batch-size sweep (bf∈{24,26,28}) — **not run** (implemented, not executed).
- [ ] 5.4 Proportional sweep (n=100,g=600 and n=200,g=1000 across mi∈{10,100,1000,10000}) — **not run** (implemented and unit-tested via `run_benchmark(..., mode="proportional")`, not executed at scale).
- [x] 5.5 Cold/warm contraction and path-finding timing — captured for all 4 completed 5.2 configs; Fig. 6 generated (`benchmarks/figures/fig6_contraction_pathfinding.png`). Proportional reference configs at mi=10000 not run (depends on 5.4).
- [ ] 5.6 Batch-size sweep (bj∈{2,5,10,15,20,24,28}) for n=100,g=600 — **not run** (`run_batch_size_sweep()` implemented and included in `benchmarks/_reproduction_run.py`'s driver, but the driver was stopped before reaching this stage).
- [x] 5.7 Write up real results in `benchmarks/validation_notes.md` — done: measured geomean speedups (10⁶× range) reported plainly against the paper's ~10⁸×/~10³× claims, what ran vs. didn't, and honest deviation notes (dependency versions, no literal `cudaq` baseline, missing CUDA runtime libs).
- [x] 5.8 Time budget did not allow the full grid — followed the prioritization spirit (real Fig. 3 data, though only 2 of 5 n values rather than 3+; Fig. 5/7 not reached at all). `validation_notes.md` records exactly what ran and what didn't, including a note that `benchmarks/_reproduction_run.py` is resumable/extensible for a follow-up session to reach the remaining grid, proportional sweep, and batch-size sweep.
