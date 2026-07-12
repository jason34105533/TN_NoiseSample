## 1. Harness fix: decouple baseline shot count

- [x] 1.1 Add `baseline_num_shots: Optional[int] = None` parameter to `run_benchmark()`; when set, pass it to `TraditionalTrajectorySimulator`/`UnoptimizedPTSBESimulator` instead of `num_shots`; default `None` → falls back to `num_shots` (unchanged behavior for existing non-proportional call sites).
- [x] 1.2 Add a unit test in `tests/test_benchmark_harness.py` confirming Traditional/Unoptimized receive `baseline_num_shots` when set, and `num_shots` when it's not.
- [x] 1.3 Sanity-check the fix: run a small proportional-mode config (e.g. n=20, g=50, num_shots=1000, baseline_num_shots=20) and confirm it completes well within a normal timeout, unlike last session's unbounded attempts. Confirmed: completed in 4.2s (n=20,g=50,mi=1000,baseline=20), measured speedup 338.9x.

## 2. Extend non-proportional grid (Fig. 3)

- [x] 2.1 Run additional (n,g) cells — done: 15 more cells added (19/25 total now, up from 4/25), full 5/5 coverage for n=50 and n=100, 3/5 for n=75/150/200 (missing g=400,800 at those three n values). All 19 succeeded (100% success rate). Still 1 instance per cell, not 10 — time went to breadth per design.md D3, not depth.
- [x] 2.2 Regenerate `benchmarks/figures/fig3_nonproportional_speedup.png` from the extended results.
- [x] 2.3 Regenerate `benchmarks/figures/fig6_contraction_pathfinding.png` from the extended results (cold/warm timing is already captured per-config).

## 3. Proportional sweep (Fig. 5)

**Not run** — stopped by explicit user direction ("stop at done non-proportional, archive now") before this task group produced any data worth keeping. The harness fix (task group 1) that unblocks this sweep is done and validated (see 1.3's sanity check); the sweep itself is future work.

- [ ] 3.1 Using the D1 fix, run the paper's two reference configs (n=100,g=600 and n=200,g=1000) across mi∈{10,100,1000,10000} with a small `baseline_num_shots` (e.g. 20-50).
- [ ] 3.2 Generate `benchmarks/figures/fig5_proportional_speedup.png` from real data.
- [ ] 3.3 Record measured speedups in `validation_notes.md` against the paper's ~1000× proportional claim.

## 4. Batch-size sweep (Fig. 7)

**Not run** — stopped before reaching this task group.

- [ ] 4.1 Run `run_batch_size_sweep(n=100, g=600, batch_sizes=[2,5,10,15,20,24,28], ...)` — implemented and unit-tested, driver script (`benchmarks/_batch_sweep.py`) written and ready, but never executed.
- [ ] 4.2 Generate `benchmarks/figures/fig7_batch_size_sweep.png` from real data.
- [ ] 4.3 Compare the qualitative trend against the paper's Fig. 7 finding.

## 5. bf-sweep anomaly investigation (Fig. 4 follow-up)

**Not run** — stopped before reaching this task group.

- [ ] 5.1 Re-run the bf sweep (bf∈{24,26,28}) at n=100,g=600 with more shots/instances — driver script (`benchmarks/_bf_anomaly_check.py`) written and ready, but never executed.
- [ ] 5.2 Run the same bf sweep at a deeper circuit (g=1000) — included in the same driver script, not executed.
- [ ] 5.3 Document the finding — deferred; the anomaly from the prior change remains unexplained.

## 6. Writeup

- [x] 6.1 Update `benchmarks/validation_notes.md` with a new dated section covering everything run in this change (harness fix + grid extension), and explicit "what did not run" for task groups 3, 4, and 5.
- [x] 6.2 Run the full `pytest` test suite and confirm no regressions from the harness change (task 1).
