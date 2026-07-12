## Why

The archived change `gpu-bounded-memory-contraction` (2026-07-12) implemented real bounded-memory GPU contraction on this H100 and validated it thoroughly (64/64 tests, numerically checked against the CPU path), but its reproduction sweep was cut short by an explicit time budget: only 4 of the paper's 25 (n,g) grid cells ran (1 instance each, not 10), and the proportional-mode sweep (Fig. 5) produced no usable data at all — every attempt timed out because `TraditionalTrajectorySimulator`'s real-GPU path builds one `NetworkState` per shot (~1-2s overhead each), and the harness gives Traditional the *same* shot count as the PTSBE side of the comparison it's timing. The engine and harness are already paper-conformant; what's missing is enough GPU time (and one harness fix) to actually produce the paper's Figs. 3, 5, and 7, plus a look at an unexplained anomaly in the one sweep (Fig. 4, final-batch-size) that did complete.

## What Changes

- Decouple the Traditional/Unoptimized baseline's shot count from PTSBE's swept shot count (`mi`) in `run_benchmark()`, since throughput is a per-shot rate, not an absolute count — the baseline only needs enough shots for a stable rate estimate (tens, not thousands), while PTSBE's `mi` is the actual figure parameter being swept. This is what actually blocked the proportional sweep last session, not a fundamental limitation.
- Extend the non-proportional grid (Fig. 3) beyond the 4 cells already run — target more of n∈{50,75,100,150,200} × g∈{200,400,600,800,1000}, with 10 instances per cell where time allows, to get real success-rate (>80%/<80%) statistics rather than 1-instance point estimates.
- Run the proportional sweep (Fig. 5) for the paper's two reference configs (n=100,g=600 and n=200,g=1000) across mi∈{10,100,1000,10000}, now that the baseline-decoupling fix makes it tractable.
- Run the batch-size sweep (Fig. 7) for n=100,g=600 across bj∈{2,5,10,15,20,24,28} — implemented and unit-tested (`run_batch_size_sweep()`) but never executed last session.
- Investigate the final-batch-size (bf) sweep anomaly from the completed Fig. 4 run: speedup *decreased* slightly with larger bf (1,275,393× → 1,081,521× for bf 24→28), opposite the paper's reported direction. Re-run with more shots and/or a deeper circuit to distinguish small-sample noise from a real effect; document the finding either way.
- Update `benchmarks/validation_notes.md` and regenerate `benchmarks/figures/` with whatever real data results from the above — reporting actual measured numbers, not adjusted ones, per the existing `paper-figure-reproduction` capability's "Honest reporting" requirement.

**Explicitly out of scope**: any further changes to the contraction engine itself (real bounded-memory GPU contraction is done and validated); multi-GPU parallelism; lightcone simplification — both already out of scope per the prior change and unaffected by this one.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `benchmarking-harness`: `run_benchmark()` SHALL support an independent (smaller) shot count for the Traditional/Unoptimized baseline than the shot count swept for `OptimizedPTSBESimulator`, since throughput is a per-shot rate and the baseline only needs enough shots to estimate that rate stably — this removes the per-instance wall-clock blocker that prevented any proportional-mode sweep from completing last session.

## Impact

- **Code**: `benchmarks/run_benchmark.py` (baseline/PTSBE shot-count decoupling — the only expected source change), possibly `benchmarks/_reproduction_run.py`-style driver scripts (new or extended sweep drivers).
- **No changes expected** to `src/tn_noise_sim/` (contraction engine and simulators are already correct and paper-conformant per the archived change's validation).
- **Compute**: substantial additional H100 GPU time — how much depends on how far the grid/sweeps are extended; unlike the prior change, no hard time budget has been set for this one unless the user specifies one.
- **Docs**: `benchmarks/validation_notes.md` updated with whatever real results this pass produces; `benchmarks/figures/` regenerated.
