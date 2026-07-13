## Why

Three prior changes got the GPU contraction engine, harness, and circuit generator fully paper-conformant and produced real (partial) data for Figs. 3, 4, and 5. Two figures remain entirely unrun (Fig. 7 batch-size sweep, and a re-check of the Fig. 4 final-batch-size anomaly), and the Fig. 5 proportional sweep is missing 4 of its 8 planned points — 3 at n=200,g=1000 (mi=100/1000/10000) and 1 at n=100,g=600 (mi=10000, which timed out at 500s). All three have ready, verified driver scripts (`benchmarks/_batch_sweep.py`, `_bf_anomaly_check.py`, `_prop_sweep_fix.py`) from the prior session — this is "run the remaining experiment," not new engineering, per the user's framing.

One real gap surfaced while diagnosing last session's n=200,g=1000 failures: `num_error_sets` (E) — the parameter whose mismatch with `baseline_num_shots` was the actual root cause of those failures (E=20 with ~20 baseline shots left Unoptimized PTSBE with ~1 shot/error-set, no path-finding amortization) — isn't recorded in `run_benchmark()`'s JSON output. Diagnosing that class of failure required cross-referencing the driver script by hand. Worth fixing now since we're about to run more configs where this exact interaction matters.

## What Changes

- Add `num_error_sets` to `run_benchmark()`'s per-instance result record, so future failure diagnosis (E vs. baseline_num_shots amortization mismatches, like the one found last session) doesn't require cross-referencing the calling script.
- Run the batch-size sweep (`benchmarks/_batch_sweep.py`): n=100,g=600, bj∈{2,5,10,15,20,24,28}. Generate `benchmarks/figures/fig7_batch_size_sweep.png`.
- Run the bf-anomaly re-check (`benchmarks/_bf_anomaly_check.py`): n=100,g=600 with 3 instances/50 shots per bf∈{24,26,28}, plus n=100,g=1000 with 1 instance/10 shots per bf value. Determine whether the previously-observed reversal (speedup decreasing with larger bf, opposite the paper's direction) persists, reverses, or looks like small-sample noise.
- Complete the proportional sweep: retry n=100,g=600 mi=10000 with a larger timeout; run n=200,g=1000 at mi∈{100,1000,10000} using `num_error_sets=5` (the fix already validated for mi=10 at this scale last session). Regenerate `benchmarks/figures/fig5_proportional_speedup.png` with the fuller dataset.
- Update `benchmarks/validation_notes.md` with a new dated section covering all three, reporting real numbers honestly (including if the bf-anomaly persists, or if large-mi proportional points still can't complete).

**Explicitly out of scope**: the n=75-vs-other-n non-proportional g-trend divergence (a separate investigation), deepening the non-proportional grid to 10 instances/cell, and fixing the underlying SIGALRM timeout-preemption limitation found last session (noted as a real issue, not addressed here — if it blocks a specific config in this change, that's recorded honestly rather than worked around with a new timeout mechanism, which would be its own change).

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `benchmarking-harness`: the per-instance results record SHALL include `num_error_sets`, so failures caused by an `num_error_sets`/`baseline_num_shots` amortization mismatch (Unoptimized PTSBE getting too few shots per pre-sampled error set to amortize path-finding) can be diagnosed directly from the JSON output.

## Impact

- **Code**: one-line addition to `run_benchmark()`'s record dict (`src/tn_noise_sim/` untouched — this is a `benchmarks/` harness change only).
- **Compute**: real H100 GPU time for three sweeps. Rough scale from prior sessions: batch-size sweep points are tens of seconds to a couple minutes each (7 points); bf-anomaly re-check's 3-instance/50-shot points are more expensive than the original 1-instance/10-shot bf sweep (~2-4x per point plausible); remaining proportional points at n=200,g=1000 took ~438s for the one successful mi=10 point last session, so mi=100/1000/10000 there could each take longer.
- **Docs**: `benchmarks/validation_notes.md` updated; `fig7_batch_size_sweep.png` created; `fig5_proportional_speedup.png` regenerated with more points.
