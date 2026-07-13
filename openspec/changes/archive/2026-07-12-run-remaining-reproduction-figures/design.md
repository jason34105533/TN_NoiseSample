## Context

All engine and harness work for these three figures is already done and spec'd from the prior two changes:
- `ContractionEngine` is real GPU-backed and validated.
- `run_benchmark()`'s `baseline_num_shots` parameter (the fix that unblocks proportional sweeps) is implemented, unit-tested, and in `openspec/specs/benchmarking-harness/spec.md`.
- `run_batch_size_sweep()` is implemented and unit-tested.
- `benchmarks/plots.py` already has `plot_fig5_proportional()` and `plot_fig7_batch_size_sweep()` functions, unused so far only because no data existed for them.
- Driver scripts `benchmarks/_prop_sweep.py`, `_batch_sweep.py`, `_bf_anomaly_check.py` were written last session but never executed (session ended early by explicit user direction after the non-proportional grid work).

So this change has no design decisions in the usual sense — it's a "run it and check the result matches spec" change. This document exists mainly to record the execution plan and what "done" looks like for each of the three pieces.

## Goals / Non-Goals

**Goals:**
- Produce real Fig. 5 and Fig. 7 data and figures.
- Resolve (or at least characterize) the Fig. 4 bf-direction anomaly from the prior session.
- Confirm the driver scripts still work against current `main` (they predate the last change's final commits, though nothing they depend on should have changed).

**Non-Goals:**
- No engine, harness, or circuit-generator code changes are expected. If running the drivers reveals a real bug (not just "this takes a while"), that's a signal to pause and propose a follow-up fix rather than patching ad hoc inside this execution-only change.
- Not attempting deeper grid coverage (10 instances/cell, remaining 6 non-proportional cells) — out of scope per the proposal.

## Decisions

### D1: Run the three sweeps sequentially, not in parallel

All three drivers target the same single H100 GPU. Running them concurrently would contend for GPU memory/compute and make timing measurements (which several tasks depend on, e.g. `_cold_warm_from_engine()`'s per-call timing) unreliable. Run proportional → batch-size → bf-anomaly in sequence, each fully checkpointing to its own JSON file before the next starts.

### D2: Treat driver-script staleness as a quick fix-forward, not a blocker

The scripts reference `run_benchmark()`/`run_batch_size_sweep()` signatures from the end of the last session; both functions are unchanged since then (confirmed via the spec, which reflects their current contract). If running a driver hits an error, fix the driver script in place (it's a benchmarking utility, not committed library code with test coverage) rather than treating it as a design problem.

### D3: bf-anomaly re-check is diagnostic, not necessarily conclusive

The prior session's finding was 1 instance, 10 shots — enough for a single data point, not enough to distinguish a real effect from noise. This change's re-check (3 instances, 50 shots at the same config, plus a deeper-circuit config) raises confidence but may still not be definitive. Report whatever it shows plainly; a truly conclusive answer might need even more instances, which is fine to flag as future work rather than force into this change.

## Risks / Trade-offs

- **[Risk] Proportional sweep at mi=10,000 could still be slow** even with `baseline_num_shots` fixing the Traditional-side bottleneck — `OptimizedPTSBESimulator` itself still has to process 10,000 shots, and while PTSBE is fast per the whole point of this paper, it's not instant. → Mitigation: the sanity check from the prior session (mi=1000, small n) completed in 4.2s; if mi=10,000 at n=200 proves much slower, that's itself worth recording as a data point (Fig. 6-style contraction-time-per-shot), not a failure.
- **[Trade-off] Deciding "how many instances is enough" for the bf-anomaly re-check** is a judgment call with no paper-given answer for this specific question (the paper doesn't report per-cell variance for the bf sweep). 3 instances / 50 shots was chosen as a reasonable middle ground between "meaningfully more than last time" and "not another open-ended time sink" — not derived from a formal power analysis.
