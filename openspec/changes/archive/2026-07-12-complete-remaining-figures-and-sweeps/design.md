## Context

Everything needed to run these three experiments already exists: `ContractionEngine` (real GPU-backed), `run_benchmark()`/`run_batch_size_sweep()` with the `baseline_num_shots` decoupling from the prior change, and `benchmarks/plots.py`'s `plot_fig5_proportional()`/`plot_fig7_batch_size_sweep()` functions. Three driver scripts (`_batch_sweep.py`, `_bf_anomaly_check.py`, `_prop_sweep_fix.py`) were written and verified against current code last session but never executed to completion — the session ended before reaching Fig. 7 and the bf-anomaly re-check, and the proportional sweep was cut short after fixing (but not fully exploiting) the `num_error_sets` amortization issue.

Known risk going in: `run_benchmark()`'s `timeout_s` (SIGALRM-based) doesn't preempt a single long blocking GPU call — it only fires once Python regains control. This was discovered, not fixed, last session. It matters here because the remaining proportional-sweep points (n=200,g=1000 at mi=100/1000/10000) are exactly the kind of large, slow calls where this could bite again.

## Goals / Non-Goals

**Goals:**
- Produce real data and figures for Fig. 7 and complete Fig. 5's dataset.
- Resolve or characterize the Fig. 4 bf-direction anomaly with more statistical weight than the original 1-instance/10-shot observation.
- Make the harness's failure records self-diagnosing for the specific E/baseline_shots mismatch class of failure found last session.

**Non-Goals:**
- Not fixing the SIGALRM timeout-preemption limitation itself (a real but separate engineering task — a proper fix needs a subprocess or thread-based watchdog, out of scope for an execution-focused change).
- Not resolving the n=75-vs-other-n g-trend divergence in the non-proportional grid (Session 2 finding) — unrelated to this change's three targets.
- Not deepening non-proportional grid coverage to 10 instances/cell.

## Decisions

### D1: Add `num_error_sets` to the result record (the one real spec change)

**Why**: last session's n=200,g=1000 failures took real debugging time to diagnose because the JSON output recorded `num_shots_requested` and `baseline_num_shots` but not `num_error_sets` — the third parameter whose ratio to `baseline_num_shots` was the actual root cause (Unoptimized PTSBE amortizes path-finding cost over `baseline_shots / num_error_sets` shots per error set; when that ratio is ~1, there's no amortization and each error set pays full cold-path-find cost). Adding this field means a future failed run's JSON alone is enough to spot this pattern, no driver-script cross-referencing needed.

**Alternative considered**: compute and record the shots-per-error-set ratio directly as a derived field. Rejected as premature — recording the raw `num_error_sets` is simpler, sufficient for diagnosis, and doesn't presume this specific ratio is the only failure mode worth flagging.

### D2: Size `num_error_sets` per-config based on scale, not one fixed value

For the remaining proportional points at n=200,g=1000, use `num_error_sets=5` (validated working for mi=10 last session) rather than the original `num_error_sets=20` (which failed at every mi value there). For n=100,g=600's mi=10000 retry, keep `num_error_sets=20` (this config's mi=10/100/1000 points already succeeded at E=20; only mi=10000 timed out, likely from Optimized PTSBE's own cost at that shot count, not the baseline/amortization issue) but raise the timeout generously (e.g. 800s) to give it more room, accepting that if it still doesn't complete, that's itself a real, reportable data point about PTSBE's own scaling at very high mi.

### D3: Bf-anomaly re-check runs as originally planned (3 instances/50 shots + deeper circuit), no design changes

The existing `_bf_anomaly_check.py` driver's approach — more shots/instances at the original (n,g), plus a deeper circuit where the paper says bf's effect is more pronounced — is still the right diagnostic; nothing learned since writing it changes that plan.

## Risks / Trade-offs

- **[Risk] The SIGALRM timeout limitation could resurface** for the n=200,g=1000 proportional points at mi=1000/10000 specifically (larger shot counts mean longer single calls). → Mitigation: generous timeouts (per D2) reduce the chance of hitting this in practice; if a config still hangs well past budget, kill it manually and record honestly rather than let it run indefinitely.
- **[Risk] bf-anomaly re-check with 3 instances/50 shots is more expensive than the original 1-instance/10-shot version** — could take considerably longer per bf value. → Mitigation: if this proves too slow mid-run, it's fine to reduce to 2 instances or stop after the n=100,g=600 portion (skipping the deeper g=1000 portion) and report what was learned from the cheaper part alone.
- **[Trade-off] `num_error_sets` field addition touches the `benchmarking-harness` spec** but is otherwise a one-line code change — low risk, but still goes through the same OpenSpec proposal/sync/archive discipline as larger changes, per this project's established practice.
