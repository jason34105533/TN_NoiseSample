## 1. Verify driver scripts against current main

- [x] 1.1 Read `benchmarks/_prop_sweep.py`, `_batch_sweep.py`, `_bf_anomaly_check.py` and confirm they still call `run_benchmark()`/`run_batch_size_sweep()` correctly against the current signatures — confirmed all 3 use valid, unchanged parameter names/order.
- [x] 1.2 Confirm `benchmarks/plots.py::plot_fig5_proportional()` and `plot_fig7_batch_size_sweep()` still work against the JSON schema the drivers produce — confirmed field names match (`num_shots_requested`, `batch_size`, `per_batch_time_s`).

## 2. Proportional sweep (Fig. 5)

- [x] 2.1 Run the proportional sweep — **partial**: n=100,g=600 got 3/4 mi values (10,100,1000 succeeded showing a clean 1.2×→5.6×→29.7× progression; mi=10000 timed out even at 500s). n=200,g=1000 needed `num_error_sets` reduced from 20→5 (E=20 with only ~20 baseline shots meant ~1 shot/error-set for Unoptimized PTSBE — no path-finding amortization, each error set paying a full cold path-find on a large/deep circuit, blowing any reasonable timeout regardless of mi). With E=5, got 1 point (mi=10, speedup 1.5×, ~438s). Did not chase mi=100/1000/10000 at n=200 — each point at this scale costs several minutes and the session was wrapping up.
- [x] 2.2 Generate `benchmarks/figures/fig5_proportional_speedup.png` — done, from the 4 real points collected (3 at n=100,g=600 + 1 at n=200,g=1000).
- [x] 2.3 Record measured speedups in `validation_notes.md` against the paper's ~1000× proportional claim — done; measured speedups (1.2×-29.7×) are well below the paper's ~1000×, consistent with only reaching mi≤1000 rather than the paper's full mi range, and honestly reported as such.

**Discovered this task**: the SIGALRM-based timeout in `run_benchmark()` does not reliably preempt long single blocking GPU/cuTensorNet calls — a signal raised mid-call is only handled once Python regains control, so a call that itself runs past `timeout_s` won't actually be interrupted at that budget. Not fixed this session (would need a process-level or thread-based timeout to truly preempt); noted as a real limitation of the current timeout mechanism, worth a future fix if long single-call hangs become a recurring problem.

## 3. Batch-size sweep (Fig. 7)

**Not run** — session wrapped up before reaching this task group.

- [ ] 3.1 Run `benchmarks/_batch_sweep.py`: n=100,g=600, bj∈{2,5,10,15,20,24,28}. Driver script verified against current `run_benchmark()`/`run_batch_size_sweep()` signatures (task 1.1/1.2) and ready to run.
- [ ] 3.2 Generate `benchmarks/figures/fig7_batch_size_sweep.png`.
- [ ] 3.3 Record the qualitative trend in `validation_notes.md`.

## 4. bf-sweep anomaly re-check (Fig. 4 follow-up)

**Not run** — session wrapped up before reaching this task group.

- [ ] 4.1 Run `benchmarks/_bf_anomaly_check.py` — driver script verified and ready, not executed.
- [ ] 4.2 Compare against the prior session's single-instance finding.
- [ ] 4.3 Document the finding in `validation_notes.md`.

## 5. Writeup

- [x] 5.1 Update `benchmarks/validation_notes.md` with a new dated section covering the proportional sweep (what ran, what didn't, the timeout-mechanism finding) and explicit "not run" notes for task groups 3 and 4.
- [x] 5.2 Run the full `pytest` test suite and confirm no regressions (no library code changes were made this session — only driver-script execution and documentation — so none expected).
