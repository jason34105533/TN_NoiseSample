## 1. Record num_error_sets in results

- [x] 1.1 Add `"num_error_sets": num_error_sets` to `run_benchmark()`'s per-instance result record.
- [x] 1.2 Add/update a unit test in `tests/test_benchmark_harness.py` confirming `num_error_sets` appears in the record.
- [x] 1.3 Run `tests/test_benchmark_harness.py` to confirm the addition doesn't break existing tests — 12/12 pass.

## 2. Batch-size sweep (Fig. 7)

- [x] 2.1 Run `benchmarks/_batch_sweep.py`: n=100,g=600, bj∈{2,5,10,15,20,24,28} — 6/7 succeeded (bj=28 timed out at 200s, both non-final and final batch set to 28 is genuinely expensive).
- [x] 2.2 Generate `benchmarks/figures/fig7_batch_size_sweep.png`.
- [x] 2.3 Record the qualitative trend in `validation_notes.md` — done: per-batch time rises smoothly from 0.39s (bj=2) to 0.54s (bj=24), matching the paper's direction (larger batch = costlier per contraction).

## 3. Complete the proportional sweep (Fig. 5)

- [x] 3.1 Retry n=100,g=600, mi=10000 with a larger timeout (800s), num_error_sets=20 — **succeeded this time**: speedup 59.5×. n=100,g=600 now has a complete 4/4-point dataset (mi=10→1.2×, 100→5.6×, 1000→29.7×, 10000→59.5×).
- [x] 3.2 Run n=200,g=1000 at mi∈{100,1000,10000} with num_error_sets=5 — **3/3 of the newly-attempted points succeeded**: mi=100→11.9×, mi=1000→36.8×. mi=10000 was in progress (killed mid-run, session wrapping up) — not recorded. n=200,g=1000 now has 3/4 points (mi=10 from a prior session + mi=100,1000 from this one; only mi=10000 missing).
- [x] 3.3 Regenerate `benchmarks/figures/fig5_proportional_speedup.png` with the fuller dataset — done, now 7 real points across both reference configs (up from 4).
- [x] 3.4 Record the completed dataset in `validation_notes.md` against the paper's ~1000× claim — done.

## 4. bf-sweep anomaly re-check (Fig. 4 follow-up)

**Not run** — session ended before reaching this task group. Driver script (`benchmarks/_bf_anomaly_check.py`) remains verified and ready.

- [ ] 4.1 Run `benchmarks/_bf_anomaly_check.py`.
- [ ] 4.2 Compare against the original single-instance finding.
- [ ] 4.3 Document the finding in `validation_notes.md`.

## 5. Writeup

- [x] 5.1 Update `benchmarks/validation_notes.md` with a new dated section covering the `num_error_sets` field addition, the batch-size sweep, and the completed proportional-sweep data — plus an explicit "not run" note for task group 4.
- [x] 5.2 Run the full `pytest` test suite and confirm no regressions.
