## 1. Complete Fig. 5's last point

- [x] 1.1 Run n=200,g=1000,mi=10000, num_error_sets=5, baseline_num_shots=10, generous timeout (900s) — **failed**: exceeded the 900s budget (actual wall-clock ran to ~1213s before the failure was recorded, consistent with the known SIGALRM-preemption limitation). This specific point (proportional mode, mi=10000, n=200) appears to be genuinely expensive, not just a tuning issue — recorded honestly rather than retried indefinitely with ever-larger timeouts.
- [x] 1.2 Regenerate `benchmarks/figures/fig5_proportional_speedup.png` — done; n=200,g=1000 still tops out at mi=1000 (3/4 points), n=100,g=600 remains fully complete (4/4).
- [x] 1.3 Record in `validation_notes.md` that this point failed, and why it's plausibly a real scaling limit rather than a fixable misconfiguration.

## 2. Corrected Fig. 4 at n=200 (matching the paper's actual configuration)

- [x] 2.1 Run non-proportional benchmarks at n=200 for g∈{200,600,1000}, bf∈{24,26,28} — **7/9 points completed** (all 3 bf values at g=200 and g=600; only bf=24 at g=1000) before the session wrapped up. All 7 succeeded.
- [x] 2.2 Regenerate `benchmarks/figures/fig4_bf_sweep.png` filtered to n=200 (`benchmarks/figures/fig4_bf_sweep_n200.png`) — a real g×bf grid this time, not a single-g snapshot.
- [x] 2.3 Determine whether throughput increases with bf at n=200 — done: raw PTSBE throughput (the correct Fig. 4 metric) is essentially **flat** across bf=24/26/28 at both g=200 and g=600 (~1% variation, likely instance noise, not the paper's claimed 2-4× per step). This is a distinct, more nuanced finding from the earlier n=100 "anomaly" (which used the speedup *ratio*, not raw throughput, and showed a clearer decreasing pattern) — recorded separately in `validation_notes.md`, not conflated.

## 3. Writeup

- [x] 3.1 Update `benchmarks/validation_notes.md` with a new dated section: the Fig. 4 n-mismatch correction and what the n=200 data actually shows, plus Fig. 5's completion status.
- [x] 3.2 Run the full `pytest` test suite and confirm no regressions (no library code changes expected).
