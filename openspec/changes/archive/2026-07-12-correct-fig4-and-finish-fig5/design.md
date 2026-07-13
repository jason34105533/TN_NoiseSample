## Context

Re-reading the paper's Sec. V-A carefully (per explicit user request to re-verify against the spec/paper before continuing) surfaced that the prior two bf-sweep attempts (`benchmarks/_bf_anomaly_check.py`'s n=100,g=600/g=1000 legs, and the original Fig. 4 run) tested the wrong n. The paper states the bf effect is "demonstrated for n = 200 systems," and Fig. 4's x-axis is g (implying a sweep across the grid at fixed n=200, bf as the varied line-per-series parameter), not a single g value. `plot_fig4_bf_sweep(results_path, output_path, n=None)` already accepts an `n` filter and groups correctly by `(bf, g)` — it was just never called with n=200 data spanning multiple g values.

Fig. 5's dataset is otherwise complete except for one point (n=200,g=1000,mi=10000) that was mid-run when the prior session ended under time pressure.

## Goals / Non-Goals

**Goals:**
- Produce a real, paper-matching Fig. 4 (n=200, multi-g, bf as separate lines) and determine honestly whether the paper's bf-increases-throughput trend holds there.
- Close Fig. 5's last gap.
- Be explicit in the writeup that the earlier n=100 "anomaly" and this n=200 result are two different experiments, not directly comparable — avoid implying the new result "confirms" or "refutes" the old one when they tested different regimes.

**Non-Goals:**
- Not attempting the full paper g-grid (200,400,600,800,1000) for Fig. 4 at n=200 — 3 g-values (200,600,1000) spanning the range is the tractable compromise given per-point cost at this n.
- Not fixing the SIGALRM timeout issue, even if it resurfaces during this change's n=200 runs (same stance as the prior change: work around it with generous timeouts and honest reporting of any failures, not a new engineering effort).

## Decisions

### D1: Fig. 5's missing point first, then Fig. 4's corrected grid

Fig. 5's remaining work is a single, well-scoped, already-validated-elsewhere config (n=200,g=1000,mi=10000, using the `num_error_sets=5` fix already confirmed to work for mi=100/1000 at this n last session). Doing it first means Fig. 5 is either fully done or clearly not, quickly, before spending the larger time budget on Fig. 4's 9-point grid.

### D2: g∈{200,600,1000} for the corrected Fig. 4 grid, not the full 5-value paper grid

Chosen to span the paper's low/mid/high g range at n=200 while keeping the point count (9, not 15) and total time tractable, consistent with every prior session's "breadth over completeness under time pressure" pattern in this project. If time allows, g=400 and g=800 are natural follow-ups, not required for this change to produce a real, honest Fig. 4.

### D3: Report the n=200 result as a distinct, new finding — not a resolution of the n=100 anomaly

Even if the n=200 trend matches the paper's direction, that doesn't retroactively explain *why* n=100 went the other way (different scale, different circuit depth relative to n, etc.) — it just means the paper's own tested regime reproduces correctly. The n=100 anomaly (from two prior sessions) remains a separate, still-open observation about a regime the paper didn't test, and should stay flagged as such in `validation_notes.md` rather than being quietly folded into "resolved."

## Risks / Trade-offs

- **[Risk] n=200 non-proportional points could still hit timeouts** given the SIGALRM limitation and this n's track record of expensive single points. → Mitigation: use generous per-call timeouts (matching what worked for n=200 non-proportional grid cells in an earlier session, ~400s) and record any failures honestly rather than silently retrying indefinitely.
- **[Trade-off] Choosing only 3 of 5 g-values for the corrected Fig. 4 is itself an incomplete reproduction** — acceptable because 3 points already establish whether the throughput-vs-g trend and the bf-ordering trend both hold at n=200, which is the core question this change is trying to answer; a denser grid would refine confidence, not change the qualitative answer.
