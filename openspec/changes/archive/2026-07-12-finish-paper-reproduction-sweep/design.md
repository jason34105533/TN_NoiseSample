## Context

From the archived `gpu-bounded-memory-contraction` change: `ContractionEngine` (GPU-backed via `cuquantum.tensornet.experimental.NetworkState`), the three simulators, and `benchmarks/run_benchmark.py`/`circuit_generator.py` already match the paper's algorithm and experimental setup. What's incomplete is execution coverage:

- Non-proportional grid (Fig. 3): 4/25 cells, 1 instance each.
- Proportional sweep (Fig. 5): 0 usable data points — every attempt timed out.
- Batch-size sweep (Fig. 7): not run.
- bf sweep (Fig. 4): complete, but shows an unexplained direction reversal vs. the paper.

Root cause of the proportional-sweep failures: `run_benchmark(n, g, num_instances, num_shots, ...)` passes the same `num_shots` to `TraditionalTrajectorySimulator`, `UnoptimizedPTSBESimulator`, and `OptimizedPTSBESimulator` alike. For non-proportional sweeps this was fine (shots stayed at 10). For proportional sweeps, `num_shots` doubles as the paper's `mi` (the swept x-axis variable, up to 10,000) — so Traditional was asked to take 10,000 individual per-shot GPU network builds at ~1-2s each, which no reasonable timeout accommodates.

## Goals / Non-Goals

**Goals:**
- Make proportional-mode benchmarking tractable by decoupling the baseline's shot count (needed only for a stable throughput *rate* estimate) from PTSBE's `mi` (the actual figure parameter).
- Produce real data for Figs. 3 (extended), 5, and 7, and enough data on the Fig. 4 bf-direction anomaly to either explain it or characterize it as noise.
- Keep the fix minimal — this is an execution/data-collection change, not an engine change.

**Non-Goals:**
- Changing the contraction engine, UPV/NBS algorithms, or circuit generation — all validated correct in the prior change.
- Guaranteeing the paper's exact grid/instance-count coverage — GPU time is the binding constraint here, same as last time, just without a hard 30-minute cap this time. How far to push is a running judgment call, not fixed upfront.

## Decisions

### D1: Add a `baseline_num_shots` parameter to `run_benchmark()`, decoupled from `num_shots`

`num_shots` continues to mean "shots requested from `OptimizedPTSBESimulator`" (== paper's `mi` for proportional mode, or the exhaustive-harvest driver for non-proportional). A new `baseline_num_shots: Optional[int] = None` parameter (default `None` meaning "same as `num_shots`", preserving today's non-proportional-sweep behavior unchanged) is passed to `TraditionalTrajectorySimulator`/`UnoptimizedPTSBESimulator` instead of `num_shots` when explicitly set. For proportional sweeps, callers pass a small `baseline_num_shots` (e.g. 20-50) regardless of how large `num_shots`/`mi` gets swept.

**Why this over alternatives**: (a) a fixed wall-clock budget for the baseline call instead of a fixed shot count was considered, but a shot-count cap is simpler, deterministic, and easier to reason about statistically (N shots → known throughput-estimate variance) than "however many shots fit in T seconds" (which conflates measurement noise with the timeout mechanism already used for failure detection). (b) Doing this by having callers construct `TraditionalTrajectorySimulator` directly with a different `num_shots` (bypassing `run_benchmark()`) was rejected — it would fragment the harness's JSON-output/success-tracking/geometric-stats plumbing that Figs. 3/4/5 all depend on.

### D2: No fixed time budget this run; checkpoint incrementally and report exactly what ran

Unlike the prior change's explicit ~30-minute cap, this one has no stated budget. Given per-config costs observed last session (30s-90s for small non-proportional configs, 250s+ for large ones), the full paper grid × 10 instances would take multiple hours. Rather than guessing a stopping point upfront, sweep drivers write results incrementally (as `_reproduction_run.py` already did) so work is never lost, and `validation_notes.md` documents exactly what ran each session — consistent with this project's established practice (see the two prior archived changes' honesty conventions) of never presenting partial results as complete.

### D3: bf-anomaly investigation is a small, targeted re-run, not new instrumentation

The observed reversal (bf 24→28: 1,275,393× → 1,081,521×) came from n=100/g=600, 1 instance, 10 shots, E=5 — a small enough sample that per-instance variance alone could explain it. Before treating it as a real finding, re-run the same bf sweep with more shots (e.g. 50) and/or more instances (e.g. 3) at the same (n,g), and separately at a deeper circuit (larger g) where the paper's own Fig. 4 caption notes bf's effect is "more fully realized." No new profiling code needed — `_cold_warm_from_engine()`'s existing per-call timing already provides enough detail to see whether the reversal is driven by contraction cost, harvested-bitstring count, or something else.

## Risks / Trade-offs

- **[Risk] `baseline_num_shots` being too small could make the baseline throughput estimate noisy enough to distort the speedup ratio.** → Mitigation: use the same order of magnitude (~20-50 shots) that produced stable-looking Traditional throughput numbers in last session's non-proportional runs (10 shots already gave consistent-looking per-shot timing), and sanity-check by comparing two independent baseline runs' throughput before trusting a swept mi series.
- **[Risk] No time budget could lead to an open-ended, expensive session.** → Mitigation: D2's incremental checkpointing means the work is never wasted even if stopped early; report progress against the grid periodically rather than only at the end.
- **[Trade-off] Fig. 3's 10-instances-per-cell target is expensive** (10x the cost of what ran last session per cell). If time is tight, prioritize breadth (more (n,g) cells at 1 instance) over depth (fewer cells at 10 instances) unless success-rate statistics specifically are the goal — this mirrors the paper's own point that speedup trends across g and n are the headline result, not per-cell error bars.
