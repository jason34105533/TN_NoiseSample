## Why

A thorough re-read of the paper (arXiv:2604.08467, Sec. V-A) against this project's specs and prior bf-sweep results found a real discrepancy: **the paper's Fig. 4 final-batch-size sweep is explicitly run at n=200** ("This effect of sample efficiency by final batch-size bf is demonstrated for n = 200 systems in Fig. 4"), with g swept across the paper's normal grid and bf∈{24,26,28} as three separate lines. Every bf sweep run in this project so far (two sessions) used **n=100**, and only a single g value (600) — not the multi-g sweep the paper's own Fig. 4 actually is. The "anomaly" flagged twice already (speedup decreasing with bf, opposite the paper's direction) was measured in a regime the paper never made a claim about; it may not be a real contradiction at all, just the wrong experiment. `plots.py::plot_fig4_bf_sweep()` was already built correctly (groups by bf, plots vs. g) — it was simply never fed the right data.

Separately, Fig. 5 (proportional sweep) is down to exactly one missing point: n=200,g=1000,mi=10000, which was mid-run when the last session ended.

## What Changes

- Run a corrected Fig. 4 reproduction: n=200 (matching the paper's actual configuration), g swept across a subset of the paper's grid (g∈{200,600,1000}, chosen for tractability — n=200 configs are expensive, ~150-350s per single point based on prior sessions' data), bf∈{24,26,28}. This is a real g×bf grid (9 points), not a single-g snapshot.
- Regenerate `benchmarks/figures/fig4_bf_sweep.png` from this corrected dataset, at n=200 to match the paper.
- Complete the one remaining Fig. 5 point: n=200,g=1000,mi=10000.
- Regenerate `benchmarks/figures/fig5_proportional_speedup.png` — this closes out both reference configs' full mi range (10-10000), matching the paper's own Fig. 5 setup exactly for the first time.
- Update `benchmarks/validation_notes.md`: report whether the paper's bf-increasing-throughput trend holds at n=200 (the paper's actual tested regime), explicitly distinguishing this from the earlier n=100 "anomaly" finding rather than conflating the two. If the n=200 result still doesn't match the paper's direction, that's a stronger, more paper-comparable finding worth flagging clearly; if it does match, that resolves the anomaly as an artifact of testing the wrong n rather than a real discrepancy.

**Explicitly out of scope**: the SIGALRM timeout-preemption limitation (unfixed, real, separate concern), the n=75-vs-other-n g-trend divergence from an earlier session, deepening the non-proportional grid to 10 instances/cell. If time runs out before covering the full g∈{200,600,1000} × bf∈{24,26,28} grid at n=200, prioritize completing Fig. 5's missing point first (well-defined, bounded cost), then as much of the corrected Fig. 4 grid as time allows — partial real data reported honestly, per this project's established convention, rather than skipped silently.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `paper-figure-reproduction`: the Fig. 4 requirement currently says "at a fixed n" without specifying which — underspecified enough that two prior sessions ran the bf sweep at n=100 instead of the paper's actual n=200. Pins the requirement to n=200 (per Sec. V-A) and adds a scenario requiring non-paper-n bf-sweep data to be explicitly labeled as such rather than presented as a Fig. 4 reproduction. `plot_fig4_bf_sweep()`, `plot_fig5_proportional()`, and `run_benchmark()` already support everything else this change needs (an `n` filter on the Fig. 4 plot function, arbitrary `final_batch_size`/`mode`/`num_shots` parameters on `run_benchmark()`) — otherwise this is pure execution.

## Impact

- **Code**: none expected in `src/tn_noise_sim/` or `benchmarks/*.py` — this is a data-collection correction, not an engineering change.
- **Compute**: real H100 GPU time. n=200 configs are the most expensive tier seen so far (single proportional points there have taken 400-1300s depending on mi); a 9-point non-proportional grid at n=200 (cheaper per-point than proportional mode, based on Fig. 3 data showing n=200 non-proportional points completing in the 150-350s range) plus the one Fig. 5 point is a substantial but bounded amount of GPU time.
- **Docs**: `benchmarks/validation_notes.md` updated with the corrected Fig. 4 finding and the completed Fig. 5 dataset; both figures regenerated.
