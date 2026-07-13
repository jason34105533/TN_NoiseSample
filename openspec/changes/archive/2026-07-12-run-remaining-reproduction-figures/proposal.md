## Why

Two prior changes (`gpu-bounded-memory-contraction`, `finish-paper-reproduction-sweep`) got the real GPU contraction engine, benchmark harness, and circuit generator fully paper-conformant, and produced real data for Figs. 3 and 6 (non-proportional speedup, 19/25 grid cells) and Fig. 4 (final-batch-size sweep). Three pieces of the reproduction remain purely execution work — no more engine or harness code is needed, per the last change's spec sync (the `baseline_num_shots` fix that unblocks proportional sweeps is already in `openspec/specs/benchmarking-harness/spec.md`): the proportional sweep (Fig. 5), the batch-size sweep (Fig. 7), and re-checking an unexplained anomaly from the Fig. 4 run (speedup *decreasing* with larger final_batch_size, opposite the paper's direction). Driver scripts for all three (`benchmarks/_prop_sweep.py`, `_batch_sweep.py`, `_bf_anomaly_check.py`) were already written last session and are ready to run.

## What Changes

- Run the proportional sweep (`benchmarks/_prop_sweep.py` or equivalent): the paper's two reference configs (n=100,g=600 and n=200,g=1000) across mi∈{10,100,1000,10000}, using the already-fixed `baseline_num_shots` decoupling. Generate `benchmarks/figures/fig5_proportional_speedup.png`.
- Run the batch-size sweep (`benchmarks/_batch_sweep.py`): n=100,g=600 across bj∈{2,5,10,15,20,24,28}. Generate `benchmarks/figures/fig7_batch_size_sweep.png`.
- Run the bf-anomaly follow-up (`benchmarks/_bf_anomaly_check.py`): more shots/instances at n=100,g=600, plus a deeper circuit (g=1000), to check whether the previously-observed reversal (speedup decreasing with larger bf) persists or was small-sample noise.
- Update `benchmarks/validation_notes.md` with a new dated section covering all three, reporting real measured numbers regardless of whether they match the paper's direction/magnitude (per the existing `paper-figure-reproduction` capability's "Honest reporting" requirement — no spec change needed, just following it).
- Re-verify the driver scripts still work against current `main` before relying on them (they were written before the last change's final commits landed).

**Explicitly out of scope**: extending the non-proportional grid further (19/25 cells is already real, substantial data; not the focus of this change), any change to the contraction engine, simulators, or circuit generator (all validated and unchanged since the prior two changes), the n=75-vs-other-n g-trend divergence noted in the last change's writeup (a separate, deeper investigation — not blocking these three figures).

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `paper-figure-reproduction`: the "Honest reporting of results below paper's claims" requirement is broadened to explicitly cover qualitative trend reversals (a measured sweep direction opposite the paper's), not only magnitude shortfalls — prompted by the concrete bf-sweep anomaly from the prior session (speedup decreasing rather than increasing with `final_batch_size`) that this change re-checks. Everything else this change needs (`baseline_num_shots`, `run_batch_size_sweep()`, the Fig. 5/Fig. 7 figure-generation requirements) is already specified and unchanged — this change is otherwise pure execution against existing specs.

## Impact

- **Code**: none expected in `src/tn_noise_sim/` or `benchmarks/run_benchmark.py`/`plots.py` — all required functionality already exists and is spec'd. Driver scripts (`benchmarks/_prop_sweep.py`, `_batch_sweep.py`, `_bf_anomaly_check.py`) may need minor fixes if they've drifted from current `main`, but no new capability code.
- **Compute**: real H100 GPU time for three sweeps; rough scale from last session — the proportional sweep's smallest point (mi=10) took ~214s, batch-size sweep points are similar order to non-proportional grid cells (tens of seconds to a few minutes each), bf-anomaly re-check adds more instances/shots on top of last session's ~110-220s-per-bf-value baseline.
- **Docs**: `benchmarks/validation_notes.md` updated; `benchmarks/figures/fig5_proportional_speedup.png` and `fig7_batch_size_sweep.png` created; a bf-anomaly finding documented either way.
