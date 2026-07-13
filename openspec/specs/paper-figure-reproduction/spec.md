# Paper Figure Reproduction

## Purpose

Reproduces the reference paper's (arXiv:2604.08467) key figures (Figs. 3-7) from benchmark result JSON, reporting actual measured values without adjustment to match the paper's headline claims.

## Requirements

### Requirement: Non-proportional speedup figure (paper Fig. 3)
The system SHALL provide a script that, given benchmark result JSON covering the paper's (n, g) grid (n∈{50,75,100,150,200}, g∈{200,400,600,800,1000}) in non-proportional mode, plots data-collection speedup (optimized PTSBE / traditional) vs. g, with one series per n, using geometric mean markers and geometric-standard-deviation error bars, and distinguishing configurations with ≥80% vs. <80% instance success rate via marker fill.

#### Scenario: Figure generated from results JSON
- **WHEN** the figure script is run against a results file containing the full (n, g) grid in non-proportional mode
- **THEN** a plot is produced with g on the x-axis, speedup on the y-axis (log scale), one line per n, and hollow markers for configurations below 80% success

### Requirement: Final-batch-size sweep figure (paper Fig. 4)
The system SHALL provide a script that plots PTSBE throughput (shots/s) vs. g for multiple final batch sizes `bf` (e.g. 24, 26, 28) at a fixed n, from benchmark results covering those `bf` values.

#### Scenario: Figure generated from bf sweep results
- **WHEN** the figure script is run against results covering `bf` in {24, 26, 28} at fixed n
- **THEN** a plot is produced with g on the x-axis, throughput on the y-axis (log scale), one line per `bf`

### Requirement: Proportional speedup figure (paper Fig. 5)
The system SHALL provide a script that plots proportional PTSBE data-collection speedup vs. shot count `mi`, for the paper's two reference configurations (n=100,g=600 and n=200,g=1000), from benchmark results run in proportional mode across a range of `mi` values.

#### Scenario: Figure generated from proportional sweep results
- **WHEN** the figure script is run against proportional-mode results covering multiple `mi` values for both reference configurations
- **THEN** a plot is produced with `mi` on the x-axis (log scale), speedup on the y-axis (log scale), one line per configuration

### Requirement: Contraction/path-finding time figure (paper Fig. 6)
The system SHALL provide a script that plots (a) contraction time per unique shot vs. g, (b) path-finding time vs. g, and (c) the ratio of path-finding time to contraction time vs. g, each with one series per n, from benchmark results that record per-call cold (first-query) and warm (subsequent-query) timing as described in the contraction-engine capability's path-finding-cost measurement approach.

#### Scenario: Figure generated with path-finding/contraction split
- **WHEN** the figure script is run against results that include per-config cold/warm contraction timing
- **THEN** three subplots are produced showing contraction time per shot, path-finding time, and their ratio, each vs. g with one line per n

### Requirement: Per-batch contraction cost figure (paper Fig. 7)
The system SHALL provide a script that plots per-batch contraction+sampling time vs. batch size `bj`, for the n=100, g=600 reference configuration, from batch-size-sweep benchmark results.

#### Scenario: Figure generated from batch-size sweep
- **WHEN** the figure script is run against batch-size-sweep results for n=100, g=600
- **THEN** a plot is produced with `bj` on the x-axis, per-batch contraction+sampling time on the y-axis (log scale), one point per swept batch size

### Requirement: Honest reporting of results below paper's claims
Figure and summary generation SHALL report the actual measured geometric-mean speedups alongside the paper's headline claims (~10⁸× non-proportional, ~10³× proportional) without altering, filtering, or cherry-picking results to match those claims. Any configuration that fails to complete (per the benchmarking-harness's success-tracking requirement) SHALL be visibly marked as such, not silently omitted from figures. This SHALL extend to qualitative trend divergences, not only magnitude shortfalls: if a measured trend runs in the opposite direction from what the paper reports (e.g. speedup decreasing rather than increasing with a swept parameter), that reversal SHALL be reported explicitly in `validation_notes.md` as an observed finding, rather than omitted or reframed to imply agreement with the paper's direction.

#### Scenario: Divergence from paper's claims is visible, not hidden
- **WHEN** measured speedups for a given configuration are below the paper's reported range
- **THEN** the figure and any accompanying summary text report the actual measured value, not an adjusted or paper-matching value

#### Scenario: Trend reversal is reported explicitly, not reframed
- **WHEN** a swept-parameter sweep (e.g. final_batch_size) shows speedup moving in the opposite direction from the paper's reported trend
- **THEN** `validation_notes.md` states the reversal plainly as a measured finding, including enough detail (sample size, configuration) for a reader to judge whether it's likely a real effect or small-sample noise
