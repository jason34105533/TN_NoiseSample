## MODIFIED Requirements

### Requirement: Honest reporting of results below paper's claims
Figure and summary generation SHALL report the actual measured geometric-mean speedups alongside the paper's headline claims (~10⁸× non-proportional, ~10³× proportional) without altering, filtering, or cherry-picking results to match those claims. Any configuration that fails to complete (per the benchmarking-harness's success-tracking requirement) SHALL be visibly marked as such, not silently omitted from figures. This SHALL extend to qualitative trend divergences, not only magnitude shortfalls: if a measured trend runs in the opposite direction from what the paper reports (e.g. speedup decreasing rather than increasing with a swept parameter), that reversal SHALL be reported explicitly in `validation_notes.md` as an observed finding, rather than omitted or reframed to imply agreement with the paper's direction.

#### Scenario: Divergence from paper's claims is visible, not hidden
- **WHEN** measured speedups for a given configuration are below the paper's reported range
- **THEN** the figure and any accompanying summary text report the actual measured value, not an adjusted or paper-matching value

#### Scenario: Trend reversal is reported explicitly, not reframed
- **WHEN** a swept-parameter sweep (e.g. final_batch_size) shows speedup moving in the opposite direction from the paper's reported trend
- **THEN** `validation_notes.md` states the reversal plainly as a measured finding, including enough detail (sample size, configuration) for a reader to judge whether it's likely a real effect or small-sample noise
