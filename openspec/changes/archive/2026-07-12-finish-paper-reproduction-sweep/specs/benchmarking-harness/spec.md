## MODIFIED Requirements

### Requirement: Phase comparison support
The harness SHALL benchmark all three phases (traditional, unoptimized PTSBE, optimized PTSBE) within a single run on the same circuit instances, enabling direct three-way throughput comparison. The harness SHALL accept an independent `baseline_num_shots` parameter for the traditional and unoptimized baselines, separate from the shot count requested from optimized PTSBE (`num_shots`, which doubles as the paper's swept `mi` parameter in proportional mode). When `baseline_num_shots` is not specified, it SHALL default to `num_shots` (preserving prior behavior for non-proportional sweeps where all three phases use the same shot count). This decoupling exists because throughput is a per-shot rate: the baseline only needs enough shots to estimate that rate stably, while `num_shots`/`mi` for optimized PTSBE is the figure parameter actually being swept and may be orders of magnitude larger.

#### Scenario: All three phases run in one benchmark invocation
- **WHEN** `run_benchmark(n=100, g=600, num_instances=10)` is called
- **THEN** all three simulator phases are evaluated and their throughputs recorded for each of the 10 instances

#### Scenario: Baseline shot count defaults to num_shots
- **WHEN** `run_benchmark(..., num_shots=1000)` is called without specifying `baseline_num_shots`
- **THEN** Traditional and Unoptimized PTSBE are each run with 1000 shots, matching prior behavior

#### Scenario: Baseline shot count can be set independently for proportional sweeps
- **WHEN** `run_benchmark(..., num_shots=10000, baseline_num_shots=20, mode="proportional")` is called
- **THEN** Traditional and Unoptimized PTSBE are each run with 20 shots (for a stable throughput-rate estimate) while `OptimizedPTSBESimulator` is run with the full 10000 shots requested
