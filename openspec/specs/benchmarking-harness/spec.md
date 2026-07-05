# Benchmarking Harness

## Purpose

Measures and compares throughput across the three simulator phases (traditional trajectory, unoptimized PTSBE, optimized PTSBE) on matched circuit configurations, computing speedup ratios and reporting results consistent with the paper's experimental methodology.

## Requirements

### Requirement: Throughput measurement
The benchmarking harness SHALL measure throughput as unique labeled bitstrings produced per GPU wall-clock second. Timing SHALL begin after error pre-sampling and path-finding (for optimized PTSBE) and cover only the contraction + sampling loop. Timing SHALL be collected using CUDA events for GPU-accurate measurement.

#### Scenario: Throughput excludes path-finding time
- **WHEN** benchmarking `OptimizedPTSBESimulator`
- **THEN** the path-finding duration is excluded from the throughput denominator, consistent with the paper's metric definition

#### Scenario: Throughput includes error injection and sampling
- **WHEN** benchmarking any simulator
- **THEN** error tensor injection, contraction, and bitstring sampling are all included in the timed region

### Requirement: Speedup ratio computation
The harness SHALL compute a data-collection speedup ratio defined as `throughput(optimized_PTSBE) / throughput(traditional)` for matched circuit configurations. It SHALL report per-circuit speedup values and aggregate statistics (mean, std, min, max) over a configurable ensemble of random circuit instances.

#### Scenario: Speedup computed per circuit instance
- **WHEN** 10 random circuit instances are benchmarked
- **THEN** 10 individual speedup ratios are reported, one per instance

#### Scenario: Aggregate statistics reported
- **WHEN** the benchmark run completes
- **THEN** the output includes mean, standard deviation, min, and max speedup across all instances

### Requirement: Configurable circuit ensemble
The harness SHALL accept circuit parameters n (qubits), g (gates), and num_instances as inputs. For each (n, g) configuration it SHALL generate `num_instances` random circuit instances using random single- and two-qubit gates with randomly sampled per-gate error probabilities in [0.02, 0.20], matching the paper's experimental setup.

#### Scenario: Circuit instances reused across simulators
- **WHEN** benchmarking all three simulator phases on the same configuration
- **THEN** the same pre-generated circuit instances are used for all three phases to ensure consistent comparison

#### Scenario: Error probabilities drawn uniformly from [0.02, 0.20]
- **WHEN** a random circuit instance is generated
- **THEN** each gate's error probability is independently sampled uniformly from the interval [0.02, 0.20]

### Requirement: Results output format
The harness SHALL write benchmark results to a JSON file containing per-circuit throughput values, speedup ratios, circuit metadata (n, g, instance_id), and simulator configuration (batch sizes, hypersamples, E). It SHALL also print a summary table to stdout in a format matching Table I of the reference paper.

#### Scenario: JSON output contains required fields
- **WHEN** a benchmark run completes and results are saved
- **THEN** the JSON file contains keys: `n`, `g`, `instance_id`, `throughput_traditional`, `throughput_unoptimized`, `throughput_optimized`, `speedup_unoptimized_vs_traditional`, `speedup_optimized_vs_traditional`

#### Scenario: Summary table printed to stdout
- **WHEN** the benchmark script runs
- **THEN** a table with columns [n, g, mean_speedup, std_speedup] is printed to stdout upon completion

### Requirement: Phase comparison support
The harness SHALL benchmark all three phases (traditional, unoptimized PTSBE, optimized PTSBE) within a single run on the same circuit instances, enabling direct three-way throughput comparison.

#### Scenario: All three phases run in one benchmark invocation
- **WHEN** `run_benchmark(n=100, g=600, num_instances=10)` is called
- **THEN** all three simulator phases are evaluated and their throughputs recorded for each of the 10 instances
