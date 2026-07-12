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
The harness SHALL compute a data-collection speedup ratio defined as `throughput(optimized_PTSBE) / throughput(traditional)` for matched circuit configurations, and separately `throughput(unoptimized_PTSBE) / throughput(traditional)`. It SHALL report per-circuit speedup values and aggregate statistics using the **geometric mean and geometric standard deviation** (not arithmetic) over a configurable ensemble of random circuit instances, matching the paper's Sec. IV-D metric definition (throughput values span multiple orders of magnitude, making geometric statistics the meaningful central tendency).

#### Scenario: Speedup computed per circuit instance
- **WHEN** 10 random circuit instances are benchmarked
- **THEN** 10 individual speedup ratios are reported, one per instance

#### Scenario: Aggregate statistics use geometric mean/std
- **WHEN** the benchmark run completes
- **THEN** the output includes the geometric mean and geometric standard deviation of speedup across all successful instances, not the arithmetic mean/std

### Requirement: Configurable circuit ensemble
The harness SHALL accept circuit parameters n (qubits), g (gates), and num_instances as inputs. For each (n, g) configuration it SHALL generate `num_instances` random circuit instances matching the paper's exact circuit-generation procedure (Sec. IV-B): single-qubit gates drawn from {H, X, Y, Z, T, Rx} and two-qubit nearest-neighbor gates drawn from {CX, CY, CZ, CH, CRx}, with 20% of gates being two-qubit operations by default. Noise channels SHALL be Pauli (X/Y/Z) errors coupled to single-qubit gates and two-qubit depolarizing errors coupled to two-qubit gates, with per-gate error probability independently sampled uniformly from [0.02, 0.20].

#### Scenario: Circuit instances reused across simulators
- **WHEN** benchmarking all three simulator phases on the same configuration
- **THEN** the same pre-generated circuit instances are used for all three phases to ensure consistent comparison

#### Scenario: Error probabilities drawn uniformly from [0.02, 0.20]
- **WHEN** a random circuit instance is generated
- **THEN** each gate's error probability is independently sampled uniformly from the interval [0.02, 0.20]

#### Scenario: Gate set matches the paper
- **WHEN** a random circuit instance is generated
- **THEN** every single-qubit gate is one of {H, X, Y, Z, T, Rx} and every two-qubit gate is one of {CX, CY, CZ, CH, CRx} acting on nearest-neighbor qubits

#### Scenario: Two-qubit gate fraction defaults to 20%
- **WHEN** `generate_circuit(n, g)` is called without an explicit `two_qubit_fraction`
- **THEN** approximately 20% of the g gates are two-qubit gates

#### Scenario: Noise channel type matches gate arity
- **WHEN** a circuit instance is generated
- **THEN** every single-qubit gate is coupled to a Pauli error channel and every two-qubit gate is coupled to a two-qubit depolarizing error channel

### Requirement: Results output format
The harness SHALL write benchmark results to a JSON file containing per-circuit throughput values, speedup ratios, circuit metadata (n, g, instance_id), simulator configuration (batch sizes, hypersamples, E), a per-instance success/failure flag, and hardware context (GPU device name and total device memory). It SHALL print a summary table to stdout including the GPU device name, geometric mean and geometric standard deviation of speedup per (n, g) configuration, and the fraction of instances that succeeded.

#### Scenario: JSON output contains required fields
- **WHEN** a benchmark run completes and results are saved
- **THEN** the JSON file contains keys: `n`, `g`, `instance_id`, `throughput_traditional`, `throughput_unoptimized`, `throughput_optimized`, `speedup_unoptimized_vs_traditional`, `speedup_optimized_vs_traditional`, `success`

#### Scenario: Summary table printed to stdout
- **WHEN** the benchmark script runs
- **THEN** a table with columns [n, g, geomean_speedup, geostd_speedup, success_rate] is printed to stdout upon completion

#### Scenario: Hardware context recorded in output
- **WHEN** a benchmark run completes on a GPU
- **THEN** the JSON output includes `gpu_device_name` and `gpu_memory_total_bytes`

### Requirement: Configuration success/failure tracking
The harness SHALL mark each circuit instance's run as succeeded or failed, where failure means the run exceeded a configurable time or memory budget. A configuration (n, g) SHALL be reported with a success-rate fraction across its instances. Configurations where more than 80% of instances succeed SHALL be marked as reliable results; configurations where fewer than 80% succeed SHALL be marked accordingly, matching the paper's Fig. 3/5/6 solid-vs-hollow-marker convention.

#### Scenario: Failed instance recorded, not silently dropped
- **WHEN** a circuit instance's simulation run exceeds the configured time or memory budget
- **THEN** that instance is recorded in the results with `success=False` rather than omitted, and does not contribute to the geometric mean/std of successful instances

#### Scenario: Configuration-level success rate computed
- **WHEN** all instances for a given (n, g) configuration have been run
- **THEN** the fraction of instances with `success=True` is computed and included in the printed summary and JSON output

### Requirement: Proportional-mode benchmarking
The harness SHALL support benchmarking `OptimizedPTSBESimulator` in proportional mode (in addition to the existing non-proportional default), accepting a `mode` parameter forwarded to the simulator's `sample()` call, so that proportional-mode speedup (paper Fig. 5) can be measured against the same Traditional baseline.

#### Scenario: Proportional mode produces distinct results from non-proportional
- **WHEN** `run_benchmark(..., mode="proportional")` is called
- **THEN** `OptimizedPTSBESimulator.sample()` is invoked with `mode="proportional"` and the resulting throughput/speedup reflects proportional NBS behavior

### Requirement: Batch-size sweep benchmarking
The harness SHALL support sweeping the non-final batch size `bj` for a fixed circuit configuration and reporting per-batch contraction+sampling time, to reproduce the paper's Fig. 7 (per-batch cost vs. batch size).

#### Scenario: Batch-size sweep produces one timing point per batch size
- **WHEN** a batch-size sweep is run over `bj` in {2, 5, 10, 15, 20, 24, 28} for a fixed (n, g) circuit
- **THEN** one per-batch contraction+sampling time is recorded for each `bj` value

### Requirement: Phase comparison support
The harness SHALL benchmark all three phases (traditional, unoptimized PTSBE, optimized PTSBE) within a single run on the same circuit instances, enabling direct three-way throughput comparison.

#### Scenario: All three phases run in one benchmark invocation
- **WHEN** `run_benchmark(n=100, g=600, num_instances=10)` is called
- **THEN** all three simulator phases are evaluated and their throughputs recorded for each of the 10 instances
