## MODIFIED Requirements

### Requirement: Results output format
The harness SHALL write benchmark results to a JSON file containing per-circuit throughput values, speedup ratios, circuit metadata (n, g, instance_id), simulator configuration (batch sizes, hypersamples, `num_error_sets`), a per-instance success/failure flag, and hardware context (GPU device name and total device memory). It SHALL print a summary table to stdout including the GPU device name, geometric mean and geometric standard deviation of speedup per (n, g) configuration, and the fraction of instances that succeeded.

#### Scenario: JSON output contains required fields
- **WHEN** a benchmark run completes and results are saved
- **THEN** the JSON file contains keys: `n`, `g`, `instance_id`, `throughput_traditional`, `throughput_unoptimized`, `throughput_optimized`, `speedup_unoptimized_vs_traditional`, `speedup_optimized_vs_traditional`, `success`, `num_error_sets`

#### Scenario: Summary table printed to stdout
- **WHEN** the benchmark script runs
- **THEN** a table with columns [n, g, geomean_speedup, geostd_speedup, success_rate] is printed to stdout upon completion

#### Scenario: Hardware context recorded in output
- **WHEN** a benchmark run completes on a GPU
- **THEN** the JSON output includes `gpu_device_name` and `gpu_memory_total_bytes`, so results from different GPU models (e.g. V100 vs H100) are distinguishable without cross-referencing external notes

#### Scenario: num_error_sets recorded for amortization diagnosis
- **WHEN** a benchmark instance fails or succeeds
- **THEN** the JSON record includes `num_error_sets` (E), so a failure caused by too few baseline shots per pre-sampled error set (no path-finding amortization for Unoptimized PTSBE) can be diagnosed directly from the record, without cross-referencing the calling script
