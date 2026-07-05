## MODIFIED Requirements

### Requirement: Results output format
The harness SHALL write benchmark results to a JSON file containing per-circuit throughput values, speedup ratios, circuit metadata (n, g, instance_id), simulator configuration (batch sizes, hypersamples, E), and hardware context (GPU device name and total device memory). It SHALL also print a summary table to stdout in a format matching Table I of the reference paper, including the GPU device name used for the run.

#### Scenario: JSON output contains required fields
- **WHEN** a benchmark run completes and results are saved
- **THEN** the JSON file contains keys: `n`, `g`, `instance_id`, `throughput_traditional`, `throughput_unoptimized`, `throughput_optimized`, `speedup_unoptimized_vs_traditional`, `speedup_optimized_vs_traditional`

#### Scenario: Summary table printed to stdout
- **WHEN** the benchmark script runs
- **THEN** a table with columns [n, g, mean_speedup, std_speedup] is printed to stdout upon completion

#### Scenario: Hardware context recorded in output
- **WHEN** a benchmark run completes on a GPU
- **THEN** the JSON output includes `gpu_device_name` and `gpu_memory_total_bytes`, so results from different GPU models (e.g. V100 vs H100) are distinguishable without cross-referencing external notes
