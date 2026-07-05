## 1. Project Scaffolding

- [x] 1.1 Initialize `src/` package with `__init__.py` and module stubs for each capability
- [x] 1.2 Create `pyproject.toml` / `setup.py` with dependencies: `cuquantum-python`, `cupy-cuda12x`, `qiskit`, `numpy`, `pytest`
- [x] 1.3 Create `tests/` directory with `conftest.py` and a small test circuit fixture (5 qubits, 10 gates, numpy backend)
- [x] 1.4 Create `benchmarks/` directory with a top-level `run_all.py` entry point

## 2. Noise Model and Error Sampling

- [x] 2.1 Implement `NoiseModel` dataclass: stores per-gate error type (pauli / depolarizing) and probability
- [x] 2.2 Implement `ErrorSampler.sample()` for proportional mode: weighted sampling of error operator sets and m_i assignment
- [x] 2.3 Implement `ErrorSampler.sample()` for non-proportional (uniform) mode: uniform sampling and equal m_i assignment
- [x] 2.4 Implement Pauli error operator draw for single-qubit gates (I/X/Y/Z by probability)
- [x] 2.5 Implement depolarizing error operator draw for two-qubit gates (15 non-identity Paulis uniform over p)
- [x] 2.6 Implement `ErrorSampler.to_list()` serialization to list of `{gate_idx: np.ndarray}` dicts
- [x] 2.7 Write unit tests for both sampling modes: verify shot count sums, operator shapes, probability weights

## 3. Tensor Network Builder

- [x] 3.1 Implement `TensorNetworkBuilder.build(circuit)`: parse Qiskit `QuantumCircuit`, extract gate unitaries, reshape to rank-2 / rank-4 tensors, assign index labels
- [x] 3.2 Add initial |0⟩ state tensors and measurement projector tensors to complete the network
- [x] 3.3 Implement `mode="insert"` error operator insertion: add k_p^i as a separate adjacent tensor node per noisy gate
- [x] 3.4 Implement `mode="fuse"` UPV error fusion: matrix-multiply k_p^i into d_l, replace gate tensor in-place, verify topology invariance
- [x] 3.5 Write unit tests: verify noiseless and `mode="fuse"` networks have identical index graphs; verify `mode="insert"` adds correct number of extra nodes

## 4. Contraction Engine

- [x] 4.1 Implement `ContractionPathCache` with `get(topology_hash)` and `put(topology_hash, path)` using a dict; implement topology hash from tensor shapes + index labels
- [x] 4.2 Implement `ContractionEngine.find_path(network, num_hypersamples)`: call `cuquantum.cutensornet` path optimizer, store result in cache
- [x] 4.3 Implement `ContractionEngine.contract_batch(network, path, batch_index, batch_size, prefix=())`: contract b qubits at batch_index, apply prefix projection if provided, return CuPy 2^b vector
- [x] 4.4 Add `final_batch_size` support: detect final batch (batch_index == f), use `final_batch_size` instead of `batch_size`
- [x] 4.5 Implement conditional marginal via qubit index projection for proportional NBS (slice the marginal tensor at observed prefix values)
- [x] 4.6 Write unit tests (numpy/CPU fallback): verify marginal sums to 1.0; verify conditioning on prefix selects correct conditional distribution for a 3-qubit test circuit

## 5. Phase 1 — Traditional Trajectory Simulator

- [x] 5.1 Implement `BaseTNSimulator` abstract base class with `sample(circuit, noise_model, num_shots) -> list[(str, int)]` interface
- [x] 5.2 Implement `TraditionalTrajectorySimulator(batch_size=24)`: per-shot loop that samples one error set, builds network with `mode="insert"`, finds path (no cache), contracts all f batches, samples one bitstring
- [x] 5.3 Write integration test: run 10 shots on a 4-qubit, 8-gate circuit; verify return type, bitstring length, and that path finding was called 10 times

## 6. Phase 2 — Unoptimized PTSBE Simulator

- [x] 6.1 Implement `UnoptimizedPTSBESimulator(batch_size=24, num_error_sets=None)`: pre-sample E error sets via `ErrorSampler`, then for each K_i build network with `mode="insert"`, find path once, run m_i-shot loop
- [x] 6.2 Wire `num_error_sets` default to `total_shots` when not specified
- [x] 6.3 Write integration test: verify path finding is called exactly E times (not m); verify return list length equals total shots

## 7. Phase 3 — Optimized PTSBE Simulator (Core)

- [x] 7.1 Implement `OptimizedPTSBESimulator(batch_size=10, final_batch_size=28, num_hypersamples=100, num_error_sets=None)` constructor and pre-sampling logic
- [x] 7.2 Implement UPV path finding: build noiseless network, call `find_path()` once with 100 hypersamples, cache result
- [x] 7.3 Implement non-proportional NBS inner loop: for each K_i fuse errors (`mode="fuse"`), run non-final batches with prefix branching, run final batch with exhaustive sampling above `min_prob` threshold
- [x] 7.4 Implement proportional NBS inner loop: maintain unique prefix set across batches, contract one marginal per unique prefix, sample count(prefix) continuations, merge into final bitstring list
- [x] 7.5 Implement exhaustive final-batch extraction: from the 2^{b_f} CuPy probability vector, return all (index, prob) pairs where prob >= min_prob as complete bitstrings
- [x] 7.6 Write integration test: verify path finding called exactly once across all E error sets; verify non-proportional mode returns >> m_i bitstrings per error set; verify proportional output distribution matches expected Born rule (KL divergence test on small circuit)

## 8. Benchmarking Harness

- [x] 8.1 Implement random circuit generator: n qubits, g gates (random single/two-qubit gates), per-gate error probabilities uniform in [0.02, 0.20]
- [x] 8.2 Implement CUDA event-based wall-clock timing wrapper that brackets the contraction+sampling loop only
- [x] 8.3 Implement `run_benchmark(n, g, num_instances, simulators)`: generate instances, run all three simulators per instance, record throughput (bitstrings / GPU-second) for each
- [x] 8.4 Implement speedup ratio computation: `throughput(optimized) / throughput(traditional)` per instance; aggregate mean/std/min/max
- [x] 8.5 Implement JSON results writer with required fields: n, g, instance_id, all three throughputs, two speedup ratios
- [x] 8.6 Implement stdout summary table printer (columns: n, g, mean_speedup, std_speedup)
- [x] 8.7 Add `benchmarks/run_all.py` CLI with `--n`, `--g`, `--instances`, `--output` flags; include default configs matching paper: (n=100,g=600) and (n=200,g=1000)

## 9. Validation Against Paper

- [ ] 9.1 Run proportional PTSBE benchmark for n=100,g=600 and n=200,g=1000 with 10 instances; confirm speedup >= 100× (paper claims up to 1000×)
- [ ] 9.2 Run non-proportional PTSBE benchmark for same configs; confirm speedup >= 10^6× (paper claims up to 10^8×)
- [ ] 9.3 Reproduce Fig. 7 batch-size sweep: measure contraction time per batch for b ∈ {2,5,10,15,20,24,28} on n=100,g=600 circuit; verify b=10 is near-optimal for non-final batches
- [x] 9.4 Document any deviation from paper's claimed numbers with explanation (hardware difference, random seed, circuit instance variation)
