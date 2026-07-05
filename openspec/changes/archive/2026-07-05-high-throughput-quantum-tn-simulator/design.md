## Context

This project implements the three-phase tensor network trajectory simulator pipeline from Patti et al., "Accelerating Quantum Tensor Network Simulations with Unified Path Variations and Non-Degenerate Batched Sampling" (arXiv:2604.08467). The authors did not open-source their code. The goal is a faithful from-scratch Python implementation using the same hardware stack (cuQuantum cuTensorNet, CuPy, Qiskit IR) that reproduces their claimed throughput numbers.

The fundamental computational target is simulating noisy quantum circuits of n qubits and g gates under a Pauli/depolarizing noise model. The output is a labeled dataset of bitstrings (quantum measurements) associated with specific error configurations — training data for quantum error correction and AI models.

Key bottlenecks in traditional trajectory simulation (that this project resolves):
1. **Per-shot path finding**: CPU contraction path search is O(seconds) per call; extracting m shots means m such calls.
2. **Sequential single-shot extraction**: Each full f-batch contraction loop yields exactly 1 shot.
3. **Fixed batch sizes**: CUDA-Q hard-codes b=24 qubits; near-optimal is b=10 for non-final batches, b=28 for final.

## Goals / Non-Goals

**Goals:**
- Implement Phase 1 (traditional), Phase 2 (unoptimized PTSBE), and Phase 3 (optimized PTSBE) as independent, comparable simulator classes.
- Implement UPV: error tensor fusion into coherent gate tensors before path finding, so one cached path covers all E error sets.
- Implement NBS: proportional mode (conditional marginal branching per unique prefix) and non-proportional mode (exhaustive final-batch sampling above a threshold).
- Expose `bj` (non-final batch size) and `bf` (final batch size) as first-class constructor parameters.
- Provide a benchmarking harness that reports throughput (unique bitstrings / GPU-second) and speedup ratios.
- Match experimental parameters from the paper: n ∈ {50,75,100,150,200}, g up to 1000, bf=28, bj=10, 100 hypersamples for PTSBE path finding vs. 1 for baseline.

**Non-Goals:**
- Lightcone simplification (explicitly noted as future work in the paper).
- Multi-node multi-GPU intra-error-set parallelism (paper notes this as future work for bf > 28).
- Statevector (non-tensor-network) PTSBE — only TN methods are in scope.
- Variational quantum algorithm support or circuit optimization.
- Production-quality CLI / GUI.

## Decisions

### D1: Python as the implementation language
**Decision**: Pure Python with NumPy/CuPy array manipulation; cuQuantum Python bindings for contraction.
**Rationale**: cuQuantum exposes a Python API (`cuquantum.cutensornet`) that matches the paper's stack. A pure-Python layer keeps implementation accessible and debuggable.
**Alternative considered**: C++/CUDA directly — rejected because it would double implementation time with no algorithmic difference.

### D2: Qiskit as the circuit IR
**Decision**: Use Qiskit's `QuantumCircuit` to represent circuits; extract gate tensors and connectivity from it.
**Rationale**: The paper uses "a Qiskit intermediate representation" for this exact purpose. Qiskit is widely used in the quantum computing community and provides the gate unitaries we need.
**Alternative considered**: Custom circuit IR — rejected as unnecessary complexity.

### D3: UPV via pre-contraction tensor fusion
**Decision**: Before path finding, each error operator `k_p^i` is matrix-multiplied into its adjacent coherent gate tensor `d_l`, producing a fused tensor with the same index structure. Path finding runs once on the fused noiseless network; the fused gate tensors are swapped per error set at contraction time.
**Rationale**: Fusing preserves tensor network topology (same number of legs, same dimensions) so the path found on the noiseless network is valid for all error combinations. This is exactly the UPV mechanism from §III-A of the paper.
**Alternative considered**: Adding error tensors as extra nodes — this changes topology and forces per-error-set path finding (the unoptimized approach).

### D4: NBS proportional mode via conditional marginal tree
**Decision**: Maintain a list of unique `(s1,...,sj-1)` prefixes; for each new batch Bj, contract one marginal per unique prefix, then sample from it. The prefix list grows combinatorially but is pruned by the user-specified branching factor.
**Rationale**: This is §III-B proportional mode: deduplicates contractions across identical prefix histories rather than repeating all f batches per shot.

### D5: NBS non-proportional mode with exhaustive final-batch sampling
**Decision**: For non-final batches, sample one (or more) prefixes per contraction. For the final batch Bf, contract once per unique prefix and return **all** entries of the 2^bf probability vector above a user-specified probability threshold, along with their indices.
**Rationale**: Final-batch bitstrings require no subsequent computation, so harvesting all of them costs only the memory to store the 2^bf vector (28 qubits → 256M entries on a single H100). This is what drives the 10^8× speedup for non-proportional sampling.

### D6: Simulator class hierarchy
**Decision**: Three concrete classes — `TraditionalTrajectorySimulator`, `UnoptimizedPTSBESimulator`, `OptimizedPTSBESimulator` — all implementing a common `BaseTNSimulator` interface with a `sample(circuit, noise_model, num_shots)` method.
**Rationale**: Enables apples-to-apples benchmarking via polymorphism; isolates algorithmic differences; makes Phase 1 and Phase 2 useful as documented reference implementations.

### D7: Contraction path caching
**Decision**: A `ContractionPathCache` singleton stores `(network_topology_hash → (path, slices, workspace_info))` tuples. For Phase 3, this hash is computed on the error-fused noiseless network, which is identical across all error sets.
**Rationale**: UPV's benefit is moot unless the path is actually cached and reused. Topology hash (index names + tensor shapes) is sufficient to key the cache.

## Risks / Trade-offs

- **GPU memory for bf=28**: A single 2^28 float32 probability vector is 1 GB. Combined with tensor workspace, this approaches H100 80GB limits for large n. → Mitigation: cap bf at 28 as the paper does; document the limit explicitly.
- **Prefix tree explosion in proportional NBS**: With high branching factor the number of unique prefixes can grow as 2^(f-1), making later batches very expensive. → Mitigation: expose `max_prefix_branches` parameter to cap growth; default matches paper's behavior.
- **cuQuantum version compatibility**: cuTensorNet API has changed across CUDA versions. → Mitigation: pin cuquantum-python to a specific release in requirements; document tested CUDA version (12.x).
- **Qiskit IR changes**: Qiskit 1.x changed the transpiler and gate representation vs. 0.x. → Mitigation: target Qiskit 1.x explicitly; use `circuit.data` gate iteration which is stable.
- **Path finding time variability**: Path finding with 100 hypersamples can take 10–100 seconds for large circuits. This is expected and negligible at HPC scale (cached), but makes unit tests slow. → Mitigation: use 1-hypersample in test mode; provide a `--fast` flag to benchmarks.

## Migration Plan

This is a greenfield project with no existing code to migrate. Deployment steps:
1. `pip install -e .[dev]` installs all dependencies.
2. Run `pytest tests/` to verify correctness on CPU-only smoke tests (small circuits, numpy backend).
3. Run `python benchmarks/run_all.py --gpu` for GPU throughput benchmarks.
4. Compare speedup table against Table I from the paper.

Rollback: N/A (no production system; experimental research code).

## Open Questions

- **Threshold for exhaustive sampling**: The paper does not specify the exact probability threshold for exhaustive final-batch sampling. Proposed default: 1/2^(bf+4) (i.e., return entries with probability > ~6% of uniform). Validate empirically.
- **Hypersample count for proportional proportional PTSBE**: Paper uses 100 hypersamples; confirm this is applied per error set or per circuit.
- **Multi-GPU path for Phase 3**: The paper notes embarrassing parallelism across E GPUs (one error set per GPU). The initial implementation will be single-GPU; multi-GPU is a future extension.
