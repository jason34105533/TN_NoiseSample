# Contraction Engine

## Purpose

Provides GPU-accelerated tensor network contraction via `cuquantum.tensornet.experimental.NetworkState`, including path finding, bounded-memory batched marginal contraction, conditional marginal contraction, contraction/network caching for UPV reuse, and flexible per-batch size control.

## Requirements

### Requirement: Contraction path finding via cuTensorNet
The `ContractionEngine` SHALL use `cuquantum.tensornet.experimental.NetworkState` to represent and contract tensor networks on GPU. Path finding SHALL be a real cuTensorNet operation (not a CPU `opt_einsum` fallback silently substituted for GPU execution) and SHALL be controlled via a `num_hypersamples` parameter forwarded to `NetworkState`'s `TNConfig(num_hyper_samples=...)`. Because `NetworkState` does not expose an explicit separate path-finding call, path-finding cost SHALL be treated as the cost incurred by the first bounded-marginal query for a given `(batch qubits, fixed-qubit-set)` pattern; subsequent queries with the same pattern (different fixed values or different error-set tensor values) SHALL reuse cuTensorNet's internally cached contraction plan.

#### Scenario: Path finding with specified hypersamples
- **WHEN** a `ContractionEngine` with `use_gpu=True` is constructed with `num_hypersamples=100` and used to contract a network
- **THEN** the underlying `NetworkState` is configured with `TNConfig(num_hyper_samples=100)`

#### Scenario: First query pays path-finding cost, later queries do not
- **WHEN** `contract_batch()` is called twice for the same batch index and the same set of fixed (conditioned) qubits, with different `prefix` values or different error-set tensor values
- **THEN** the second call's wall-clock time is substantially lower than the first, reflecting reuse of the cached contraction plan rather than rediscovery

### Requirement: Batched marginal contraction
The `ContractionEngine` SHALL support batched contraction where a subset of b qubits is contracted at each step to produce a 2^b marginal probability vector, with **peak GPU memory bounded by `O(2^b)`, not `O(2^n)`**, regardless of the total qubit count n. The qubit batching order SHALL follow the sequential scan order used in PTSBE (qubit 0..b-1 in batch B_1, qubits b..2b-1 in B_2, etc.). Contractions SHALL execute on GPU via `NetworkState.compute_reduced_density_matrix(where=batch_qubits, diagonal=True)`, whose output tensor is bounded in size by the requested `where` modes.

#### Scenario: Marginal vector has correct length
- **WHEN** `contract_batch(network, path, batch_size=b)` is called for batch B_j
- **THEN** the returned array has shape (2**b,) and sums to 1.0 (within floating-point tolerance) when unconditioned

#### Scenario: GPU execution for contraction
- **WHEN** `contract_batch()` is called on a GPU-resident `NetworkState`
- **THEN** all contraction work executes on the GPU via `compute_reduced_density_matrix`, and peak GPU memory for that call does not scale with `2^n`

#### Scenario: Bounded memory at circuit scale beyond dense-statevector feasibility
- **WHEN** `contract_batch()` is called with `batch_size=28` on a network with n=200 qubits
- **THEN** the call completes without attempting to materialize a `2^200`-entry array at any point

### Requirement: Conditional marginal contraction
For proportional NBS, the `ContractionEngine` SHALL support conditioning on a bitstring prefix (s_1,...,s_{j-1}) when contracting batch B_j. Conditioning SHALL be implemented by translating the prefix into a `fixed={qubit_index: bit_value}` dict passed to `NetworkState.compute_reduced_density_matrix(where=batch_qubits, fixed=fixed_dict, diagonal=True)`, which projects the fixed qubits to their observed values without materializing any un-fixed, non-batch qubit dimension.

#### Scenario: Conditioning on prefix selects a slice
- **WHEN** `contract_batch(network, path, batch_index=j, prefix=(0,1))` is called
- **THEN** qubits 0 and 1 are fixed to values 0 and 1 respectively via the `fixed` argument, and the returned marginal is the (unnormalized) conditional distribution over the next b qubits given that prefix

#### Scenario: Unconditioned first batch
- **WHEN** `contract_batch(network, path, batch_index=1, prefix=())` is called
- **THEN** `fixed={}` is passed and the full 2^b marginal over qubits 0..b-1 is returned

### Requirement: Contraction path cache
For Unified Path Variations (UPV), the `ContractionEngine` SHALL support a persistent-`NetworkState` mode in which one `NetworkState` is constructed for the noiseless circuit topology (via `apply_tensor_operator` per gate, in circuit order), and reused across all E pre-sampled error sets by calling `NetworkState.update_tensor_operator(tensor_id, fused_operand)` for only the gates each error set's fused operators touch — leaving the network's cuTensorNet-internal structure and cached contraction plan untouched. The `ContractionPathCache` SHALL key on circuit topology (as today) but, for the GPU backend, SHALL cache a handle to the persistent `NetworkState` plus its per-gate `tensor_id` mapping rather than an opaque path object.

#### Scenario: Single NetworkState reused across error sets
- **WHEN** E error sets are pre-sampled for the same circuit and processed via `OptimizedPTSBESimulator`
- **THEN** exactly one `NetworkState` is constructed for that circuit, and each error set is applied to it via `update_tensor_operator` calls rather than constructing a new `NetworkState`

#### Scenario: Cache hit avoids recomputation
- **WHEN** contracting a second circuit instance whose topology hash matches a previously cached one
- **THEN** the cached `NetworkState`/`tensor_id` mapping is reused instead of rebuilding the network from scratch

#### Scenario: Cache miss triggers network construction
- **WHEN** contracting a network whose topology hash is not in the cache
- **THEN** a new `NetworkState` is constructed via `apply_tensor_operator` calls, its `tensor_id` mapping is stored in the cache, and then used

### Requirement: Flexible batch size interface
The `ContractionEngine` SHALL accept separate `batch_size` and `final_batch_size` parameters. Non-final batches SHALL use `batch_size` and the final batch SHALL use `final_batch_size`. Both parameters SHALL be forwarded to the batched contraction routine independently.

#### Scenario: Non-final batches use batch_size
- **WHEN** contracting batches B_1 through B_{f-1}
- **THEN** each contraction covers exactly `batch_size` qubits

#### Scenario: Final batch uses final_batch_size
- **WHEN** contracting batch B_f
- **THEN** the contraction covers exactly `final_batch_size` qubits and returns a 2^{final_batch_size} probability vector
