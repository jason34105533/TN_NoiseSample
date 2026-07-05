## ADDED Requirements

### Requirement: Contraction path finding via cuTensorNet
The `ContractionEngine` SHALL use `cuquantum.cutensornet` to find contraction paths. It SHALL accept a `num_hypersamples` parameter controlling optimizer iterations. The path SHALL be returned as a cuTensorNet `ContractionOptimizerInfo` object and stored by the `ContractionPathCache`.

#### Scenario: Path finding with specified hypersamples
- **WHEN** `ContractionEngine.find_path(network, num_hypersamples=100)` is called
- **THEN** cuTensorNet's path optimizer runs 100 iterations and returns the best path found

#### Scenario: Path object is cache-storable
- **WHEN** a path is found
- **THEN** it can be serialized to / deserialized from the `ContractionPathCache` without loss of contraction order information

### Requirement: Batched marginal contraction
The `ContractionEngine` SHALL support batched contraction where a subset of b qubits is contracted at each step to produce a 2^b marginal probability vector. The qubit batching order SHALL follow the sequential scan order used in PTSBE (qubit 0..b-1 in batch B_1, qubits b..2b-1 in B_2, etc.). Contractions SHALL execute on GPU using CuPy arrays.

#### Scenario: Marginal vector has correct length
- **WHEN** `contract_batch(network, path, batch_size=b)` is called for batch B_j
- **THEN** the returned CuPy array has shape (2**b,) and sums to 1.0 (within floating-point tolerance)

#### Scenario: GPU execution for contraction
- **WHEN** `contract_batch()` is called on a GPU-resident network
- **THEN** all einsum contractions execute on the GPU and the result array remains on GPU memory

### Requirement: Conditional marginal contraction
For proportional NBS, the `ContractionEngine` SHALL support conditioning on a bitstring prefix (s_1,...,s_{j-1}) when contracting batch B_j. Conditioning SHALL be implemented by projecting the measured qubit indices to the observed values before contracting the remaining open indices.

#### Scenario: Conditioning on prefix selects a slice
- **WHEN** `contract_batch(network, path, batch_index=j, prefix=(0,1))` is called
- **THEN** qubits 0 and 1 are fixed to values 0 and 1 respectively, and the returned marginal is the conditional distribution over the next b qubits given that prefix

#### Scenario: Unconditioned first batch
- **WHEN** `contract_batch(network, path, batch_index=1, prefix=())` is called
- **THEN** no projection is applied and the full 2^b marginal over qubits 0..b-1 is returned

### Requirement: Contraction path cache
The `ContractionPathCache` SHALL store paths keyed by a topology hash derived from tensor shapes and index labels of the noiseless network. It SHALL provide `get(key)` and `put(key, path)` methods. Cache lookups SHALL be O(1). The cache SHALL persist for the lifetime of the Python process.

#### Scenario: Cache hit avoids recomputation
- **WHEN** `find_path()` is called for a network whose topology hash already exists in the cache
- **THEN** the cached path is returned immediately without invoking the cuTensorNet optimizer

#### Scenario: Cache miss triggers path finding
- **WHEN** `find_path()` is called for a network whose topology hash is not in the cache
- **THEN** the optimizer runs, the result is stored in the cache, and then returned

### Requirement: Flexible batch size interface
The `ContractionEngine` SHALL accept separate `batch_size` and `final_batch_size` parameters. Non-final batches SHALL use `batch_size` and the final batch SHALL use `final_batch_size`. Both parameters SHALL be forwarded to the batched contraction routine independently.

#### Scenario: Non-final batches use batch_size
- **WHEN** contracting batches B_1 through B_{f-1}
- **THEN** each contraction covers exactly `batch_size` qubits

#### Scenario: Final batch uses final_batch_size
- **WHEN** contracting batch B_f
- **THEN** the contraction covers exactly `final_batch_size` qubits and returns a 2^{final_batch_size} probability vector
