## ADDED Requirements

### Requirement: Unified Path Variations (UPV)
The `OptimizedPTSBESimulator` SHALL compute exactly one contraction path on the error-free (noiseless) tensor network and reuse it for all E pre-sampled error sets. Before path finding, each error operator k_p^i SHALL be fused (matrix-multiplied) into its adjacent coherent gate tensor d_l, producing a fused tensor with identical leg structure and dimensions. The path SHALL be stored in the `ContractionPathCache` keyed by the noiseless network topology.

#### Scenario: Single path-find for all error sets
- **WHEN** E error sets are pre-sampled and m = sum(m_i) total shots requested
- **THEN** the contraction path optimizer is invoked exactly once, not E times

#### Scenario: Error fusion preserves topology
- **WHEN** error operator k of shape (d, d) is fused into gate tensor d_l of shape (d_in, d_out)
- **THEN** the fused tensor has the same index legs as d_l (topology unchanged) so the cached path applies directly

#### Scenario: Cached path reused across error sets
- **WHEN** processing error set K_2 after K_1 with the same circuit
- **THEN** path finding is skipped and the cached path P^j is retrieved from `ContractionPathCache`

### Requirement: Non-Degenerate Batched Sampling — proportional mode
In proportional mode, the simulator SHALL maintain a set of unique bitstring prefixes (s_1,...,s_{j-1}) after each batch B_j. For each unique prefix, it SHALL contract exactly one marginal distribution over B_j conditioned on that prefix and sample count(s_1,...,s_{j-1}) continuations. The first batch B_1 SHALL require only one contraction (no conditioning). The final set of complete bitstrings (s_1,...,s_f) SHALL faithfully represent the original quantum probability distribution.

#### Scenario: First batch requires one contraction
- **WHEN** processing batch B_1 in proportional mode
- **THEN** exactly one contraction is performed (no conditioning on prior batches)

#### Scenario: Subsequent batches deduplicate by prefix
- **WHEN** j > 1 and K unique prefixes (s_1,...,s_{j-1}) exist from prior batches
- **THEN** exactly K contractions are performed for batch B_j (one per unique prefix)

#### Scenario: Output distribution matches quantum statistics
- **WHEN** proportional PTSBE is run with large m_i
- **THEN** the empirical bitstring frequencies converge to the Born rule probabilities of the circuit under the given noise model

### Requirement: Non-Degenerate Batched Sampling — non-proportional mode
In non-proportional mode, the simulator SHALL sample one or more bitstring prefixes from each non-final batch B_j (j < f). For the final batch B_f, the simulator SHALL contract a full 2^{b_f} probability vector for each unique prefix and return all entries whose probability exceeds a configurable threshold `min_prob` as distinct labeled bitstrings. This exhaustive final-batch extraction SHALL be the primary source of high-throughput data collection.

#### Scenario: Non-final batches branch by user-specified rate
- **WHEN** non-proportional mode is used with `non_final_samples_per_branch=k`
- **THEN** each unique prefix spawns k new sub-prefixes per non-final batch

#### Scenario: Final batch yields 2^{b_f} candidate bitstrings
- **WHEN** the final batch B_f is processed for a unique prefix
- **THEN** a probability vector of length 2^{b_f} is computed and all entries with p >= min_prob are returned as complete bitstrings

#### Scenario: No degenerate contractions
- **WHEN** two prefixes (s_1,...,s_{f-1}) are identical
- **THEN** the final-batch contraction is performed once and its results shared between both prefixes (no duplicate contractions)

### Requirement: Flexible per-batch contraction sizes
The simulator SHALL accept separate batch size parameters for non-final batches (`batch_size`) and the final batch (`final_batch_size`). The defaults SHALL be `batch_size=10` and `final_batch_size=28` per the paper's optimized configuration. Both SHALL be validated to be ≤ n (circuit qubit count).

#### Scenario: Default batch sizes match paper
- **WHEN** `OptimizedPTSBESimulator` is instantiated with no batch size arguments
- **THEN** non-final batches use 10 qubits and the final batch uses 28 qubits

#### Scenario: Custom batch sizes respected
- **WHEN** `OptimizedPTSBESimulator(batch_size=8, final_batch_size=24)` is used
- **THEN** non-final contractions process 8 qubits per batch and the final contraction processes 24

#### Scenario: Batch size validation
- **WHEN** `final_batch_size` is set larger than the number of circuit qubits n
- **THEN** a `ValueError` is raised at instantiation time

### Requirement: Hypersample count for path optimization
The simulator SHALL accept a `num_hypersamples` parameter that controls the number of path optimizer iterations. The default SHALL be 100. Path finding with 100 hypersamples SHALL be performed once and its result cached; all subsequent evaluations for the same circuit SHALL skip path finding.

#### Scenario: Default hypersample count is 100
- **WHEN** `OptimizedPTSBESimulator` is instantiated without specifying `num_hypersamples`
- **THEN** the path optimizer runs 100 iterations during the single path-finding call

#### Scenario: Path finding cost is amortized
- **WHEN** the same circuit is sampled multiple times in the same process
- **THEN** path finding runs exactly once; subsequent calls retrieve the path from the cache in O(1)
