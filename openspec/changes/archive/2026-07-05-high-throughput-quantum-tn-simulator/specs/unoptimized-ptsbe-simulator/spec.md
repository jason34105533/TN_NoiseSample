## ADDED Requirements

### Requirement: Pre-trajectory error set sampling
The `UnoptimizedPTSBESimulator` SHALL pre-sample all E error sets before constructing any tensor network. Each error set K_i SHALL be sampled according to a user-specified rule (proportional or uniform) and assigned a shot count m_i based on that rule. This pre-sampling step SHALL have polynomial complexity and complete before any GPU work begins.

#### Scenario: All error sets sampled upfront
- **WHEN** `sample(circuit, noise_model, total_shots=m)` is called
- **THEN** E error sets are sampled and shot counts assigned before any tensor network is constructed or any GPU kernel is launched

#### Scenario: Shot counts sum to total
- **WHEN** proportional sampling is used with total_shots=m
- **THEN** sum(m_i for all i in 1..E) equals m (within rounding)

### Requirement: Per-error-set contraction path finding
For each pre-sampled error set K_i, the simulator SHALL build the tensor network T_i^D (coherent gates + error operators inserted as separate nodes) and compute f contraction paths P_i^j on a fixed batch size b. These paths SHALL be computed once per error set and stored for reuse across all m_i shots for that error set.

#### Scenario: Paths found once per error set
- **WHEN** error set K_i requires m_i shots
- **THEN** the contraction paths P_i^j are computed exactly once for K_i and reused for all m_i shots

#### Scenario: Distinct paths per error set
- **WHEN** two error sets K_1 and K_2 differ in at least one error operator
- **THEN** two independent path-finding calls are made (one per error set), producing paths P_1^j and P_2^j that may differ

### Requirement: Sequential single-shot extraction per error set
Within each error set K_i, the simulator SHALL extract shots one at a time by executing all f contraction batches and sampling one bitstring per pass. This inner shot loop SHALL run m_i times for error set K_i.

#### Scenario: One shot per contraction loop pass
- **WHEN** extracting shots for error set K_i
- **THEN** each of the m_i shots requires a full pass through all f contraction batches, yielding one bitstring per pass

#### Scenario: Fixed batch size across all batches
- **WHEN** contracting batch B_j
- **THEN** the qubit batch size b is constant across j=1..f (rigid preset, not per-batch tunable)

### Requirement: Standard interface compliance
`UnoptimizedPTSBESimulator` SHALL implement the `BaseTNSimulator` interface and accept an optional `num_error_sets` (E) parameter. When E is not specified it SHALL default to total_shots (one error set per shot, matching traditional trajectory behavior).

#### Scenario: E error sets are used
- **WHEN** `num_error_sets=E` is provided
- **THEN** exactly E error sets are pre-sampled and shot counts m_i are distributed among them

#### Scenario: Return type matches interface
- **WHEN** `sample()` completes
- **THEN** the return value is a list of (bitstring, error_set_id) tuples, one per shot across all error sets
