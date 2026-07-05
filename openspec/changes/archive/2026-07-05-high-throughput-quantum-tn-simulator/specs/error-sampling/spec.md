## ADDED Requirements

### Requirement: Proportional error set sampling
The `ErrorSampler` SHALL support proportional sampling, where each error set K_i is drawn with probability proportional to the product of per-gate error probabilities under the noise model. The shot count m_i assigned to K_i SHALL be proportional to that same probability, such that the aggregate shot distribution matches the Born rule for the noisy circuit.

#### Scenario: Error sets drawn by probability weight
- **WHEN** `ErrorSampler.sample(noise_model, num_gates, num_error_sets=E, mode="proportional")` is called
- **THEN** each K_i is drawn by weighted sampling from the error operator product distribution

#### Scenario: Shot counts proportional to weights
- **WHEN** proportional mode is used with total_shots=m and E error sets
- **THEN** m_i = round(m * p_i / sum(p)) where p_i is the sampling weight of K_i

### Requirement: Non-proportional (uniform) error set sampling
The `ErrorSampler` SHALL support non-proportional (uniform) sampling, where error sets are drawn uniformly at random from all possible error operator combinations, and each error set is assigned an equal shot count m_i = ceil(total_shots / E).

#### Scenario: Error sets drawn uniformly
- **WHEN** `mode="non_proportional"` is specified
- **THEN** each K_i is independently sampled with uniform probability across the noise model's error operator space

#### Scenario: Shot counts equal across error sets
- **WHEN** non-proportional mode with total_shots=m and E error sets
- **THEN** m_i = ceil(m / E) for all i, with at most 1 extra shot distributed across sets to account for rounding

### Requirement: Noise model compatibility
The `ErrorSampler` SHALL accept a noise model object specifying per-gate error channels. For single-qubit gates it SHALL support Pauli error channels (X, Y, Z). For two-qubit gates it SHALL support depolarizing channels. Error probabilities SHALL be per-gate scalars in [0, 1].

#### Scenario: Pauli error operator sampled for single-qubit gate
- **WHEN** a single-qubit gate has Pauli error probabilities (p_X, p_Y, p_Z)
- **THEN** the sampled error operator for that gate is one of {I, X, Y, Z} drawn with probabilities {1-p_X-p_Y-p_Z, p_X, p_Y, p_Z}

#### Scenario: Depolarizing error for two-qubit gate
- **WHEN** a two-qubit gate has depolarizing error probability p
- **THEN** the sampled error operator is one of the 15 non-identity two-qubit Pauli operators, each with probability p/15, or identity with probability 1-p

### Requirement: Error set serialization
Each sampled error set K_i SHALL be representable as a dict mapping gate index to a sampled error operator matrix (numpy/cupy array). The `ErrorSampler` SHALL expose a `to_list()` method that returns all E error sets as a list of such dicts.

#### Scenario: Error set keyed by gate index
- **WHEN** `ErrorSampler.to_list()` is called after sampling
- **THEN** each element is a dict where keys are integer gate indices and values are 2D numpy arrays of the sampled Kraus operator
