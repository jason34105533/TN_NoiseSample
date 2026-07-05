# Traditional Trajectory Simulator

## Purpose

Implements the traditional (baseline) quantum trajectory simulation approach: one contraction path computed per shot, shots processed sequentially one at a time, with a fixed qubit batch size across all batches. Serves as the performance baseline for speedup comparisons against PTSBE approaches.

## Requirements

### Requirement: Per-shot contraction path finding
The `TraditionalTrajectorySimulator` SHALL compute a new CPU contraction path for every individual shot. Path finding SHALL use cuQuantum's path optimizer with a fixed hypersample count of 1. The path SHALL NOT be cached or reused across shots.

#### Scenario: Each shot computes a fresh path
- **WHEN** `sample(circuit, noise_model, num_shots=m)` is called
- **THEN** exactly m separate calls to the path optimizer are made, one per shot

#### Scenario: Contraction path is not shared across shots
- **WHEN** two shots use the same circuit but different sampled error sets
- **THEN** each shot independently recomputes the contraction path

### Requirement: Sequential shot extraction
The simulator SHALL process shots one at a time. For each shot it SHALL: (1) sample a random error operator set, (2) build the tensor network with those error operators inserted, (3) find a contraction path, (4) execute f batches of contractions to obtain one bitstring. No batching of shots or bitstrings is permitted.

#### Scenario: Single bitstring per contraction loop
- **WHEN** a shot is extracted
- **THEN** exactly one bitstring of length n is returned from each full pass through the f batch contraction loop

#### Scenario: Shot loop runs m times
- **WHEN** `sample()` is called with `num_shots=m`
- **THEN** the full (path-find + contract + sample) cycle runs exactly m times sequentially

### Requirement: Fixed batch size
The simulator SHALL use a single fixed qubit batch size b for all contraction batches. The default SHALL be b=24, matching the CUDA-Q reference implementation. The batch size SHALL NOT vary per batch or be exposed as a per-batch parameter.

#### Scenario: Default batch size is 24
- **WHEN** `TraditionalTrajectorySimulator` is instantiated with no batch size argument
- **THEN** all contraction batches use b=24 qubits

#### Scenario: Custom fixed batch size is respected
- **WHEN** `TraditionalTrajectorySimulator` is instantiated with `batch_size=b`
- **THEN** all batches use that b value

### Requirement: Standard interface compliance
`TraditionalTrajectorySimulator` SHALL implement the `BaseTNSimulator` interface, accepting a `QuantumCircuit`, a noise model, and a shot count, and returning a list of (bitstring, error_set_id) tuples.

#### Scenario: Return type matches interface
- **WHEN** `sample(circuit, noise_model, num_shots=10)` completes
- **THEN** the return value is a list of exactly 10 `(str, int)` tuples where each string has length n
