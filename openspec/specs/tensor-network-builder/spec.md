# Tensor Network Builder

## Purpose

Converts a Qiskit QuantumCircuit into an uncontracted tensor network compatible with cuTensorNet, supporting both error operator insertion (unoptimized mode) and error operator fusion (UPV mode), as well as correct tensorization of single- and two-qubit gates.

## Requirements

### Requirement: Tensor network construction from Qiskit circuit
The `TensorNetworkBuilder` SHALL accept a Qiskit `QuantumCircuit` and return an uncontracted tensor network as a list of tensors with named indices, compatible with cuTensorNet's `NetworkOperand` format. Each gate in the circuit SHALL become one tensor node; initial qubit state tensors (|0⟩) SHALL be prepended. Measurement tensors SHALL be appended.

#### Scenario: Gate count matches circuit
- **WHEN** `TensorNetworkBuilder.build(circuit)` is called on a circuit with g gates
- **THEN** the returned tensor list contains g gate tensors plus n initial-state tensors

#### Scenario: Index contraction structure is correct
- **WHEN** gate A's output qubit leg connects to gate B's input qubit leg
- **THEN** those two tensors share the same index label, expressing the contraction between them

### Requirement: Error operator insertion (unoptimized mode)
In unoptimized mode, the builder SHALL insert error operator tensors as separate nodes adjacent to their target gate tensors. Each error operator k_p^i for gate p in error set K_i SHALL be added as an extra tensor node connected to the gate's output leg, changing the tensor network topology relative to the error-free circuit.

#### Scenario: Error tensor added per noisy gate
- **WHEN** `build(circuit, error_set=K_i, mode="insert")` is called
- **THEN** each gate p ∈ K_i gains one additional adjacent tensor node for its error operator

#### Scenario: Noiseless and noisy networks differ in topology
- **WHEN** comparing `build(circuit)` vs `build(circuit, error_set=K_i, mode="insert")`
- **THEN** the noisy network has more tensor nodes (one per error operator in K_i), producing a different topology

### Requirement: Error operator fusion (UPV mode)
In UPV mode, the builder SHALL fuse each error operator k_p^i into its adjacent coherent gate tensor d_l by matrix multiplication, producing a fused tensor with the same index structure as d_l. The tensor network topology SHALL be identical to the error-free network after fusion.

#### Scenario: Fused tensor has same legs as original gate
- **WHEN** `build(circuit, error_set=K_i, mode="fuse")` is called
- **THEN** every gate tensor p ∈ K_i is replaced by the product (d_l @ k_p^i), retaining d_l's index labels and dimensions

#### Scenario: Fused topology matches noiseless topology
- **WHEN** comparing `build(circuit)` vs `build(circuit, error_set=K_i, mode="fuse")`
- **THEN** both networks have the same number of tensor nodes and the same index connectivity graph

### Requirement: Multi-qubit gate support
The builder SHALL correctly tensorize single-qubit gates (2×2 unitaries reshaped to (2,2)) and two-qubit gates (4×4 unitaries reshaped to (2,2,2,2)), assigning one index per qubit leg (input and output).

#### Scenario: Single-qubit gate produces rank-2 tensor
- **WHEN** a single-qubit gate G is processed
- **THEN** the resulting tensor has shape (2, 2) with one input and one output index

#### Scenario: Two-qubit gate produces rank-4 tensor
- **WHEN** a two-qubit gate G is processed
- **THEN** the resulting tensor has shape (2, 2, 2, 2) with two input and two output indices
