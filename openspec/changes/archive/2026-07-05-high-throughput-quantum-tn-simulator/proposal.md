## Why

Simulating noisy quantum circuits at scale is essential for quantum error correction research and AI-assisted quantum device design, yet existing tools (e.g., CUDA-Q) rely on traditional trajectory methods that recompute contraction paths per shot and extract only one measurement per full contraction loop — leaving 8+ orders of magnitude of throughput on the table. This project implements the full progression of tensor network trajectory simulators described in Patti et al. (2025), from the traditional baseline through the Optimized PTSBE framework, to reproduce and open-source the claimed speedups (up to 10⁸× for non-proportional and 10³× for proportional sampling).

## What Changes

- Introduce a **Phase 1 Traditional Trajectory Simulator**: per-shot CPU contraction path finding, sequential single-shot extraction, fixed batch sizes — serves as the benchmark baseline.
- Introduce a **Phase 2 Unoptimized TN PTSBE Simulator**: pre-samples E error sets upfront, finds contraction paths once per error set (not per shot), but still uses sequential single-shot extraction and rigid batch sizes.
- Introduce a **Phase 3 Optimized TN PTSBE Simulator** implementing:
  - **Unified Path Variations (UPV)**: fuses error operator tensors into adjacent coherent gate tensors so the error-free contraction path is computed once and reused for all E error sets.
  - **Non-Degenerate Batched Sampling (NBS)**: batch-processes intermediate bitstring prefixes (proportional mode) and exhaustively harvests all final-batch bitstrings above a threshold (non-proportional mode), eliminating redundant partial contractions.
  - **Flexible Contraction Interface**: exposes per-batch qubit batch sizes `bj` and final batch size `bf` as tunable hyperparameters, enabling contraction-rate optimization.
- Introduce a **benchmarking harness** that measures data-collection throughput (unique labeled bitstrings per GPU-second) for all three phases and reports speedup ratios.

## Capabilities

### New Capabilities

- `traditional-trajectory-simulator`: Phase 1 baseline — per-shot path finding, sequential shot extraction, fixed batch size.
- `unoptimized-ptsbe-simulator`: Phase 2 — pre-sampled error sets, per-error-set path finding, sequential single-shot extraction, rigid batch sizes.
- `optimized-ptsbe-simulator`: Phase 3 core — UPV path reuse, NBS proportional and non-proportional batched sampling, flexible batch size interface.
- `error-sampling`: Pre-trajectory sampling of E error operator sets according to proportional or non-proportional rules; assigns per-error-set shot counts `mᵢ`.
- `tensor-network-builder`: Constructs tensor networks from Qiskit intermediate representations; handles coherent gate tensors and error operator insertion/fusion for UPV.
- `contraction-engine`: Wraps cuQuantum cuTensorNet for multi-GPU contraction; exposes configurable batch sizes and hypersample counts for path optimization.
- `benchmarking-harness`: Measures throughput and data-collection speedup across all three simulator phases on configurable circuit ensembles.

### Modified Capabilities

## Impact

- **New dependencies**: `cuquantum-python` (cuTensorNet), `cupy`, `qiskit`, `numpy`; NVIDIA GPU with CUDA required for GPU contraction paths.
- **New source tree**: All simulator code lives under `src/` with one module per capability; benchmarks live under `benchmarks/`.
- **No existing code modified**: this is a greenfield implementation.
