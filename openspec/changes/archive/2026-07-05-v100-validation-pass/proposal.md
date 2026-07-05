## Why

The Phase 1–3 simulators (traditional, unoptimized PTSBE, optimized PTSBE) were implemented greenfield and archived as complete, but have never been executed against real GPU hardware or against the dependency versions actually installed on this machine. `validation_notes.md` only contains a CPU toy benchmark (n=6, g=20, ~18× speedup) against the paper's claimed up to 10⁸× (non-proportional) / 10³× (proportional) speedups. This machine has 4× Tesla V100-32GB GPUs with a working CuPy/cuQuantum stack sitting idle. Before any hardware-specific tuning is worth doing, we need real numbers: does the existing implementation even run correctly against installed Qiskit 2.2.2 / NumPy 2.2.6, and what speedup does it actually achieve on one V100 at increasing circuit scale?

## What Changes

- Create a dedicated conda environment for this project (not `base`) and install the package editable with `dev` + `gpu` extras.
- Run the existing `pytest` suite against the real installed dependency versions (Qiskit 2.2.2, NumPy 2.2.6 — both newer than the original design's Qiskit 1.x assumption) and fix forward any compatibility breakage found (e.g. `circuit.data` iteration or transpiler API surface, per the original design's own flagged risk). No preemptive downgrade to Qiskit 1.x — only reconsider pinning if 2.x proves fundamentally incompatible.
- Run the existing benchmarking harness on a single V100, starting small and scaling up toward the paper's regime (n=100–200, g=600–1000), passing a reduced `final_batch_size` (`bf`) only if 32GB memory proves insufficient at the paper's `bf=28` — determined empirically during the run, not decided in advance.
- Update `benchmarks/validation_notes.md` with real single-GPU V100 throughput and speedup numbers, replacing the CPU toy `n=6` result, and compare against the paper's Table I / Figs. 4–7 claims.
- Fix any correctness or execution bugs surfaced by actually running Phase 1–3 on real GPU hardware for the first time.

**Explicitly out of scope**: multi-GPU / multi-error-set parallelism across the 4 available V100s (matches the original design's own non-goal; revisit when moving to an H100 machine), any new algorithmic capability, CLI/GUI work.

## Capabilities

### New Capabilities

None. This is a validation and hardening pass over existing capabilities — no new user-facing capability is introduced.

- `benchmarking-harness`: results output SHALL also record hardware context (GPU device name, total device memory) alongside existing throughput/speedup/config fields, so results from different GPU models (this V100 pass vs. a future H100 pass) are distinguishable without cross-referencing external notes.

Beyond this, no other requirement changes are expected. The goal is to make the existing `traditional-trajectory-simulator`, `unoptimized-ptsbe-simulator`, `optimized-ptsbe-simulator`, `tensor-network-builder`, and `contraction-engine` capabilities work correctly against real GPU hardware and currently-installed dependency versions, without changing their documented requirements or defaults (e.g. `final_batch_size` default remains 28; a smaller value would only ever be passed explicitly at benchmark invocation time, not become a new default). If a compatibility fix is found during implementation to require an actual behavior/contract change, it will be called out and a delta spec added at that point rather than guessed upfront.

## Impact

- **Environment**: new dedicated conda environment for this project (sibling to existing `cvenv`/`nlpenv`); package installed editable (`pip install -e ".[dev,gpu]"`).
- **Dependencies**: no version changes planned — validates against what's already installed (qiskit 2.2.2, numpy 2.2.6, cupy-cuda12x 13.4.1, cuquantum-python-cu12 25.9.1). Qiskit may need a pin decision if 2.x proves incompatible.
- **Code**: likely small compatibility fixes in `tensor_network.py` / wherever Qiskit circuit iteration happens, if the test run surfaces breakage. No algorithmic changes to UPV/NBS logic expected.
- **Docs**: `benchmarks/validation_notes.md` updated with real GPU numbers.
- **Hardware**: single Tesla V100-32GB for this pass; multi-GPU and H100 deferred.
