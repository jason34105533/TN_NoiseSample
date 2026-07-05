## Context

The archived implementation (`2026-07-05-high-throughput-quantum-tn-simulator`) built Phase 1–3 simulators from a paper (Patti et al., arXiv:2604.08467) but was only ever validated on CPU with a toy circuit (n=6, g=20), per `benchmarks/validation_notes.md`. That doc explicitly says GPU validation requires an NVIDIA GPU + CUDA 12.x + `cuquantum-python-cu12` + `cupy-cuda12x` — all of which are now available on this machine (4× Tesla V100-32GB, driver 580.159.03, CUDA toolkit 12.0, cupy 13.4.1, cuquantum-python-cu12 25.9.1 already importable at the system level).

The original design (D2) targeted Qiskit 1.x specifically because "Qiskit 1.x changed the transpiler and gate representation vs 0.x." The system now has Qiskit 2.2.2 and NumPy 2.2.6 installed — versions the original design never considered. No project-specific Python environment exists yet; `pytest` isn't installed anywhere and the package has never been `pip install -e`'d.

## Goals / Non-Goals

**Goals:**
- Stand up a working, isolated environment for this project and get the existing test suite passing against real installed dependency versions.
- Get real single-GPU throughput/speedup numbers on a V100 at increasing circuit scale, replacing the CPU toy benchmark.
- Fix only what real execution proves broken — no speculative rewrites.

**Non-Goals:**
- Multi-GPU or multi-error-set parallelism across the 4 available V100s (matches the original design's own stated non-goal; this is future work for an H100 machine).
- Algorithmic changes to UPV or NBS.
- Downgrading Qiskit preemptively — only considered if 2.x proves fundamentally incompatible.
- Reproducing the paper's exact H100 numbers — V100 has less compute and 32GB vs 80GB memory, so absolute throughput will differ; the goal is a real, honest measurement on this hardware, not matching the paper's absolute figures.

## Decisions

### D1: Dedicated conda environment, not `base`
**Decision**: Create a new conda environment (e.g. `tn-noise-sim`) sibling to the existing `cvenv`/`nlpenv` environments, and do all installs/testing/benchmarking there.
**Rationale**: Installing project-pinned dependencies into `base` risks colliding with whatever else lives there; matches the user's existing per-project environment convention.
**Alternative considered**: venv — rejected only because conda is already the established pattern on this machine (cvenv, nlpenv).

### D2: Keep Qiskit 2.2.2; fix forward, don't downgrade
**Decision**: Run the test suite against whatever's already installed (Qiskit 2.2.2, NumPy 2.2.6). Fix compatibility breakage in our code as it's found. Only pin back to Qiskit 1.x if a fix would require reimplementing core gate-extraction logic rather than a small API adjustment.
**Rationale**: Guessing at incompatibility upfront and downgrading blind wastes effort if 2.x mostly just works; the actual pytest errors will tell us exactly what's broken and how deep the fix needs to go.
**Alternative considered**: Pin to Qiskit 1.x immediately to match the original design's tested assumption — rejected as premature; only revisit if 2.x proves genuinely blocking.

### D3: Empirical `bf` (final_batch_size) determination, not upfront reduction
**Decision**: Start benchmark runs at the paper's default `bf=28`. Only reduce it if it OOMs on the 32GB V100, and only for the specific (n, g) configurations where it actually fails. Do not change the simulator's documented default of 28.
**Rationale**: 32GB is less than the paper's 80GB H100, but headroom depends on n, g, and workspace overhead — this is only knowable by actually running it. Matches the proposal's "validate first" framing.
**Alternative considered**: Precompute a "safe" `bf` from a memory formula upfront — rejected because it substitutes guessing for measurement, and the design's own D-series risk notes flag `bf=28` memory pressure as circuit-size-dependent, not a constant.

### D4: Scale-up ladder for GPU benchmarking
**Decision**: Run the benchmark harness at increasing scale rather than jumping straight to the paper's n=200/g=1000: start with something like n=20–30/g=50–100 as a smoke test on real GPU execution, then step up toward n=100–200/g=600–1000 as each step succeeds.
**Rationale**: This is the first time this code has ever touched a real GPU. A smoke-scale run surfaces execution bugs (memory layout, cuTensorNet API mismatches, dtype issues) cheaply, before burning time on a large run that might fail for a trivial reason.
**Alternative considered**: Go straight to paper-scale — rejected as needlessly risky for a first real GPU execution.

## Risks / Trade-offs

- **Qiskit 2.x API breakage may be deep, not shallow** → Mitigation: D2's fix-forward approach with an explicit fallback to pinning 1.x if the first attempt shows the breakage is structural (e.g. gate iteration semantics fundamentally changed, not just renamed).
- **V100 lacks H100's memory and raw throughput; absolute numbers won't match the paper** → Mitigation: goal is documented, honest V100 numbers, not paper parity; `validation_notes.md` will state hardware clearly.
- **cuQuantum/cuTensorNet API version drift** (25.9.1 installed, unknown version at original design time) could break `contraction.py`'s calls → Mitigation: this is exactly the kind of thing the smoke-scale run (D4) is meant to catch early.
- **First real GPU execution may surface correctness bugs, not just compatibility bugs** (e.g. Phase 3's sampled distribution silently diverging from Phase 1's under real contraction) → Mitigation: where feasible, cross-check that Phase 1 (traditional) and Phase 3 (optimized) produce statistically consistent bitstring distributions on the same small circuit, not just that both run without crashing.

## Migration Plan

1. Create dedicated conda environment; `pip install -e ".[dev,gpu]"`.
2. Run `pytest tests/`; triage and fix any Qiskit 2.x / NumPy 2.x compatibility failures.
3. Run `benchmarks/run_all.py` at smoke scale on 1 GPU; confirm all three simulator phases execute end-to-end on real cuTensorNet.
4. Scale up toward n=100–200 / g=600–1000, backing off `bf` only where memory forces it.
5. Update `benchmarks/validation_notes.md` with real V100 numbers and an honest comparison to the paper's claims.

Rollback: N/A — experimental research code, no production system affected.

## Open Questions

- How far can we actually push n/g on a single 32GB V100 before OOM, and at what `bf`? (Answer comes from D4's scale-up ladder, not predicted here.)
- Does Qiskit 2.x break `circuit.data` iteration in `tensor_network.py`, or does it still work as-is? (Answer comes from the first pytest run.)
