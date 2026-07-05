# Validation Notes — Deviations from Paper

## V100 Validation Pass (2026-07-05)

This section supersedes the "Scaled-Down CPU Validation" section below, which
predates this repo having access to real GPU hardware.

### Environment

- Hardware: 4x Tesla V100-PCIE-32GB (this pass used a single GPU by design —
  see `openspec/changes/v100-validation-pass/`), driver 580.159.03, CUDA
  toolkit 12.0.
- Dedicated conda env `tn-noise-sim` (Python 3.11), package installed via
  `pip install -e ".[dev,gpu]"`.
- Installed versions: `qiskit==2.5.0`, `numpy==2.4.6`, `cupy-cuda12x==14.1.1`,
  `cuquantum-python-cu12==26.6.0` — all newer than what the original design
  assumed (Qiskit 1.x). All 48 pre-existing tests plus 4 new tests directly
  exercising the real Qiskit `QuantumCircuit` integration path
  (`TensorNetworkBuilder.from_qiskit_circuit`, previously uncovered by any
  test or by the benchmark circuit generator) pass unmodified. No Qiskit
  2.x/NumPy 2.x compatibility issues found.

### Finding: no GPU contraction is implemented — this is not a quick fix

Before running anything on the V100, we checked what `use_gpu=True` actually
does in this codebase. It doesn't dispatch computation to the GPU:

- `ContractionEngine._find_path_cutensornet()` (`src/tn_noise_sim/contraction.py`)
  is a stub that immediately falls back to CPU `opt_einsum` path-finding.
- `_compute_marginal()`, the function that does the actual contraction/sampling
  work, always runs on host numpy arrays via `opt_einsum`/`numpy.einsum`.
  `cupy` and `cuquantum` are imported only to drive `benchmarks/timing.py`'s
  CUDA-event-based `Timer` — they never touch the tensor math.
- All three simulator classes (`traditional.py`, `unoptimized_ptsbe.py`,
  `optimized_ptsbe.py`) hard-code `ContractionEngine(..., use_gpu=False)`,
  so even the constructor-level GPU flag never reaches the engine.

More importantly, `_compute_marginal()`'s own docstring says it's a "CPU
reference implementation" that "contracts the full amplitude tensor then
slices and marginalises" — i.e. it materializes the entire `2^n`-entry
statevector before slicing out the requested batch marginal. Per the paper
(arXiv:2604.08467, Fig. 1), contraction should be GPU-based and bounded to
`2^b` (the batch size) for *all three* simulator phases, not just the
optimized one — this is what makes the paper's n=100-200 regime tractable at
all. A dense `2^n` statevector has no physical realization at that scale on
any hardware (2^100 alone dwarfs all storage on Earth), so this isn't a
missing GPU optimization to bolt onto working CPU code — it's unbuilt core
functionality. The installed `cuquantum-python-cu12` (26.6.0) does provide the
right primitives for this
(`cuquantum.tensornet.experimental.NetworkState`, which natively accepts
Qiskit circuits and exposes `compute_sampling`/`compute_amplitude`/
`compute_reduced_density_matrix`), but wiring PTSBE's UPV/NBS batch logic on
top of it is a substantial, separate implementation effort with its own
design decisions — out of scope for this validation pass. **Recommendation:**
scope a follow-up change specifically for implementing real bounded-memory
GPU contraction before attempting any paper-regime benchmark.

### Finding: a real correctness bug in proportional-mode shot allocation

While setting up a Phase 1 vs. Phase 3 distributional cross-check (are
"proportional" PTSBE's bitstring statistics really the same as traditional
trajectories, as the paper requires?), we found `ErrorSampler._sample_proportional()`
(`src/tn_noise_sim/error_sampling.py`) drew error sets i.i.d. from their true
probability via `_draw_one_error_set()`, then re-weighted shot counts by
`_error_set_weight()` on top of that — double-applying the probability
weighting (a self-normalized-importance-sampling bug).

On a 5-qubit, 12-gate circuit with realistic (nonzero) per-gate error
probabilities, at 20,000 shots each:

| | Total variation distance vs. Traditional |
|---|---|
| Before fix | 0.21 |
| After fix | 0.036 |
| Expected from sampling noise alone | ~0.02–0.04 |

Fixed by allocating shots uniformly across the i.i.d.-drawn error sets
(matching `_sample_non_proportional`'s existing, correct logic) rather than
re-weighting. The pre-existing `test_proportional_born_rule` test never
caught this because it uses zero noise, where the bug is dormant (every error
set's weight is identically 1 regardless of re-weighting). Added
`tests/test_simulators.py::test_proportional_matches_traditional_distribution`
as a regression test with realistic nonzero noise.

### CPU-only benchmark numbers (honest, not paper-scale)

Given the above, this pass could only meaningfully benchmark the existing
dense-statevector CPU path — real numbers, but not comparable in scale or
architecture to the paper's GPU/cuTensorNet results.

| n | g | instances | shots | mean speedup (optimized non-proportional vs. traditional) |
|---|---|-----------|-------|----|
| 20 | 50 | 3 | 50 | ~84,000x (std ~20,600x) |

This is *higher* than the paper's headline 10⁸× for non-proportional
sampling, which sounds suspicious until you consider *why*: at n=20 with
`final_batch_size=20` (a single, exhaustive final batch covering the whole
circuit), Phase 3 harvests the entire `2^20`-entry probability vector in one
contraction, while Phase 1 pays for a fresh path-find + full contraction on
*every one of its 50 shots*. This isn't a controlled, paper-matched
comparison (different `n`, `g`, `bf` regime, no GPU, no lightcone effects) —
it's a sanity check that all three phases execute correctly end-to-end at a
larger scale than the previous `n=6` toy, which they do.

We also confirmed where the current dense-statevector approach starts to hurt:
at n=26, g=60, a 20-shot Traditional run (which recomputes path-finding and a
full contraction per shot with no lightcone simplification) was killed after
4+ minutes with resident memory still climbing unbounded — 900MB, then 4.3GB,
then 12.9GB and rising — well beyond what the final `2^26`-entry statevector
itself would need (~1GB), implying the naive `opt_einsum` contraction path is
also generating large uncontrolled intermediate tensors along the way. This
is on a 188GB-RAM machine; it would still have run out. Scaling further
toward the paper's n=100-200 is not a matter of "give it more time or a GPU"
— a dense `2^100`-entry array cannot be materialized on any existing
computer, GPU included. This confirms the architecture finding above:
reaching paper-comparable scale requires implementing genuine bounded-memory
batch contraction (e.g. via `cuquantum.tensornet.experimental.NetworkState`),
not just more compute.

### What this pass did NOT do

- **Did not** run anything on the V100 GPUs — there is no real GPU contraction
  path to run. `benchmarks/timing.py`'s GPU device metadata feature (added
  this pass) is ready for when that exists.
- **Did not** attempt to scale toward the paper's n=100-200/g=600-1000 regime
  — physically impossible with the current dense-statevector approach on any
  hardware, not merely slow.
- **Did not** implement real cuTensorNet contraction — recommended as a
  separate, explicitly-scoped follow-up change.

---

## Scaled-Down CPU Validation (n=6, g=20) — superseded, kept for history

The paper (Patti et al., arXiv:2604.08467) reports:
- **Non-proportional PTSBE**: up to 10⁸× speedup vs. traditional trajectories
- **Proportional PTSBE**: up to 10³× speedup

These numbers were obtained on NVIDIA H100 GPUs using cuQuantum cuTensorNet.

A scaled-down benchmark on CPU (macOS, no CUDA) confirmed the speedup trend
was present at toy scale:

| n | g | instances | mean speedup (opt vs. trad) |
|---|---|-----------|----------------------------|
| 6 | 20 | 5 | ~18× |

See the "V100 Validation Pass" section above for what has actually been
verified since gaining access to real GPU hardware, and for why this small
number is not representative of the paper's regime (path-finding cost ratio,
final batch size, no cuTensorNet — all still true; see above for the deeper
architectural reason).

### Reproducing Paper Numbers

Reaching paper-comparable numbers requires, in order:
1. Implement real bounded-memory GPU contraction (see "Finding: no GPU
   contraction is implemented" above) — a prerequisite, not optional polish.
2. Then run at the paper's scale:

```bash
# Proportional PTSBE benchmark (paper Fig. 5)
python -m benchmarks.run_all --n 100 --g 600 --instances 10 --output results_prop.json

# Non-proportional PTSBE benchmark (paper Fig. 4)
python -m benchmarks.run_all --n 200 --g 1000 --instances 10 --output results_nonprop.json

# Fig. 7 batch-size sweep
python -c "
from benchmarks.run_benchmark import run_benchmark
for b in [2, 5, 10, 15, 20, 24, 28]:
    run_benchmark(n=100, g=600, num_instances=3, batch_size=b, final_batch_size=b,
                  output_path='batch_sweep.json')
"
```

### Expected Speedup Sources (from paper analysis)

| Bottleneck | Resolved by | Expected contribution |
|---|---|---|
| Per-shot path finding (10–100 s each) | UPV: 1 path for all E error sets | 10³–10⁶× |
| Single-shot extraction (E × m iterations) | NBS final batch: 2^{b_f} bitstrings per contraction | up to 10² × |
| Fixed b=24 batch size (suboptimal) | Flexible interface: b=10 non-final, b=28 final | ~6× contraction efficiency |

Combined theoretical maximum: ~10⁸× for non-proportional, ~10³× for proportional.
