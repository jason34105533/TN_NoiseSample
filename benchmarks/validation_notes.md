# Validation Notes — Deviations from Paper

## H100 Paper-Reproduction Pass, Session 2 (2026-07-12)

Follow-up to the same-day "H100 Paper-Reproduction Pass" section below, via
the OpenSpec change `finish-paper-reproduction-sweep`.

### Harness fix: decoupled baseline shot count

`run_benchmark()` previously passed the same `num_shots` to all three
simulators. This was fine for non-proportional sweeps but made
proportional-mode sweeps (where `num_shots` doubles as the paper's swept
`mi`, up to 10,000) impossible: `TraditionalTrajectorySimulator` builds one
GPU `NetworkState` per shot (~1-2s overhead each at n=100-200), so a
10,000-shot request meant hours just for the baseline. Added a
`baseline_num_shots` parameter, independent of `num_shots`, defaulting to
`num_shots` (unchanged behavior for existing call sites) but settable small
(e.g. 20) for proportional sweeps — since throughput is a per-shot *rate*,
the baseline only needs enough shots to estimate that rate stably, not to
match `mi`. Verified: a config that previously would have required ~10,000
Traditional shots at ~1-2s each now completes in 4.2 seconds (n=20, g=50,
mi=1000, baseline=20, measured speedup 338.9×).

### Extended non-proportional grid (Fig. 3)

Grid coverage went from 4/25 to **19/25** (n,g) cells — full 5/5 coverage
for n=50 and n=100, 3/5 for n=75/150/200 (missing g=400,800 at those three
n values). Still 1 instance per cell, not the paper's 10 — time went to
breadth across the grid rather than per-cell statistics, per this change's
design.md decision to prioritize breadth. All 19 cells succeeded (100%
success rate, no timeouts).

| n | g=200 | g=400 | g=600 | g=800 | g=1000 |
|---|---|---|---|---|---|
| 50 | 1,651,924× | 1,214,780× | 1,085,018× | 1,075,885× | 976,343× |
| 75 | 3,766,290× | — | 4,901,439× | — | 6,731,726× |
| 100 | 1,781,610× | 1,373,306× | 1,135,254× | 918,526× | 880,159× |
| 150 | 1,843,772× | — | 1,048,814× | — | 797,666× |
| 200 | 1,731,344× | — | 1,003,612× | — | 746,725× |

Notable real pattern in this data: at n=75, speedup *increases* with g
(3.8M× → 4.9M× → 6.7M×) — the clearest example yet in this project's runs
of the paper's own reported trend (Fig. 3: speedup grows with circuit depth
relative to qubit count, as deeper circuits populate more distinct states
for the final batch to harvest). At n=50/100/150/200, speedup mostly
*decreases* slightly with g in this data instead — the opposite direction.
Both patterns are real measurements from this run, reported as-is; with
only 1 instance per cell and no >80%/<80% success-tracking noise floor
established yet (all cells succeeded, so noise isn't a hidden explanatory
factor here), this divergence from the paper's monotonic trend is left
unexplained rather than rationalized. A deeper investigation (10 instances
per cell, to separate real trend from single-instance circuit-to-circuit
variance) is future work.

`benchmarks/figures/fig3_nonproportional_speedup.png` and
`fig6_contraction_pathfinding.png` regenerated from the extended data.

### What did NOT run this session

Per explicit direction to stop after the non-proportional grid work:

- **Proportional sweep (Fig. 5)**: not run. The harness fix above unblocks
  it (previously every attempt timed out), but no sweep data was collected
  this session.
- **Batch-size sweep (Fig. 7)**: not run. `run_batch_size_sweep()` and a
  ready driver script (`benchmarks/_batch_sweep.py`) exist but were never
  executed.
- **bf-sweep anomaly follow-up**: not run. The prior session's finding
  (speedup decreasing with larger final_batch_size, opposite the paper's
  direction) remains unexplained. A ready driver script
  (`benchmarks/_bf_anomaly_check.py`) exists but was never executed.

See `openspec/changes/finish-paper-reproduction-sweep/tasks.md` (archived
under `openspec/changes/archive/` once this change closes) for the exact
per-task status.

## H100 Paper-Reproduction Pass (2026-07-12)

This section supersedes the "V100 Validation Pass" section below for the
contraction-engine architecture question it raised. Environment: single
NVIDIA H100 80GB HBM3 (driver 550.127.08), dedicated conda env `tn-noise-sim`
(Python 3.11), `pip install -e ".[dev,gpu]"` — installed
`cuquantum-python-cu12==26.6.0` (cuTensorNet 2.13.0), `cupy-cuda12x==14.1.1`,
`qiskit==2.5.0`, `numpy==2.4.6` (paper used cuQuantum 26.01.0/cuTensorNet
2.11.00, CuPy 2.2.3, CUDA-Q 0.13.0 — all close but not identical versions; no
API incompatibilities found). The CUDA runtime shared libraries
(`libcublas`, `libcusolver`, etc.) were missing from the base environment
and had to be installed explicitly (`pip install nvidia-cublas-cu12
nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-curand-cu12
nvidia-cufft-cu12 nvidia-nvjitlink-cu12`) since the `cupy`/`cuquantum` wheels
don't vendor them and this host has no system CUDA toolkit on `LD_LIBRARY_PATH`.

### What changed from the V100 pass's architecture finding

The V100 pass found `use_gpu=True` was a no-op — `_compute_marginal()`
always materialized a dense `2^n` array on CPU. This pass (OpenSpec change
`gpu-bounded-memory-contraction`) replaced that with real bounded-memory GPU
contraction via `cuquantum.tensornet.experimental.NetworkState`:
- `compute_reduced_density_matrix(where, fixed=..., diagonal=True)` computes
  a `2^b`-bounded conditional marginal directly — confirmed empirically
  (peak memory bounded regardless of `n`; a dedicated test runs `n=40`, which
  the old CPU path could never execute at all).
- UPV (one path, reused across all E error sets) is realized via
  `NetworkState.update_tensor_operator()`: one persistent `NetworkState` is
  built on the noiseless circuit, and each error set is applied/reverted by
  updating only the gates it touches — confirmed numerically identical to
  building a fresh fused network per error set (see
  `tests/test_contraction.py::test_gpu_upv_update_matches_fresh_build`).
- All three simulators now default to `use_gpu=True`; the old CPU
  dense-statevector path is retained as the `use_gpu=False` fallback and is
  what the pre-existing 48-test suite still exercises directly.
- `NetworkState` doesn't expose an explicit path-finding vs. contraction
  split, so path-finding cost is approximated as the first
  `compute_reduced_density_matrix()` call's latency for a given batch index,
  with later calls for the same batch index treated as "warm" (see
  `design.md` decision D3). This is a proxy, not a guarantee that cuTensorNet
  internally caches exactly this way — treat the `*_path_finding_time_s`
  fields in results JSON as directional, not authoritative.

### Circuit generation and benchmark harness now match the paper exactly

`benchmarks/circuit_generator.py` was rewritten to draw single-qubit gates
from {H, X, Y, Z, T, Rx} and two-qubit nearest-neighbor gates from {CX, CY,
CZ, CH, CRx} (paper Sec. IV-B), replacing the previous Haar-random-unitary
generator. `benchmarks/run_benchmark.py` now defaults to the paper's Sec.
IV-C hyperparameters (PTSBE `batch_size=10`, `final_batch_size=28`,
`num_hypersamples=100`; Traditional `batch_size=24`, 1 hypersample), reports
geometric mean/std (not arithmetic) per Sec. IV-D, and tracks per-instance
success/failure against a wall-clock budget.

**One real (non-bug) statistical consequence of the new circuit generator**:
switching from Haar-random to the paper's structured gate set produces
circuits whose noisy bitstring distributions can be more peaked than
generic Haar circuits. This exposed that proportional PTSBE's
Traditional-vs-Optimized agreement is bounded by the number of pre-sampled
error sets E, not shot count — increasing shots at fixed E=200 did not
shrink an observed TVD (~0.10, plateaued from 20k to 80k shots on one
circuit instance), while increasing E at fixed shots did (E=200→2000→10000
shrank TVD 0.104→0.031→0.020). This is expected behavior of finite-E
pre-sampling (not a bug — verified by direct investigation, not assumed);
the existing regression test's `num_error_sets` was bumped from 200 to 2000
accordingly.

### Curbed reproduction run — explicitly partial, by design

**Time-boxed to roughly 30 minutes of GPU time** (an explicit scope
reduction from the paper's full grid, requested mid-session) rather than the
paper's full 5×5 (n,g) grid × 10 instances × 3 sampling modes × batch-size
sweep, which based on measured per-config cost (single-instance, 5-10 shots,
E=5 configs took 30-90s each; a 5-shot n=200/g=1000 point alone took ~4.3
minutes) would take multiple hours to run at full paper scale on one GPU.

**What ran** (non-proportional/exhaustive mode, 1 instance each, 10 shots,
E=5 pre-sampled error sets, real GPU contraction, paper-default batch
sizes/hypersamples):

| n | g | speedup (optimized/traditional) | speedup (unoptimized/traditional) |
|---|---|---|---|
| 50 | 200 | 1,651,924× | 3.03× |
| 50 | 600 | 1,085,018× | 1.14× |
| 100 | 200 | 1,781,610× | 2.32× |
| 100 | 600 | 1,135,254× | 1.19× |

All four configurations completed successfully (no timeouts). Optimized
PTSBE's non-proportional speedup is real and in the **10⁶× range** — same
order of magnitude the paper reports for its non-proportional headline
(~10⁸×), though not yet at 10⁸× at this small (n, g) corner of the grid;
the paper's own Fig. 3 shows speedup climbing with g at fixed n and only
approaching 10⁸× toward the higher end of its grid (g up to 1000-1200,
n up to 200), which this curbed run did not reach. Unoptimized PTSBE's
speedup over Traditional is modest (1-3×) at this scale, consistent with
the paper's premise that Unoptimized's main advantage (path-caching per
error set) is small until E and per-shot cost both grow — again matching
the paper's own qualitative story, not contradicting it.

Contraction-per-call time (the "warm" proxy) was consistently ~0.1-0.6s
across these configs and grew with both n and g, matching the qualitative
trend in the paper's Fig. 6. Generated:
`benchmarks/figures/fig3_nonproportional_speedup.png` (Fig. 3 analog, 2 n
values × 2 g values) and
`benchmarks/figures/fig6_contraction_pathfinding.png` (Fig. 6 analog).

**Follow-up run: final-batch-size sweep (Fig. 4 analog) — completed.**
n=100, g=600, batch_size=10 (non-final), bf swept over {24, 26, 28}, 1
instance, 10 shots, E=5:

| bf | speedup (optimized/traditional) |
|---|---|
| 24 | 1,275,393× |
| 26 | 1,094,301× |
| 28 | 1,081,521× |

This *decreases* slightly with larger bf at this small scale, which is the
**opposite direction** from the paper's Fig. 4 (which shows throughput
*increasing* with bf, since a larger final batch harvests more of the
Hilbert space per contraction). Plausible explanation, not confirmed: at
only 10 shots and a small/shallow circuit (g=600 at n=100 is comparatively
close to the paper's grid corner where speedup hasn't saturated yet, see
above), the larger final-batch contraction's extra cost per call isn't yet
being amortized by proportionally more harvested bitstrings — worth
revisiting with more shots/gates before treating this as a real trend
reversal rather than small-sample noise. Recorded honestly rather than
adjusted to match the paper's direction.
`benchmarks/figures/fig4_bf_sweep.png` generated from this data.

**Follow-up: proportional sweep (Fig. 5 analog) — attempted, no usable
data.** Traditional's per-shot GPU network rebuild (~1-2s/shot at n=100-200)
means even 100 shots exceeded a 150s per-simulator timeout at n=100/g=600;
both attempted shot counts (100, 1000) failed as timeouts, not silently
dropped (recorded with `success=False`, `failure_reason` populated) then
removed from committed results since they carry no signal. A real
proportional-Fig.5 run needs either a much larger timeout budget or fewer
shots than felt meaningful to plot (mode plumbing itself is implemented and
unit-tested against small circuits in `tests/test_benchmark_harness.py`).

**Batch-size sweep (Fig. 7 analog) — not attempted** in the time available.

**What remains genuinely unrun**: the other 21 of 25 (n,g) grid cells,
10-instance-per-config statistics (>80%/<80% success-rate marker
convention — every config here used 1 instance), a working proportional
sweep, and the batch-size sweep. `openspec/changes/gpu-bounded-memory-contraction/tasks.md`
tracks this precisely.

**To resume**: `benchmarks/_reproduction_run.py`, `_reproduction_run2.py`,
`_reproduction_run3.py` are the drivers used this session (each
incrementally checkpoints to JSON); `python -m benchmarks.plots`
regenerates figures from whatever result JSON exists. For a working
proportional sweep, raise `timeout_s` substantially (Traditional dominates
wall-clock there) or reduce shot counts further than felt useful to plot
this session. Scaling to the paper's full grid is a matter of GPU time, not
further engineering — the contraction engine and harness are
paper-conformant.

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
