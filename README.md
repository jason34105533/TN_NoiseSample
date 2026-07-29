# TN_NoiseSample

A tensor-network simulator for sampling noisy quantum circuit trajectories on GPU, built to reproduce the results of [arXiv:2604.08467](https://arxiv.org/abs/2604.08467), *"Accelerating Quantum Tensor Network Simulations with Unified Path Variations and Non-Degenerate Batched Sampling."*

## What this repo is for

The paper proposes two techniques for speeding up trajectory-based simulation of noisy quantum circuits: **Unified Path Variations (UPV)**, which finds a contraction path once on the noiseless circuit and reuses it across every sampled error configuration instead of re-searching for each one, and **Non-Degenerate Batched Sampling (NBS)**, which contracts a whole batch of qubits' worth of amplitudes at once instead of extracting one bitstring per contraction. Together the paper reports speedups of roughly 10⁸× (exhaustive, non-proportional sampling) and 10³× (proportional sampling) over a traditional per-shot baseline.

Our goal was to actually build this — not just read about it — on real hardware matching the paper's own setup (a single NVIDIA H100 80GB), and see how close we could get to the reported numbers, being upfront about wherever our results fall short or diverge. The three simulators in `src/tn_noise_sim/simulators/` (`traditional.py`, `unoptimized_ptsbe.py`, `optimized_ptsbe.py`) mirror the paper's own ablation: no optimizations, NBS only, and UPV+NBS together, all sharing the same GPU contraction engine (`src/tn_noise_sim/contraction.py`), built on `cuquantum.tensornet.experimental.NetworkState`.

## What we've reproduced so far

Reproduction was done in stages, tracked as a series of [OpenSpec](https://github.com/Fission-AI/OpenSpec) changes (see `openspec/changes/archive/`), each one archived with its own design notes and task list. The honest, un-adjusted results — including where we fall short of the paper or measure a trend running the other way — are recorded in full in [`benchmarks/validation_notes.md`](benchmarks/validation_notes.md) and summarized in [`benchmarks/reproduction_report.md`](benchmarks/reproduction_report.md) (also available as a formatted [PDF](benchmarks/reproduction_report.pdf)).

| Figure | What it shows | Status |
|---|---|---|
| Fig. 3 — non-proportional speedup | Speedup vs. traditional sampling across the (n, g) grid | 19 of 25 (n, g) cells run; speedup lands in the 10⁵–10⁶× range, same order of magnitude as the paper but not yet at its 10⁸× ceiling |
| Fig. 4 — final batch size sweep | PTSBE throughput vs. g at several final batch sizes, at the paper's n=200 | 7 of 9 grid points; throughput comes out essentially flat across batch sizes rather than the paper's reported 2–4× step, at 1 instance per point |
| Fig. 5 — proportional speedup | Speedup vs. shot count at two reference configurations | Speedup climbs correctly with shot count (up to 59.5×) but hasn't reached the paper's ~10³× plateau; one data point (n=200, g=1000, largest shot count) still times out |
| Fig. 6 — contraction / path-finding time | Per-call contraction cost vs. g | Generated from the extended non-proportional grid; qualitative trend matches the paper |
| Fig. 7 — batch size vs. per-batch cost | Per-batch contraction time vs. batch size | 6 of 7 points; cost rises with batch size as expected, largest tested size times out |

None of the above numbers have been tuned or filtered to look better than they are — that's a hard requirement written into `openspec/specs/paper-figure-reproduction/spec.md`. Where a result falls short of the paper, or a trend runs in the opposite direction, it's stated plainly in the validation notes rather than smoothed over. The full breakdown of what's still missing (the last Fig. 5 point, two Fig. 4 grid cells, deeper per-cell statistics, a couple of unexplained trend divergences) is in the reproduction report's closing sections.

## Getting set up

```bash
git clone https://github.com/jason34105533/TN_NoiseSample.git
cd TN_NoiseSample
pip install -e ".[dev]"       # CPU-only, enough to run the test suite
pip install -e ".[dev,gpu]"   # add cupy + cuquantum for real GPU contraction
```

The GPU path needs an NVIDIA GPU, CUDA 12, and cuQuantum's runtime libraries. If `cupy`/`cuquantum` can't find `libcublas` and friends at import time, they're likely missing from the environment rather than the system — installing `nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-nvjitlink-cu12` alongside the `gpu` extras resolves it. Without a GPU, everything still runs — simulators fall back to a CPU dense-statevector path, which is correct but only practical for small circuits (n well under 30).

## Using it

**Run the test suite:**
```bash
pytest tests/
```
GPU-only tests are marked and will skip automatically if no GPU is available.

**Run a single benchmark configuration:**
```python
from benchmarks.run_benchmark import run_benchmark

results = run_benchmark(
    n=100, g=600, num_instances=3, num_shots=100,
    mode="non_proportional",   # or "proportional"
    use_gpu=True,
)
```
This runs all three simulators (traditional, unoptimized PTSBE, optimized PTSBE) on the same circuit instances and records timing, throughput, and speedup for each — see `benchmarks/run_benchmark.py` for the full set of parameters (batch sizes, hypersamples, number of error sets, timeout).

**Run the full benchmark suite from the command line:**
```bash
python -m benchmarks.run_all --n 100 --g 600 --instances 10 --output results.json
```

**Regenerate the figures from whatever result JSON exists:**
```bash
python -m benchmarks.plots
```
Output goes to `benchmarks/figures/`.

**Use a simulator directly**, if you want to sample a circuit without going through the benchmark harness:
```python
from tn_noise_sim.simulators.optimized_ptsbe import OptimizedPTSBESimulator
from tn_noise_sim.noise_model import NoiseModel

sim = OptimizedPTSBESimulator(batch_size=10, final_batch_size=28, use_gpu=True)
shots = sim.sample(circuit, noise_model, num_shots=1000, mode="non_proportional")
```

## Repository layout

- `src/tn_noise_sim/` — the simulator package: contraction engine, tensor network construction, noise model, error sampling, and the three simulators.
- `benchmarks/` — the paper's circuit generator, the benchmarking harness, plotting scripts, and the validation notes / reproduction report.
- `tests/` — unit tests, split between CPU-only (always run) and GPU-only (skipped without a GPU).
- `openspec/` — the spec-driven development trail: current specs under `specs/`, and every completed change (with its design rationale) under `changes/archive/`.

## A note on scope

This was run on a single GPU with limited continuous session time, so most (n, g) configurations were run with 1 circuit instance rather than the paper's 10, and the grid isn't fully covered. The engineering and the underlying mechanism (UPV, NBS, bounded-memory GPU contraction) are in place and tested; what's missing is mostly a matter of more GPU time, not further implementation work. `benchmarks/reproduction_report.md` lays out exactly what's done and what's left.
