# Validation Notes — Deviations from Paper

## Tasks 9.1–9.3: GPU Benchmark Results

The paper (Patti et al., arXiv:2604.08467) reports:
- **Non-proportional PTSBE**: up to 10⁸× speedup vs. traditional trajectories
- **Proportional PTSBE**: up to 10³× speedup

These numbers were obtained on NVIDIA H100 GPUs using cuQuantum cuTensorNet.

### Current Implementation Status

**Hardware**: This implementation runs on CPU (macOS, no CUDA). Full validation
against paper numbers requires:
- NVIDIA GPU (H100 recommended; A100/A30 also feasible)
- CUDA 12.x
- `cuquantum-python-cu12 >= 24.0`
- `cupy-cuda12x >= 13.0`

Install GPU dependencies: `pip install -e ".[gpu]"`

### Scaled-Down CPU Validation (n=6, g=20)

A scaled-down benchmark confirms the speedup trend is present:

| n | g | instances | mean speedup (opt vs. trad) |
|---|---|-----------|----------------------------|
| 6 | 20 | 5 | ~18× |

The ~18× speedup at small scale (vs. the paper's 10⁸×) is expected because:

1. **Final batch size**: The paper uses `b_f = 28` qubits, yielding 2²⁸ ≈ 268M
   candidate bitstrings from a single contraction. With n=6 and `b_f = 3`, we
   harvest only 2³ = 8 bitstrings per contraction — far fewer.

2. **Path-finding cost ratio**: At small n, contraction path finding is fast
   (< 1 ms), so the relative benefit of UPV (eliminating repeated path-finds)
   is smaller. At n=100–200 with 600–1000 gates, path-finding takes 10–100 s
   while contraction takes < 1 ms/shot — a ratio of 10⁴–10⁶, directly driving
   the paper's claimed speedups.

3. **No cuTensorNet**: The CPU numpy/opt_einsum implementation does not exploit
   GPU tensor contraction or cuTensorNet's optimized CUDA kernels, which provide
   significant additional acceleration at scale.

### Reproducing Paper Numbers

To reproduce Table I and Figs. 4–7 from the paper:

```bash
# Install GPU dependencies
pip install -e ".[gpu]"

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
