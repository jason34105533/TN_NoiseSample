"""Curbed paper-reproduction sweep, sized to fit ~20-25 minutes of GPU time.

Not the paper's full 5x5 grid x 10 instances (which would take hours) -- a
small representative subset at low shots/E, per an explicit time-budget
request. Honest partial results, not a full reproduction.
"""
import json
import time
from benchmarks.run_benchmark import run_benchmark, run_batch_size_sweep

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


# ── Non-proportional mini-grid (Fig. 3 subset) ─────────────────────────────
nonprop_configs = [
    (50, 200), (50, 600),
    (100, 200), (100, 600),
    (200, 600),
]
nonprop_results = []
for n, g in nonprop_configs:
    log(f"non-proportional n={n} g={g}")
    r = run_benchmark(
        n=n, g=g, num_instances=1, num_shots=10, num_error_sets=5,
        mode="non_proportional", use_gpu=True, timeout_s=200,
    )
    nonprop_results.extend(r)
    with open("benchmarks/results_nonprop.json", "w") as f:
        json.dump(nonprop_results, f, indent=2)
    log(f"done n={n} g={g}, cumulative results={len(nonprop_results)}")

# ── Proportional reference configs (Fig. 5 subset, single mi point) ───────
prop_configs = [(100, 600), (200, 600)]
prop_results = []
for n, g in prop_configs:
    log(f"proportional n={n} g={g}")
    r = run_benchmark(
        n=n, g=g, num_instances=1, num_shots=200, num_error_sets=20,
        mode="proportional", use_gpu=True, timeout_s=200,
    )
    prop_results.extend(r)
    with open("benchmarks/results_prop.json", "w") as f:
        json.dump(prop_results, f, indent=2)
    log(f"done n={n} g={g}")

# ── Batch-size sweep (Fig. 7 subset) ───────────────────────────────────────
log("batch-size sweep n=100 g=600")
bsweep = run_batch_size_sweep(
    n=100, g=600, batch_sizes=[10, 20, 28], num_instances=1,
    num_shots=10, num_error_sets=5, output_path="benchmarks/results_batchsweep.json",
    use_gpu=True, timeout_s=200,
)
log("batch-size sweep done")

log(f"TOTAL ELAPSED {time.time() - t_start:.1f}s")
