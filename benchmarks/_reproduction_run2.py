"""Finish the remaining curbed-sweep tasks: bf sweep (Fig.4), proportional
sweep (Fig.5), batch-size sweep (Fig.7). Kept small/fast like the first
curbed run, not the paper's full scale."""
import json
import time
from benchmarks.run_benchmark import run_benchmark, run_batch_size_sweep

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


# ── Final-batch-size sweep (Fig. 4 subset) ─────────────────────────────────
bf_results = []
for bf in [24, 26, 28]:
    log(f"bf sweep n=100 g=600 bf={bf}")
    r = run_benchmark(
        n=100, g=600, num_instances=1, num_shots=10, num_error_sets=5,
        final_batch_size=bf, mode="non_proportional", use_gpu=True, timeout_s=150,
    )
    bf_results.extend(r)
    with open("benchmarks/results_bf_sweep.json", "w") as f:
        json.dump(bf_results, f, indent=2)
    log(f"done bf={bf}")

# ── Proportional reference configs (Fig. 5 subset) ─────────────────────────
prop_results = []
for n, g in [(100, 600), (200, 600)]:
    for mi in [100, 1000]:
        log(f"proportional n={n} g={g} mi={mi}")
        r = run_benchmark(
            n=n, g=g, num_instances=1, num_shots=mi, num_error_sets=20,
            mode="proportional", use_gpu=True, timeout_s=150,
        )
        prop_results.extend(r)
        with open("benchmarks/results_prop.json", "w") as f:
            json.dump(prop_results, f, indent=2)
        log(f"done n={n} g={g} mi={mi}")

# ── Batch-size sweep (Fig. 7 subset) ────────────────────────────────────────
log("batch-size sweep n=100 g=600")
bsweep = run_batch_size_sweep(
    n=100, g=600, batch_sizes=[5, 10, 20, 28], num_instances=1,
    num_shots=10, num_error_sets=5, output_path="benchmarks/results_batchsweep.json",
    use_gpu=True, timeout_s=150,
)
log("batch-size sweep done")

log(f"TOTAL ELAPSED {time.time() - t_start:.1f}s")
