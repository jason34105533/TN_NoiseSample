"""Finish the proportional sweep (Fig.5) and batch-size sweep (Fig.7).
Shot counts tuned down from _reproduction_run2 (which timed out on Traditional's
100-shot proportional runs at n=100/200) so Traditional actually completes."""
import json
import time
from benchmarks.run_benchmark import run_benchmark, run_batch_size_sweep

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


# ── Proportional reference configs (Fig. 5 subset) ─────────────────────────
# mi=20,50: small enough for Traditional's per-shot GPU builds to finish
# within budget at n=100/200, while still being real, non-trivial shot counts.
prop_results = []
for n, g in [(100, 600), (200, 600)]:
    for mi in [20, 50]:
        log(f"proportional n={n} g={g} mi={mi}")
        r = run_benchmark(
            n=n, g=g, num_instances=1, num_shots=mi, num_error_sets=10,
            mode="proportional", use_gpu=True, timeout_s=200,
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
