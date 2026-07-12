"""Proportional sweep (Fig. 5): paper's two reference configs across
mi in {10,100,1000,10000}, using the new baseline_num_shots decoupling fix
(task 1 of finish-paper-reproduction-sweep) so Traditional stays fast
regardless of how large mi gets."""
import json
import time
from benchmarks.run_benchmark import run_benchmark

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


RESULTS_PATH = "benchmarks/results_prop.json"
all_results = []

configs = [(100, 600), (200, 1000)]
mis = [10, 100, 1000, 10000]

for n, g in configs:
    for mi in mis:
        log(f"proportional n={n} g={g} mi={mi}")
        r = run_benchmark(
            n=n, g=g, num_instances=1, num_shots=mi, baseline_num_shots=20,
            num_error_sets=20, mode="proportional", use_gpu=True, timeout_s=300,
        )
        all_results.extend(r)
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        log(f"done n={n} g={g} mi={mi}")

log(f"TOTAL ELAPSED {time.time() - t_start:.1f}s")
