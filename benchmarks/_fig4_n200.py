"""Corrected Fig. 4 reproduction: the paper runs this bf sweep at n=200
(Sec. V-A), not n=100 as prior sessions mistakenly used. Real g x bf grid:
g in {200,600,1000}, bf in {24,26,28}, 9 points at n=200."""
import json
import time
from benchmarks.run_benchmark import run_benchmark

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


RESULTS_PATH = "benchmarks/results_bf_sweep_n200.json"
all_results = []

for g in [200, 600, 1000]:
    for bf in [24, 26, 28]:
        log(f"n=200 g={g} bf={bf}")
        r = run_benchmark(
            n=200, g=g, num_instances=1, num_shots=10, num_error_sets=5,
            final_batch_size=bf, mode="non_proportional", use_gpu=True, timeout_s=500,
        )
        all_results.extend(r)
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        log(f"done g={g} bf={bf}: success={r[0]['success']}")

log(f"TOTAL ELAPSED {time.time() - t_start:.1f}s")
