"""Fill in the gaps from _prop_sweep.py: n=200,g=1000 failed at all 4 mi
values with num_error_sets=20 (only ~1 shot/error-set for Unoptimized at
baseline_num_shots=20, meaning no path-finding amortization -- each of 20
error sets pays a full cold path-find on a large/deep circuit). Retry with
fewer error sets and a larger timeout. Also retry n=100,g=600 mi=10000,
which missed the 300s budget narrowly."""
import json
import time
from benchmarks.run_benchmark import run_benchmark

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


RESULTS_PATH = "benchmarks/results_prop.json"
with open(RESULTS_PATH) as f:
    all_results = json.load(f)

log("retry n=100 g=600 mi=10000")
r = run_benchmark(
    n=100, g=600, num_instances=1, num_shots=10000, baseline_num_shots=20,
    num_error_sets=20, mode="proportional", use_gpu=True, timeout_s=500,
)
all_results.extend(r)
with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
log("done n=100 g=600 mi=10000")

for mi in [10, 100, 1000, 10000]:
    log(f"retry n=200 g=1000 mi={mi} (E=5 instead of 20)")
    r = run_benchmark(
        n=200, g=1000, num_instances=1, num_shots=mi, baseline_num_shots=20,
        num_error_sets=5, mode="proportional", use_gpu=True, timeout_s=400,
    )
    all_results.extend(r)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"done n=200 g=1000 mi={mi}")

log(f"TOTAL ELAPSED {time.time() - t_start:.1f}s")
