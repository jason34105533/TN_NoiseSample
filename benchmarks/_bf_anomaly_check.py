"""bf-sweep anomaly follow-up (Fig. 4): last session's bf=24/26/28 sweep at
n=100,g=600 (1 instance, 10 shots) showed speedup *decreasing* with larger
bf -- opposite the paper's direction. Re-run with more shots/instances, and
separately at a deeper circuit, to check if it persists."""
import json
import time
from benchmarks.run_benchmark import run_benchmark

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


RESULTS_PATH = "benchmarks/results_bf_anomaly_check.json"
all_results = []

# More shots/instances at the same (n,g) as the original observation
for bf in [24, 26, 28]:
    log(f"n=100 g=600 bf={bf} (3 instances, 50 shots)")
    r = run_benchmark(
        n=100, g=600, num_instances=3, num_shots=50, num_error_sets=10,
        final_batch_size=bf, mode="non_proportional", use_gpu=True, timeout_s=400,
    )
    all_results.extend(r)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"done bf={bf}")

# Deeper circuit (paper notes bf's effect is more pronounced with more gates)
for bf in [24, 26, 28]:
    log(f"n=100 g=1000 bf={bf} (1 instance, 10 shots)")
    r = run_benchmark(
        n=100, g=1000, num_instances=1, num_shots=10, num_error_sets=5,
        final_batch_size=bf, mode="non_proportional", use_gpu=True, timeout_s=400,
    )
    all_results.extend(r)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"done g=1000 bf={bf}")

log(f"TOTAL ELAPSED {time.time() - t_start:.1f}s")
