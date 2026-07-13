"""Finish Fig. 5: the one remaining point, n=200,g=1000,mi=10000, using the
num_error_sets=5 amortization fix already validated for mi=100/1000 at this
(n,g) in a prior session."""
import json
import time
from benchmarks.run_benchmark import run_benchmark

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


RESULTS_PATH = "benchmarks/results_prop.json"
with open(RESULTS_PATH) as f:
    all_results = json.load(f)

log("n=200 g=1000 mi=10000 (E=5, timeout=900s)")
r = run_benchmark(
    n=200, g=1000, num_instances=1, num_shots=10000, baseline_num_shots=10,
    num_error_sets=5, mode="proportional", use_gpu=True, timeout_s=900,
)
all_results.extend(r)
with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
log(f"done: success={r[0]['success']}")
