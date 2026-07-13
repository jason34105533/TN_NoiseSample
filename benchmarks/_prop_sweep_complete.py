"""Complete the remaining Fig. 5 proportional-sweep points: retry
n=100,g=600 mi=10000 with a bigger timeout, and n=200,g=1000 at
mi in {100,1000,10000} with num_error_sets=5 (the amortization fix
validated for mi=10 last session)."""
import json
import time
from benchmarks.run_benchmark import run_benchmark

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


RESULTS_PATH = "benchmarks/results_prop.json"
with open(RESULTS_PATH) as f:
    all_results = json.load(f)

log("retry n=100 g=600 mi=10000 (timeout=800s)")
r = run_benchmark(
    n=100, g=600, num_instances=1, num_shots=10000, baseline_num_shots=20,
    num_error_sets=20, mode="proportional", use_gpu=True, timeout_s=800,
)
all_results.extend(r)
with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
log(f"done n=100 g=600 mi=10000: success={r[0]['success']}")

for mi in [100, 1000, 10000]:
    log(f"n=200 g=1000 mi={mi} (E=5, timeout=800s)")
    r = run_benchmark(
        n=200, g=1000, num_instances=1, num_shots=mi, baseline_num_shots=10,
        num_error_sets=5, mode="proportional", use_gpu=True, timeout_s=800,
    )
    all_results.extend(r)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"done n=200 g=1000 mi={mi}: success={r[0]['success']}")

log(f"TOTAL ELAPSED {time.time() - t_start:.1f}s")
