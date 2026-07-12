"""Extend the non-proportional grid (Fig. 3) beyond the 4 cells already run.
Prioritizes breadth (spread across n and g) over depth (instances per cell),
per design.md D2/D3 of the finish-paper-reproduction-sweep change. Appends
to the existing benchmarks/results_nonprop.json."""
import json
import time
from benchmarks.run_benchmark import run_benchmark

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


RESULTS_PATH = "benchmarks/results_nonprop.json"

with open(RESULTS_PATH) as f:
    all_results = json.load(f)
already_done = {(r["n"], r["g"]) for r in all_results}
log(f"already have: {sorted(already_done)}")

# New cells: spread across all 5 n values and a range of g, breadth-first.
new_configs = [
    (50, 400), (50, 800), (50, 1000),
    (75, 200), (75, 600), (75, 1000),
    (100, 400), (100, 800), (100, 1000),
    (150, 200), (150, 600), (150, 1000),
    (200, 200), (200, 600), (200, 1000),
]

for n, g in new_configs:
    if (n, g) in already_done:
        log(f"skip n={n} g={g} (already have)")
        continue
    log(f"non-proportional n={n} g={g}")
    r = run_benchmark(
        n=n, g=g, num_instances=1, num_shots=10, num_error_sets=5,
        mode="non_proportional", use_gpu=True, timeout_s=400,
    )
    all_results.extend(r)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"done n={n} g={g}, cumulative={len(all_results)}")

log(f"TOTAL ELAPSED {time.time() - t_start:.1f}s")
