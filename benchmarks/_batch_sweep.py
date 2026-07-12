"""Batch-size sweep (Fig. 7): n=100, g=600 across bj in {2,5,10,15,20,24,28}."""
import time
from benchmarks.run_benchmark import run_batch_size_sweep

t_start = time.time()
print(f"[0.0s] batch-size sweep n=100 g=600", flush=True)
run_batch_size_sweep(
    n=100, g=600, batch_sizes=[2, 5, 10, 15, 20, 24, 28], num_instances=1,
    num_shots=10, num_error_sets=5, output_path="benchmarks/results_batchsweep.json",
    use_gpu=True, timeout_s=200,
)
print(f"[{time.time()-t_start:.1f}s] TOTAL ELAPSED", flush=True)
