"""Entry point for full benchmark suite. Run: python -m benchmarks.run_all"""
import argparse
from .run_benchmark import run_benchmark

DEFAULT_CONFIGS = [
    {"n": 100, "g": 600},
    {"n": 200, "g": 1000},
]


def main():
    parser = argparse.ArgumentParser(description="TN PTSBE benchmark suite")
    parser.add_argument("--n", type=int, default=None, help="Number of qubits")
    parser.add_argument("--g", type=int, default=None, help="Number of gates")
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--fast", action="store_true",
                        help="Use 1 hypersample for quick testing")
    parser.add_argument("--mode", type=str, default="non_proportional",
                        choices=["proportional", "non_proportional"])
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--num-error-sets", type=int, default=10)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args()

    configs = DEFAULT_CONFIGS if args.n is None else [{"n": args.n, "g": args.g}]
    all_results = []
    for cfg in configs:
        results = run_benchmark(
            n=cfg["n"],
            g=cfg["g"],
            num_instances=args.instances,
            num_shots=args.shots,
            num_error_sets=args.num_error_sets,
            mode=args.mode,
            output_path=args.output,
            fast=args.fast,
            use_gpu=not args.no_gpu,
            timeout_s=args.timeout_s,
        )
        all_results.extend(results)

    return all_results


if __name__ == "__main__":
    main()
