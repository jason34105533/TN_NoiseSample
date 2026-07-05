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
    args = parser.parse_args()

    configs = DEFAULT_CONFIGS if args.n is None else [{"n": args.n, "g": args.g}]
    all_results = []
    for cfg in configs:
        results = run_benchmark(
            n=cfg["n"],
            g=cfg["g"],
            num_instances=args.instances,
            output_path=args.output,
            fast=args.fast,
        )
        all_results.extend(results)

    return all_results


if __name__ == "__main__":
    main()
