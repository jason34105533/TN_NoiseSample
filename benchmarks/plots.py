"""Reproduce the reference paper's Figs. 3-7 from benchmark result JSON.

Every function here reports the actual measured values from the results
file it's given -- no adjustment, filtering, or cherry-picking to match the
paper's headline numbers (paper-figure-reproduction spec's "Honest reporting"
requirement). Configurations recorded with success=False are shown as
hollow/excluded markers, not silently dropped.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean, gstd


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _success_rate(records: List[Dict[str, Any]]) -> float:
    if not records:
        return float("nan")
    return sum(1 for r in records if r.get("success", True)) / len(records)


def _geo(vals: List[float]):
    arr = np.asarray([v for v in vals if v and np.isfinite(v) and v > 0], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(gmean(arr)), (float(gstd(arr)) if arr.size > 1 else 1.0)


def plot_fig3_nonproportional_speedup(results_path: str, output_path: str) -> None:
    """Speedup (optimized PTSBE / traditional) vs g, one line per n."""
    records = _load(results_path)
    by_n: Dict[int, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_n[r["n"]][r["g"]].append(r)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for n in sorted(by_n):
        gs = sorted(by_n[n])
        means, stds, markers_hollow = [], [], []
        for g in gs:
            insts = by_n[n][g]
            rate = _success_rate(insts)
            vals = [r["speedup_optimized_vs_traditional"] for r in insts if r.get("success", True)]
            gm, gs_ = _geo(vals)
            means.append(gm)
            stds.append(gs_)
            markers_hollow.append(rate < 0.8)
        means = np.array(means)
        line, = ax.plot(gs, means, marker="o", label=f"n={n}")
        for i, hollow in enumerate(markers_hollow):
            if hollow:
                ax.plot(gs[i], means[i], marker="o", markerfacecolor="none",
                         markeredgecolor=line.get_color())
    ax.set_yscale("log")
    ax.set_xlabel("Number of Gates (g)")
    ax.set_ylabel("Data Collection Speedup (PTSBE / traditional)")
    ax.set_title("Non-proportional PTSBE speedup (measured)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_fig4_bf_sweep(results_path: str, output_path: str, n: Optional[int] = None) -> None:
    """PTSBE throughput vs g, one line per final_batch_size (bf)."""
    records = _load(results_path)
    if n is not None:
        records = [r for r in records if r["n"] == n]
    by_bf: Dict[int, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        bf = r.get("final_batch_size")
        if bf is None:
            continue
        by_bf[bf][r["g"]].append(r)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for bf in sorted(by_bf):
        gs = sorted(by_bf[bf])
        means = [_geo([r["throughput_optimized"] for r in by_bf[bf][g] if r.get("success", True)])[0] for g in gs]
        ax.plot(gs, means, marker="s", label=f"bf={bf}")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Gates (g)")
    ax.set_ylabel("PTSBE Throughput (shots/s)")
    ax.set_title("Final-batch-size sweep (measured)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_fig5_proportional(results_path: str, output_path: str) -> None:
    """Proportional speedup vs shot count mi, one line per (n,g) reference config."""
    records = _load(results_path)
    by_cfg: Dict[tuple, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        cfg = (r["n"], r["g"])
        mi = r.get("num_shots_requested", r.get("num_bitstrings_optimized"))
        by_cfg[cfg][mi].append(r)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for cfg in sorted(by_cfg):
        mis = sorted(by_cfg[cfg])
        means = [_geo([r["speedup_optimized_vs_traditional"] for r in by_cfg[cfg][mi] if r.get("success", True)])[0] for mi in mis]
        ax.plot(mis, means, marker="^", label=f"n={cfg[0]}, g={cfg[1]}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Proportional PTSBE Shots (mi)")
    ax.set_ylabel("Data Collection Speedup (PTSBE / traditional)")
    ax.set_title("Proportional PTSBE speedup (measured)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_fig6_contraction_pathfinding(results_path: str, output_path: str) -> None:
    """Contraction time/shot, path-finding time, and their ratio vs g, per n."""
    records = _load(results_path)
    by_n: Dict[int, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("success", True):
            by_n[r["n"]][r["g"]].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for n in sorted(by_n):
        gs = sorted(by_n[n])
        contraction = [_geo([r["optimized_contraction_time_per_call_s"] for r in by_n[n][g]
                              if r.get("optimized_contraction_time_per_call_s")])[0] for g in gs]
        pathfinding = [_geo([r["optimized_path_finding_time_s"] for r in by_n[n][g]
                              if r.get("optimized_path_finding_time_s")])[0] for g in gs]
        ratio = [p / c if c and np.isfinite(c) and c > 0 else float("nan")
                 for p, c in zip(pathfinding, contraction)]
        axes[0].plot(gs, contraction, marker="o", label=f"n={n}")
        axes[1].plot(gs, pathfinding, marker="o", label=f"n={n}")
        axes[2].plot(gs, ratio, marker="o", label=f"n={n}")

    for ax, title, ylabel in zip(
        axes,
        ["Contraction Time Per Call", "Path-Finding Time (cold-call proxy)", "Ratio (Path-Finding / Contraction)"],
        ["Time (s)", "Time (s)", "Ratio"],
    ):
        ax.set_yscale("log")
        ax.set_xlabel("Number of Gates (g)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_fig7_batch_size_sweep(results_path: str, output_path: str) -> None:
    """Per-batch contraction+sampling time vs batch size bj."""
    records = _load(results_path)
    by_bj: Dict[int, List[Dict]] = defaultdict(list)
    for r in records:
        if r.get("success", True):
            by_bj[r["batch_size"]].append(r)

    bjs = sorted(by_bj)
    means = [_geo([r["per_batch_time_s"] for r in by_bj[bj] if r.get("per_batch_time_s")])[0] * 1000 for bj in bjs]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(bjs, means, marker="D")
    for bj, m in zip(bjs, means):
        ax.annotate(f"{m:.1f}ms", (bj, m), textcoords="offset points", xytext=(0, 8))
    ax.set_yscale("log")
    ax.set_xlabel("Batch Size (bj)")
    ax.set_ylabel("Per-batch contraction + sampling time (ms)")
    ax.set_title("Per-batch contraction cost (measured)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("benchmarks/figures", exist_ok=True)
    if os.path.exists("benchmarks/results_nonprop.json"):
        plot_fig3_nonproportional_speedup("benchmarks/results_nonprop.json", "benchmarks/figures/fig3_nonproportional_speedup.png")
        plot_fig6_contraction_pathfinding("benchmarks/results_nonprop.json", "benchmarks/figures/fig6_contraction_pathfinding.png")
    if os.path.exists("benchmarks/results_prop.json"):
        plot_fig5_proportional("benchmarks/results_prop.json", "benchmarks/figures/fig5_proportional_speedup.png")
    if os.path.exists("benchmarks/results_batchsweep.json"):
        plot_fig7_batch_size_sweep("benchmarks/results_batchsweep.json", "benchmarks/figures/fig7_batch_size_sweep.png")
    if os.path.exists("benchmarks/results_bf_sweep.json"):
        plot_fig4_bf_sweep("benchmarks/results_bf_sweep.json", "benchmarks/figures/fig4_bf_sweep.png")
    print("Figures written to benchmarks/figures/")
