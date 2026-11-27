"""Generate 4-panel benchmark summary figure from `results/exp_results_v3.json`."""

from __future__ import annotations

import json
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _pareto_front(points: list[tuple[str, float, float]]) -> list[tuple[float, float]]:
    """Compute exact 2D minimization Pareto frontier over (overall_nll, rare_nll)."""
    pts_sorted = sorted(points, key=lambda item: (item[1], item[2]))
    front: list[tuple[float, float]] = []
    best_y = float("inf")
    for _, x, y in pts_sorted:
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


def main():
    res_path = REPO_ROOT / "results" / "exp_results_v3.json"
    d = json.loads(res_path.read_text(encoding="utf-8"))
    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.titlesize": 10.4,
            "axes.labelsize": 9.2,
            "legend.fontsize": 7.7,
            "figure.dpi": 180,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.8))

    # Panel A: Step-Level Exceedance Telemetry (Proposition 2B & Shock Selectivity)
    ax = axes[0, 0]
    ex = d["step_telemetry_and_throughput"]["exceedance_and_energy"]
    cats = [
        "Unseen Lit Rows\nNo PriorFloor (C11)",
        "Unseen Lit Rows\nWith PriorFloor (T13)",
        "Shock-Hit Tensors\n(T13)",
        "Clean Unhit Tensors\n(T13)",
    ]
    vals = [
        ex["unseen_lit_rows_exceed_pct_no_prior_floor_first25"],
        ex["unseen_lit_rows_exceed_pct_with_prior_floor_first25"],
        ex["shock_hit_tensors_exceed_pct_mean"],
        ex["clean_unhit_tensors_exceed_pct_mean"],
    ]
    cols_a = ["#d62728", "#2ca02c", "#9467bd", "#4c78a8"]
    bars = ax.bar(np.arange(len(cats)), vals, color=cols_a, width=0.52, alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.6,
            f"{v:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_ylim(0, 44)
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats, fontsize=8.3)
    ax.set_ylabel("Coordinates Exceeding Peak |z| > z_* (%)")
    ax.set_title("(A) Step Telemetry: PriorFloor Slashes Awakening False-Positives 6.9x")
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Dev Books 4-LR Sensitivity Curves under R_heavy_sas
    ax = axes[0, 1]
    lrs = [1.2e-3, 2.5e-3, 5.0e-3, 8.0e-3]
    dev_h = d["dev_lr_grid_scores"]["R_heavy_sas"]
    curve_specs = [
        ("C01_AdamW", "AdamW (C01)", "#d62728", "-o"),
        ("B16_GradClip", "Global GradClip (B16)", "#ff7f0e", "-s"),
        ("B17_StableAdamW_Exact", "Exact StableAdamW (B17)", "#8c564b", "--^"),
        ("B18_LargeEps_1e-4", "LargeEps 1e-4 (B18)", "#17becf", ":d"),
        ("N20_ExactTAdamW", "Exact TAdamW (N20)", "#e377c2", "-.v"),
        ("C09_FULD_raw", "FULD rawV (C09)", "#9467bd", "-*"),
        ("T13_FlatTop", "T13_FlatTop P(z)", "#1f77b4", "--P"),
        ("T13_AdaFULD_PriorFloor", "T13 AdaFULD ψ(z)", "#2ca02c", "-D"),
    ]
    for m, lbl, col, mk in curve_specs:
        ys = [dev_h[m][str(lr)] for lr in lrs]
        ax.plot(lrs, ys, mk, color=col, linewidth=1.6, markersize=5, label=lbl)
    ax.set_xscale("log")
    ax.set_xticks(lrs)
    ax.set_xticklabels(["1.2e-3", "2.5e-3", "5.0e-3", "8.0e-3"])
    ax.set_xlabel("Learning Rate η (Log Scale, Evaluated on Dev Books)")
    ax.set_ylabel("Dev Books NLL at Step 350 (R_heavy_sas)")
    ax.set_title("(B) Dev LR Grid: Bounded Innovations Widen High-LR Stability")
    ax.legend(loc="upper right", ncol=2, framealpha=0.92)
    ax.grid(alpha=0.25)

    # Panel C: Common Fixed LR (2.5e-3) Paired ΔNLL vs. AdamW on 8 Fresh Test Seeds (R_heavy_sas)
    ax = axes[1, 0]
    raw_test = d["raw_test_seed_runs"]
    n_s = len(raw_test)
    t_crit = float(stats.t.ppf(0.975, df=n_s - 1))
    iso_arms = [
        ("C11_FULD_cap_noFloor", "C11 (No\nPriorFloor)"),
        ("B18_LargeEps_1e-4", "B18\n(LargeEps)"),
        ("T13_AdaFULD_PriorFloor", "T13\n(AdaFULD)"),
        ("T13_Plus_MildEps", "T13 +\nMildEps"),
        ("T13_FlatTop", "T13_FlatTop\n(Monotone P)"),
    ]
    base_iso = np.array([sr["fixed_lr_2p5e3_isolation"]["R_heavy_sas"]["C01_AdamW"]["test_nll_all"] for sr in raw_test])
    mdiff, cidiff = [], []
    for a, _ in iso_arms:
        va = np.array([sr["fixed_lr_2p5e3_isolation"]["R_heavy_sas"][a]["test_nll_all"] for sr in raw_test])
        df = va - base_iso
        mdiff.append(float(df.mean()))
        cidiff.append(t_crit * float(df.std(ddof=1)) / math.sqrt(n_s))
    xb = np.arange(len(iso_arms))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax.bar(
        xb,
        mdiff,
        yerr=cidiff,
        capsize=4,
        color=["#d62728", "#17becf", "#2ca02c", "#66c2a5", "#1f77b4"],
        width=0.52,
        alpha=0.9,
    )
    ax.set_xticks(xb)
    ax.set_xticklabels([lbl for _, lbl in iso_arms])
    ax.set_ylabel("Paired ΔTestNLL vs. AdamW at lr=2.5e-3 (±95% CI)")
    ax.set_title("(C) Common Fixed LR (2.5e-3) on Held-Out Test Books (8 Seeds)")
    ax.grid(axis="y", alpha=0.25)

    # Panel D: Complete Sample-Mean Overall vs. Rare-Token Frontier (All Tuned + Common-LR Configs)
    ax = axes[1, 1]
    t_heavy = d["test_tuned_summary_8seeds"]["R_heavy_sas"]
    i_heavy = d["test_fixed_lr_2p5e3_isolation_8seeds"]["R_heavy_sas"]
    pts_d = [
        ("GradClip (2.5e-3*)", t_heavy["B16_GradClip"]["test_nll_all_mean"], t_heavy["B16_GradClip"]["rare_lit_nll_mean"], "#ff7f0e", "s"),
        ("AdamW (2.5e-3*)", t_heavy["C01_AdamW"]["test_nll_all_mean"], t_heavy["C01_AdamW"]["rare_lit_nll_mean"], "#d62728", "o"),
        ("StableAdamW (2.5e-3*)", t_heavy["B17_StableAdamW_Exact"]["test_nll_all_mean"], t_heavy["B17_StableAdamW_Exact"]["rare_lit_nll_mean"], "#8c564b", "^"),
        ("ExactTAdamW (5e-3*)", t_heavy["N20_ExactTAdamW"]["test_nll_all_mean"], t_heavy["N20_ExactTAdamW"]["rare_lit_nll_mean"], "#e377c2", "v"),
        ("LargeEps (5e-3*)", t_heavy["B18_LargeEps_1e-4"]["test_nll_all_mean"], t_heavy["B18_LargeEps_1e-4"]["rare_lit_nll_mean"], "#17becf", "d"),
        ("C09 FULD (8e-3*)", t_heavy["C09_FULD_raw"]["test_nll_all_mean"], t_heavy["C09_FULD_raw"]["rare_lit_nll_mean"], "#9467bd", "*"),
        ("C05 FlatTop (8e-3*)", t_heavy["C05_FlatTop_raw"]["test_nll_all_mean"], t_heavy["C05_FlatTop_raw"]["rare_lit_nll_mean"], "#7f7f7f", "h"),
        ("T13_FlatTop (5e-3*)", t_heavy["T13_FlatTop"]["test_nll_all_mean"], t_heavy["T13_FlatTop"]["rare_lit_nll_mean"], "#1f77b4", "P"),
        ("T13 (8e-3*)", t_heavy["T13_AdaFULD_PriorFloor"]["test_nll_all_mean"], t_heavy["T13_AdaFULD_PriorFloor"]["rare_lit_nll_mean"], "#2ca02c", "D"),
        ("CoordClip (8e-3*)", t_heavy["N19_CoordClip"]["test_nll_all_mean"], t_heavy["N19_CoordClip"]["rare_lit_nll_mean"], "#bcbd22", "<"),
        ("C11 NoFloor (2.5e-3)", i_heavy["C11_FULD_cap_noFloor"]["test_nll_all_mean"], i_heavy["C11_FULD_cap_noFloor"]["rare_lit_nll_mean"], "#a50f15", "X"),
        ("T13 (2.5e-3)", i_heavy["T13_AdaFULD_PriorFloor"]["test_nll_all_mean"], i_heavy["T13_AdaFULD_PriorFloor"]["rare_lit_nll_mean"], "#2ca02c", "o"),
        ("T13+MildEps (2.5e-3)", i_heavy["T13_Plus_MildEps"]["test_nll_all_mean"], i_heavy["T13_Plus_MildEps"]["rare_lit_nll_mean"], "#006d2c", "X"),
        ("T13_FlatTop (2.5e-3)", i_heavy["T13_FlatTop"]["test_nll_all_mean"], i_heavy["T13_FlatTop"]["rare_lit_nll_mean"], "#08519c", "p"),
    ]
    front = _pareto_front([(lbl, ox, ry) for lbl, ox, ry, _, _ in pts_d])
    fx = [p[0] for p in front]
    fy = [p[1] for p in front]
    ax.plot(fx, fy, linestyle="--", color="#333333", linewidth=1.1, alpha=0.75, label="Sample-Mean Pareto Front")

    custom_offsets = {
        "T13_FlatTop (2.5e-3)": (5, -7),
        "T13 (2.5e-3)": (5, 4),
        "T13+MildEps (2.5e-3)": (5, -4),
        "CoordClip (8e-3*)": (-48, 6),
    }
    for lbl, ox, ry, col, mk in pts_d:
        ax.scatter(ox, ry, s=72, color=col, marker=mk, edgecolors="black", linewidth=0.65, zorder=4)
        dx, dy = custom_offsets.get(lbl, (4, 2))
        ax.annotate(lbl, (ox, ry), xytext=(dx, dy), textcoords="offset points", fontsize=6.9)
    ax.set_xlim(5.075, 5.36)
    ax.set_xlabel("Held-Out Test Overall NLL (Lower is Better → Left)")
    ax.set_ylabel("Held-Out Test Rare Literary Token NLL (Lower is Better → Down)")
    ax.set_title("(D) Complete Overall vs. Rare-Token Frontier (* = Dev-Selected LR)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Single-Pass Domain Shift & Heavy-Tailed Shock Study (8 Held-Out Test Seeds, Equal LR Budget)",
        fontsize=11.8,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_png = REPO_ROOT / "figures" / "benchmark_overview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    print(f"Saved figure to {out_png}")


if __name__ == "__main__":
    main()
