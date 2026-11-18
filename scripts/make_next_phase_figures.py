#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "technical_report"
TARGETS = ["D2R", "Mpro", "TYK2", "USP7"]
PALETTE = {
    "phlox": "#CAA9F3",
    "verbena": "#B37AD4",
    "periwinkle": "#7997E6",
    "atlantis": "#206ABC",
    "phthalo": "#0E155E",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#AEB7C6",
            "axes.grid": True,
            "grid.color": "#DDE3EC",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def calibration_figure() -> None:
    data = pd.read_csv(ROOT / "results" / "calibration" / "legacy_scoring_summary.csv")
    data = data[(data.scope == "target") & data.group.isin(TARGETS)].set_index("group").reindex(TARGETS)
    scorable = 1.0 - data.unscorable_rate.to_numpy(float)
    conjunction = data.conjunction_success_rate.to_numpy(float)
    x = np.arange(len(TARGETS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    a = ax.bar(x - width / 2, scorable, width, color=PALETTE["periwinkle"], label="Forecasts with testable thresholds")
    b = ax.bar(x + width / 2, conjunction, width, color=PALETTE["phthalo"], label="All stated clauses succeeded")
    ax.bar_label(a, labels=[f"{v:.0%}" for v in scorable], padding=3, fontsize=8.5)
    ax.bar_label(b, labels=[f"{v:.0%}" for v in conjunction], padding=3, fontsize=8.5)
    ax.set_xticks(x, TARGETS)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Fraction of decisions")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "forecast_testability_and_success.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def availability_figure() -> None:
    leaders = pd.read_csv(
        ROOT / "results" / "availability_constrained" / "analysis" / "constrained_policy_leaders.csv"
    ).set_index("target").reindex(TARGETS)
    x = np.arange(len(TARGETS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    unconstrained = leaders.unconstrained_top2_mean.to_numpy(float)
    constrained = leaders.constrained_top2_mean.to_numpy(float)
    a = ax.bar(x - width / 2, unconstrained, width, color=PALETTE["phlox"], label="Full library")
    b = ax.bar(x + width / 2, constrained, width, color=PALETTE["phthalo"], label="80% available")
    ax.bar_label(a, labels=[f"{v:.3f}" for v in unconstrained], padding=3, fontsize=8)
    ax.bar_label(b, labels=[f"{v:.3f}" for v in constrained], padding=3, fontsize=8)
    ax.set_xticks(x, [f"{t}\n{leaders.loc[t, 'policy']}" for t in TARGETS])
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Mean top-2% recall at 360 measurements")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "availability_policy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def availability_agent_figure() -> None:
    agent_full = pd.read_csv(ROOT / "results" / "codex_agent_dev" / "trajectories.csv")
    agent_full = agent_full[agent_full.cycle == 10].groupby("target").top_2p_recall.mean().reindex(TARGETS)
    agent_available = pd.read_csv(
        ROOT / "results" / "availability_constrained" / "agent_full" / "canonical_40_endpoints.csv"
    ).groupby("target").top_2p_recall.mean().reindex(TARGETS)
    fixed_available = pd.read_csv(
        ROOT / "results" / "availability_constrained" / "analysis" / "constrained_policy_leaders.csv"
    ).set_index("target").reindex(TARGETS)
    x = np.arange(len(TARGETS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.3, 4.9))
    series = [
        (agent_full.to_numpy(float), "Agent, full library", PALETTE["phlox"]),
        (agent_available.to_numpy(float), "Agent, 80% available", PALETTE["phthalo"]),
        (fixed_available.constrained_top2_mean.to_numpy(float), "Best fixed, 80% available", PALETTE["atlantis"]),
    ]
    for offset, (values, label, color) in zip((-width, 0, width), series):
        bars = ax.bar(x + offset, values, width, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v + 1e-12:.3f}" for v in values], padding=3, fontsize=8)
    ax.set_xticks(x, TARGETS)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Mean top-2% recall at 360 measurements")
    ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "availability_agent_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def ablation_figure() -> None:
    path = ROOT / "results" / "information_ablations" / "full" / "endpoint_metrics.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    variants = ["full_reference", "minus_uncertainty", "minus_diversity", "minus_memory", "shuffled_evidence"]
    labels = ["Full", "No uncertainty", "No diversity", "No memory", "Shuffled evidence"]
    means = data.groupby(["target", "variant"]).top_2p_recall.mean().unstack("variant").reindex(TARGETS)
    x = np.arange(len(TARGETS))
    width = 0.16
    colors = [PALETTE["phthalo"], PALETTE["atlantis"], PALETTE["periwinkle"], PALETTE["verbena"], PALETTE["phlox"]]
    fig, ax = plt.subplots(figsize=(10.2, 5.1))
    for i, (variant, label, color) in enumerate(zip(variants, labels, colors)):
        values = means[variant].to_numpy(float)
        ax.bar(x + (i - 2) * width, values, width, color=color, label=label)
    ax.set_xticks(x, TARGETS)
    ax.set_ylim(0, min(1.0, float(means.max().max()) + 0.15))
    ax.set_ylabel("Mean top-2% recall at 360 measurements")
    ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "information_ablation_endpoints.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    calibration_figure()
    availability_figure()
    availability_agent_figure()
    ablation_figure()
    print(f"Wrote next-phase figures to {OUT}")


if __name__ == "__main__":
    main()
