#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


PALETTE = {
    "phlox": "#CAA9F3",
    "verbena": "#B37AD4",
    "periwinkle": "#7997E6",
    "atlantis": "#206ABC",
    "phthalo": "#0E155E",
}
TARGETS = ["D2R", "Mpro", "TYK2", "USP7"]
LEADERS = {
    "D2R": "balanced",
    "Mpro": "explore-heavy",
    "TYK2": "exploit",
    "USP7": "explore-heavy",
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


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def endpoint_chart(agent: pd.DataFrame, baseline: pd.DataFrame, output: Path) -> None:
    endpoint_a = agent[agent.cycle == 10]
    endpoint_b = baseline[baseline.cycle == 10]
    rows = []
    for target in TARGETS:
        a = endpoint_a[endpoint_a.target == target].top_2p_recall.mean()
        r = endpoint_b[(endpoint_b.target == target) & (endpoint_b.policy == "random")].top_2p_recall.mean()
        p = LEADERS[target]
        f = endpoint_b[(endpoint_b.target == target) & (endpoint_b.policy == p)].top_2p_recall.mean()
        rows.append((target, a, r, f, p))
    frame = pd.DataFrame(rows, columns=["target", "agent", "random", "fixed", "leader"])
    x = np.arange(len(frame))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for offset, column, label, color in [
        (-width, "random", "Random", PALETTE["phlox"]),
        (0, "agent", "GPT-5.6 Sol agent", PALETTE["phthalo"]),
        (width, "fixed", "Best fixed policy", PALETTE["atlantis"]),
    ]:
        bars = ax.bar(x + offset, frame[column], width, label=label, color=color)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in frame[column]], padding=3, fontsize=8)
    ax.set_xticks(x, frame.target)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Top-2% recall at 360 measurements")
    ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    for i, row in frame.iterrows():
        ax.text(i + width, 0.015, row.leader, rotation=90, ha="center", va="bottom", fontsize=7, color="white")
    fig.tight_layout()
    fig.savefig(output / "endpoint_top2_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def learning_curves(agent: pd.DataFrame, baseline: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), sharex=True, sharey=True)
    series = [
        ("Random", PALETTE["phlox"], "random", "--"),
        ("GPT-5.6 Sol agent", PALETTE["phthalo"], "agent", "-"),
        ("Best fixed policy", PALETTE["atlantis"], "leader", "-"),
    ]
    for ax, target in zip(axes.flat, TARGETS):
        for label, color, kind, linestyle in series:
            if kind == "agent":
                data = agent[agent.target == target]
            else:
                policy = "random" if kind == "random" else LEADERS[target]
                data = baseline[(baseline.target == target) & (baseline.policy == policy)]
            stats = data.groupby("measured").top_2p_recall.agg(["mean", "sem"]).reset_index()
            x = stats.measured.to_numpy()
            y = stats["mean"].to_numpy()
            ci = 1.96 * stats["sem"].fillna(0).to_numpy()
            ax.plot(x, y, color=color, linewidth=2.2, linestyle=linestyle, label=label)
            ax.fill_between(x, np.maximum(0, y - ci), np.minimum(1, y + ci), color=color, alpha=0.12)
        ax.set_title(f"{target}   best fixed: {LEADERS[target]}", loc="left")
        ax.set_xlim(60, 360)
        ax.set_ylim(0, 0.78)
        ax.grid(axis="x", visible=False)
    axes[1, 0].set_xlabel("Measured compounds")
    axes[1, 1].set_xlabel("Measured compounds")
    axes[0, 0].set_ylabel("Top-2% recall")
    axes[1, 0].set_ylabel("Top-2% recall")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(output / "top2_learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def recall_chart(agent: pd.DataFrame, output: Path) -> None:
    endpoint = agent[agent.cycle == 10]
    metrics = [
        ("top_1p_recall", "Top 1%", PALETTE["verbena"]),
        ("top_2p_recall", "Top 2%", PALETTE["periwinkle"]),
        ("top_5p_recall", "Top 5%", PALETTE["atlantis"]),
    ]
    x = np.arange(len(TARGETS))
    width = 0.23
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for i, (metric, label, color) in enumerate(metrics):
        means = endpoint.groupby("target")[metric].mean().reindex(TARGETS)
        sds = endpoint.groupby("target")[metric].std().reindex(TARGETS)
        bars = ax.bar(x + (i - 1) * width, means, width, yerr=sds, capsize=3, label=label, color=color)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in means], padding=8, fontsize=8)
    ax.set_xticks(x, TARGETS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of true high-value compounds recovered")
    ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(output / "endpoint_recall_levels.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def allocation_chart(decisions: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3), sharex=True, sharey=True)
    columns = ["n_exploit", "n_uncertain", "n_diverse"]
    labels = ["Exploit", "Uncertain", "Diverse"]
    colors = [PALETTE["atlantis"], PALETTE["verbena"], PALETTE["phlox"]]
    for ax, target in zip(axes.flat, TARGETS):
        means = decisions[decisions.target == target].groupby("cycle")[columns].mean()
        ax.stackplot(means.index, *[means[c] for c in columns], labels=labels, colors=colors, alpha=0.96)
        ax.set_title(target, loc="left")
        ax.set_ylim(0, 30)
        ax.set_yticks([0, 10, 20, 30])
        ax.grid(axis="x", visible=False)
    axes[1, 0].set_xlabel("Acquisition cycle")
    axes[1, 1].set_xlabel("Acquisition cycle")
    axes[0, 0].set_ylabel("Mean compounds per 30-item batch")
    axes[1, 0].set_ylabel("Mean compounds per 30-item batch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(output / "agent_allocations_by_cycle.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def difference_heatmap(agent: pd.DataFrame, baseline: pd.DataFrame, output: Path) -> None:
    a = agent[agent.cycle == 10][["target", "seed", "top_2p_recall"]]
    b = baseline[baseline.cycle == 10][["target", "seed", "policy", "top_2p_recall"]]
    merged = b.merge(a, on=["target", "seed"], suffixes=("_baseline", "_agent"))
    merged["difference"] = merged.top_2p_recall_agent - merged.top_2p_recall_baseline
    table = merged.pivot_table(index="policy", columns="target", values="difference", aggfunc="mean").reindex(columns=TARGETS)
    table["Average"] = table.mean(axis=1)
    table = table.sort_values("Average", ascending=False)
    # Blue scale matching the learning-curve palette; signed labels preserve direction.
    cmap = LinearSegmentedColormap.from_list(
        "benchmark_blue",
        ["#F1F5FD", "#C6D4F5", PALETTE["periwinkle"], PALETTE["atlantis"]],
    )
    values = table.to_numpy()
    fig, ax = plt.subplots(figsize=(8.7, 6.0))
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=float(values.min()), vmax=float(values.max()))
    ax.set_xticks(np.arange(len(table.columns)), table.columns)
    ax.set_yticks(np.arange(len(table.index)), table.index)
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            value = table.iloc[i, j]
            normalized = (value - values.min()) / (values.max() - values.min())
            color = "white" if normalized > 0.66 else "#111827"
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8.5, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Agent minus fixed policy")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output / "agent_vs_policy_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(agent: pd.DataFrame, baseline: pd.DataFrame, output: Path) -> None:
    rng = np.random.default_rng(20260910)
    a = agent[agent.cycle == 10].set_index(["target", "seed"])
    b = baseline[baseline.cycle == 10].set_index(["target", "seed", "policy"])
    rows = []
    for target in TARGETS:
        av = a.loc[target].sort_index().top_2p_recall.to_numpy()
        for comparator in ("random", LEADERS[target]):
            bv = b.xs((target, comparator), level=("target", "policy")).sort_index().top_2p_recall.to_numpy()
            difference = av - bv
            low, high = bootstrap_ci(difference, rng)
            rows.append(
                {
                    "target": target,
                    "comparator": comparator,
                    "agent_mean": av.mean(),
                    "comparator_mean": bv.mean(),
                    "paired_mean_difference": difference.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "wins": int((difference > 0).sum()),
                    "ties": int(np.isclose(difference, 0).sum()),
                    "losses": int((difference < 0).sum()),
                }
            )
    pd.DataFrame(rows).to_csv(output / "paired_top2_summary.csv", index=False)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "figures" / "technical_report"
    output.mkdir(parents=True, exist_ok=True)
    agent = pd.read_csv(root / "results" / "codex_agent_dev" / "trajectories.csv")
    decisions = pd.read_csv(root / "results" / "codex_agent_dev" / "decisions.csv")
    baseline = pd.read_csv(root / "results" / "baselines_10seeds" / "trajectories.csv")
    style()
    endpoint_chart(agent, baseline, output)
    learning_curves(agent, baseline, output)
    recall_chart(agent, output)
    allocation_chart(decisions, output)
    difference_heatmap(agent, baseline, output)
    write_summary(agent, baseline, output)
    print(f"Wrote figures and summary to {output}")


if __name__ == "__main__":
    main()
