#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["top_1p_recall", "top_2p_recall", "top_5p_recall", "simple_regret", "unique_top_5p_scaffolds"]
POLICY_ORDER = ["random", "explore", "heuristic", "sandwich", "alternating", "explore-heavy", "exploit-heavy", "balanced", "exploit"]


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(sampled, [0.025, 0.975]).tolist())


def validate(frame: pd.DataFrame, decisions: pd.DataFrame, cycles: int, measured: int) -> dict[str, int]:
    checks = {
        "campaigns": int(frame[["target", "policy", "seed"]].drop_duplicates().shape[0]),
        "trajectory_rows": len(frame),
        "decision_rows": len(decisions),
        "duplicate_trajectory_keys": int(frame.duplicated(["target", "policy", "seed", "cycle"]).sum()),
        "duplicate_decision_keys": int(decisions.duplicated(["target", "policy", "seed", "cycle"]).sum()),
        "invalid_recall_values": int((~frame["top_2p_recall"].between(0, 1) | ~frame["top_5p_recall"].between(0, 1)).sum()),
        "negative_regret_values": int((frame["simple_regret"] < -1e-10).sum()),
        "wrong_endpoint_budget": int((frame.loc[frame["cycle"] == cycles, "measured"] != measured).sum()),
        "nonpaired_initial_groups": int((frame[frame["cycle"] == 0].groupby(["target", "seed"])[["best_affinity", "top_2p_recall"]].nunique() != 1).any(axis=1).sum()),
    }
    failures = {key: value for key, value in checks.items() if key not in {"campaigns", "trajectory_rows", "decision_rows"} and value}
    if failures:
        raise ValueError(f"Result validation failed: {failures}")
    return checks


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260909)
    ordered = frame.sort_values(["target", "policy", "seed", "measured"]).copy()
    budget_span = ordered["measured"].max() - ordered["measured"].min()
    campaign_keys = ["target", "policy", "seed"]
    aulc = ordered[campaign_keys].drop_duplicates().copy().set_index(campaign_keys)
    for metric in ["top_1p_recall", "top_2p_recall", "top_5p_recall"]:
        aulc[f"aulc_{metric}"] = ordered.groupby(campaign_keys).apply(
            lambda group: np.trapz(group[metric], group["measured"]) / budget_span,
            include_groups=False,
        )
    end = ordered[ordered["cycle"] == ordered["cycle"].max()].merge(aulc.reset_index(), on=campaign_keys)
    summary_rows = []
    summary_metrics = METRICS + ["aulc_top_1p_recall", "aulc_top_2p_recall", "aulc_top_5p_recall"]
    for (target, policy), group in end.groupby(["target", "policy"], sort=True):
        row = {"target": target, "policy": policy, "seeds": len(group)}
        for metric in summary_metrics:
            low, high = bootstrap_mean_ci(group[metric].to_numpy(), rng)
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    paired_rows = []
    for target, target_group in end.groupby("target"):
        random = target_group[target_group["policy"] == "random"].set_index("seed")
        for policy, group in target_group.groupby("policy"):
            if policy == "random":
                continue
            indexed = group.set_index("seed").loc[random.index]
            for metric in ["top_2p_recall", "aulc_top_2p_recall", "simple_regret"]:
                difference = indexed[metric].to_numpy() - random[metric].to_numpy()
                low, high = bootstrap_mean_ci(difference, rng)
                paired_rows.append({
                    "target": target,
                    "policy": policy,
                    "metric": metric,
                    "mean_difference_from_random": difference.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "wins": int((difference > 0).sum()),
                    "ties": int(np.isclose(difference, 0).sum()),
                    "losses": int((difference < 0).sum()),
                })
    return summary, pd.DataFrame(paired_rows)


def plot_curves(frame: pd.DataFrame, output: Path) -> None:
    colors = dict(zip(POLICY_ORDER, plt.cm.tab10.colors[: len(POLICY_ORDER)]))
    for metric, ylabel, filename in [
        ("top_2p_recall", "Top-2% recall", "top2_recall_curves.png"),
        ("simple_regret", "Simple regret", "simple_regret_curves.png"),
    ]:
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for axis, (target, target_data) in zip(axes.flat, frame.groupby("target", sort=True)):
            stats = target_data.groupby(["policy", "measured"])[metric].agg(["mean", "sem"]).reset_index()
            for policy in POLICY_ORDER:
                policy_data = stats[stats["policy"] == policy]
                if policy_data.empty:
                    continue
                x = policy_data["measured"].to_numpy()
                mean = policy_data["mean"].to_numpy()
                sem = policy_data["sem"].to_numpy()
                axis.plot(x, mean, label=policy, color=colors[policy], linewidth=1.8)
                axis.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=colors[policy], alpha=0.12)
            axis.set_title(target)
            axis.grid(alpha=0.2)
            axis.set_ylabel(ylabel)
        for axis in axes[-1]:
            axis.set_xlabel("Measured compounds")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="center right", frameon=False)
        figure.tight_layout(rect=(0, 0, 0.86, 1))
        figure.savefig(output / filename, dpi=220, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input / "trajectories.csv")
    decisions = pd.read_csv(args.input / "decisions.csv")
    checks = validate(frame, decisions, cycles=10, measured=360)
    summary, paired = summarize(frame)
    summary.to_csv(args.output / "summary_by_target.csv", index=False)
    paired.to_csv(args.output / "paired_vs_random.csv", index=False)
    (args.output / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    plot_curves(frame, args.output)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
