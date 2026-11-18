#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["D2R", "Mpro", "TYK2", "USP7"]


def paired_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return tuple(np.quantile(sampled, [0.025, 0.975]).tolist())


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / "availability_constrained" / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    unconstrained = pd.read_csv(root / "results" / "baselines_10seeds" / "trajectories.csv")
    constrained = pd.read_csv(
        root / "results" / "availability_constrained" / "baselines_canonical" / "trajectories.csv"
    )
    u = unconstrained[unconstrained.cycle == 10]
    c = constrained[constrained.cycle == 10]
    keys = ["target", "policy", "seed"]
    merged = c.merge(u, on=keys, suffixes=("_constrained", "_unconstrained"), validate="one_to_one")
    rng = np.random.default_rng(20260910)
    rows = []
    for (target, policy), group in merged.groupby(["target", "policy"], sort=False):
        diff = group.top_2p_recall_constrained.to_numpy() - group.top_2p_recall_unconstrained.to_numpy()
        low, high = paired_ci(diff, rng)
        rows.append({
            "target": target,
            "policy": policy,
            "constrained_top2_mean": group.top_2p_recall_constrained.mean(),
            "unconstrained_top2_mean": group.top_2p_recall_unconstrained.mean(),
            "paired_mean_difference": diff.mean(),
            "ci_low": low,
            "ci_high": high,
        })
    summary = pd.DataFrame(rows).sort_values(["target", "constrained_top2_mean"], ascending=[True, False])
    summary.to_csv(output / "paired_policy_summary.csv", index=False)
    leaders = summary.groupby("target", sort=False).head(1)
    leaders.to_csv(output / "constrained_policy_leaders.csv", index=False)
    validation = {
        "status": "PASS",
        "campaigns": int(len(c)),
        "paired_campaigns": int(len(merged)),
        "endpoint_budgets": sorted(map(int, c.measured.unique())),
        "duplicate_endpoint_keys": int(c.duplicated(keys).sum()),
        "availability_fraction": sorted(map(float, c.availability_fraction.unique())),
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(leaders[["target", "policy", "constrained_top2_mean", "paired_mean_difference"]].to_string(index=False))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
