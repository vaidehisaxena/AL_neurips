#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _session_files(root: Path) -> list[Path]:
    files = list(root.glob("*/session_seed*.json")) + list(root.glob("*/session_seed_*.json"))
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/codex_agent_dev"))
    parser.add_argument("--baselines", type=Path, default=Path("results/baselines_10seeds/trajectories.csv"))
    args = parser.parse_args()

    trajectories: list[dict] = []
    decisions: list[dict] = []
    for path in _session_files(args.input):
        session = json.loads(path.read_text())
        if session.get("cycle") != session.get("cycles"):
            continue
        common = {
            "target": session["target"],
            "policy": "codex-agent",
            "model": session.get("controller_model", "gpt-5.6-sol"),
            "seed": session["seed"],
        }
        trajectories.extend({**common, **row} for row in session["trajectory"])
        decisions.extend({**common, **row} for row in session["history"])

    for target_dir in sorted(args.input.iterdir()):
        for path in sorted(target_dir.glob("trajectory_seed*.csv")):
            trajectories.extend(pd.read_csv(path).to_dict("records"))
        for path in sorted(target_dir.glob("decisions_seed*.csv")):
            decisions.extend(pd.read_csv(path).to_dict("records"))

    trajectory = pd.DataFrame(trajectories).drop_duplicates(
        ["target", "policy", "seed", "cycle"], keep="last"
    )
    decision = pd.DataFrame(decisions).drop_duplicates(
        ["target", "policy", "seed", "cycle"], keep="last"
    )
    endpoint = trajectory[trajectory["cycle"] == 10].copy()
    validation = {
        "complete_campaigns": int(len(endpoint)),
        "trajectory_rows": int(len(trajectory)),
        "decision_rows": int(len(decision)),
        "invalid_allocation_totals": int(
            ((decision[["n_exploit", "n_uncertain", "n_diverse"]].sum(axis=1)) != 30).sum()
        ),
        "wrong_endpoint_budget": int((endpoint["measured"] != 360).sum()),
        "duplicate_campaign_cycles": int(
            trajectory.duplicated(["target", "policy", "seed", "cycle"]).sum()
        ),
    }
    if validation["invalid_allocation_totals"] or validation["wrong_endpoint_budget"]:
        raise ValueError(f"Validation failed: {validation}")

    baselines = pd.read_csv(args.baselines)
    base_end = baselines[baselines["cycle"] == 10]
    comparison_rows = []
    for row in endpoint.itertuples(index=False):
        paired = base_end[(base_end.target == row.target) & (base_end.seed == row.seed)]
        for metric in ("top_2p_recall", "simple_regret", "best_affinity"):
            values = paired.set_index("policy")[metric]
            comparison_rows.append(
                {
                    "target": row.target,
                    "seed": int(row.seed),
                    "metric": metric,
                    "codex_agent": float(getattr(row, metric)),
                    "random": float(values["random"]),
                    "best_fixed": float(values.max() if metric != "simple_regret" else values.min()),
                    "delta_from_random": float(getattr(row, metric) - values["random"]),
                }
            )

    args.input.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(args.input / "trajectories.csv", index=False)
    decision.to_csv(args.input / "decisions.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(args.input / "paired_baseline_comparison.csv", index=False)
    (args.input / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
