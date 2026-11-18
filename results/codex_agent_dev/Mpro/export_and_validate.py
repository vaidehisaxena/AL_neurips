#!/usr/bin/env python3
"""Export, validate, and compare the ten completed Mpro controller campaigns."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEEDS = tuple(range(10))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def nondecreasing(values: list[float], tolerance: float = 1e-12) -> bool:
    return all(right + tolerance >= left for left, right in zip(values, values[1:]))


trajectories: list[dict] = []
decisions: list[dict] = []
campaign_checks: dict[str, dict] = {}
(HERE / "exports").mkdir(exist_ok=True)

for seed in SEEDS:
    session = json.loads((HERE / f"session_seed_{seed}.json").read_text())
    require(session["protocol"] == "codex-agent-v1", "Unexpected protocol")
    require(session["controller_model"] == "gpt-5.6-sol", "Wrong controller model")
    require(session["target"] == "Mpro" and session["seed"] == seed, "Campaign identity mismatch")
    require(session["cycle"] == session["cycles"] == 10, "Campaign is incomplete")
    require(session["batch_size"] == 30, "Wrong batch size")
    require(len(session["history"]) == 10, "Expected ten decisions")
    require(len(session["trajectory"]) == 11, "Expected initial point plus ten cycles")
    require(len(session["observed_indices"]) == 360, "Wrong measured count")
    require(len(set(session["observed_indices"])) == 360, "Duplicate assayed compound")
    require(len(session["observed_affinities"]) == 360, "Observed label count mismatch")
    (HERE / "exports" / f"seed_{seed}.json").write_text(
        json.dumps(session, indent=2, sort_keys=True) + "\n"
    )

    expected_measured = list(range(60, 361, 30))
    require([row["measured"] for row in session["trajectory"]] == expected_measured, "Budget trajectory mismatch")

    selected_all: list[int] = []
    for cycle, decision in enumerate(session["history"], start=1):
        require(decision["cycle"] == cycle, "Decision cycle mismatch")
        counts = [decision["n_exploit"], decision["n_uncertain"], decision["n_diverse"]]
        require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in counts), "Invalid allocation count")
        require(sum(counts) == 30, "Allocation does not sum to 30")
        require(bool(decision["rationale"].strip()), "Missing rationale")
        require(bool(decision["expected_outcome"].strip()), "Missing expected outcome")
        require(len(decision["selected_indices"]) == 30, "Wrong selected batch size")
        require(len(set(decision["selected_indices"])) == 30, "Within-batch duplicate")
        selected_all.extend(decision["selected_indices"])
        action = json.loads((HERE / "actions" / f"seed_{seed}_cycle_{cycle}.json").read_text())
        for field in ("n_exploit", "n_uncertain", "n_diverse", "rationale", "expected_outcome"):
            require(action[field] == decision[field], f"Recorded action mismatch: seed={seed}, cycle={cycle}, field={field}")
        decisions.append({"target": "Mpro", "seed": seed, **decision, "selected_indices": ";".join(map(str, decision["selected_indices"]))})

    require(len(set(selected_all)) == 300, "Duplicate selection across cycles")
    require(not set(selected_all).intersection(session["observed_indices"][:60]), "Initial-set compound selected again")

    best = [float(row["best_affinity"]) for row in session["trajectory"]]
    regret = [float(row["simple_regret"]) for row in session["trajectory"]]
    for metric in ("top_1p_recall", "top_2p_recall", "top_5p_recall"):
        values = [float(row[metric]) for row in session["trajectory"]]
        require(all(0.0 <= value <= 1.0 for value in values), f"{metric} outside [0,1]")
        require(nondecreasing(values), f"{metric} is not nondecreasing")
    require(nondecreasing(best), "Best affinity is not nondecreasing")
    require(all(value >= -1e-12 for value in regret), "Negative simple regret")
    require(nondecreasing([-value for value in regret]), "Simple regret increased")

    for row in session["trajectory"]:
        trajectories.append({"target": "Mpro", "policy": "codex-agent", "controller_model": session["controller_model"], "seed": seed, **row})

    campaign_checks[str(seed)] = {
        "cycles": 10,
        "measured": 360,
        "unique_measured": 360,
        "valid_actions": 10,
        "valid_action_rate": 1.0,
        "initial_best_affinity": best[0],
        "final_best_affinity": best[-1],
        "final_simple_regret": regret[-1],
        "final_top_1p_recall": session["trajectory"][-1]["top_1p_recall"],
        "final_top_2p_recall": session["trajectory"][-1]["top_2p_recall"],
        "final_top_5p_recall": session["trajectory"][-1]["top_5p_recall"],
        "final_unique_top_5p_scaffolds": session["trajectory"][-1]["unique_top_5p_scaffolds"],
    }

trajectory_frame = pd.DataFrame(trajectories).sort_values(["seed", "cycle"])
decision_frame = pd.DataFrame(decisions).sort_values(["seed", "cycle"])
trajectory_frame.to_csv(HERE / "trajectories.csv", index=False)
decision_frame.to_csv(HERE / "decisions.csv", index=False)

baseline_frame = pd.read_csv(ROOT / "results" / "baselines_10seeds" / "trajectories.csv")
baseline_endpoints = baseline_frame[
    (baseline_frame["target"] == "Mpro")
    & (baseline_frame["seed"].isin(SEEDS))
    & (baseline_frame["cycle"] == 10)
]
agent_endpoints = trajectory_frame[trajectory_frame["cycle"] == 10]
comparison_rows: list[dict] = []
comparison_metrics = (
    "best_affinity",
    "simple_regret",
    "top_1p_recall",
    "top_2p_recall",
    "top_5p_recall",
    "unique_top_5p_scaffolds",
)
for policy, baseline_policy in baseline_endpoints.groupby("policy"):
    paired = agent_endpoints.merge(baseline_policy, on="seed", suffixes=("_agent", "_baseline"))
    require(len(paired) == len(SEEDS), f"Incomplete paired baseline: {policy}")
    for metric in comparison_metrics:
        agent_values = paired[f"{metric}_agent"]
        baseline_values = paired[f"{metric}_baseline"]
        comparison_rows.append(
            {
                "baseline_policy": policy,
                "metric": metric,
                "pairs": len(paired),
                "agent_mean": float(agent_values.mean()),
                "baseline_mean": float(baseline_values.mean()),
                "paired_delta_agent_minus_baseline": float((agent_values - baseline_values).mean()),
            }
        )
pd.DataFrame(comparison_rows).sort_values(["metric", "baseline_policy"]).to_csv(
    HERE / "paired_baseline_comparison.csv", index=False
)

git_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
).stdout.strip()
validation = {
    "status": "PASS",
    "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    "campaigns": campaign_checks,
    "checks": {
        "campaigns_complete": True,
        "allocation_schema_valid": True,
        "all_allocations_sum_to_30": True,
        "all_selected_compounds_unique": True,
        "exact_total_measured_360": True,
        "metric_ranges_valid": True,
        "best_affinity_nondecreasing": True,
        "simple_regret_nonincreasing": True,
        "recalls_nondecreasing": True,
    },
}
(HERE / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
endpoints = trajectory_frame.loc[trajectory_frame["cycle"] == 10]
metadata = {
    "status": "COMPLETE",
    "protocol": "codex-agent-v1",
    "target": "Mpro",
    "seeds": list(SEEDS),
    "controller_model": "gpt-5.6-sol",
    "campaigns": len(SEEDS),
    "cycles_per_campaign": 10,
    "batch_size": 30,
    "initial_size": 60,
    "measured_per_campaign": 360,
    "git_commit_at_export": git_commit,
    "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    "leakage_control": (
        "Allocations used only emitted safe state summaries: revealed observed affinities, "
        "GP prediction quantiles, scaffold counts, and prior revealed outcomes. Full "
        "trajectory metrics were not inspected until all campaign decisions were complete."
    ),
    "aggregate_endpoint": {
        "mean_best_affinity": float(endpoints["best_affinity"].mean()),
        "std_best_affinity": float(endpoints["best_affinity"].std(ddof=1)),
        "mean_simple_regret": float(endpoints["simple_regret"].mean()),
        "std_simple_regret": float(endpoints["simple_regret"].std(ddof=1)),
        "mean_top_1p_recall": float(endpoints["top_1p_recall"].mean()),
        "std_top_1p_recall": float(endpoints["top_1p_recall"].std(ddof=1)),
        "mean_top_2p_recall": float(endpoints["top_2p_recall"].mean()),
        "std_top_2p_recall": float(endpoints["top_2p_recall"].std(ddof=1)),
        "mean_top_5p_recall": float(endpoints["top_5p_recall"].mean()),
        "std_top_5p_recall": float(endpoints["top_5p_recall"].std(ddof=1)),
        "mean_unique_top_5p_scaffolds": float(endpoints["unique_top_5p_scaffolds"].mean()),
        "std_unique_top_5p_scaffolds": float(endpoints["unique_top_5p_scaffolds"].std(ddof=1)),
    },
}
(HERE / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
print(json.dumps({"validation": validation, "metadata": metadata}, indent=2, sort_keys=True))
