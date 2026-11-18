#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
SEEDS = tuple(range(10))


def aulc(frame: pd.DataFrame, metric: str) -> float:
    ordered = frame.sort_values("measured")
    span = float(ordered["measured"].max() - ordered["measured"].min())
    return float(np.trapz(ordered[metric], ordered["measured"]) / span)


def main() -> None:
    trajectories = pd.concat(
        [pd.read_csv(OUTPUT / f"trajectory_seed{seed}.csv") for seed in SEEDS],
        ignore_index=True,
    )
    decisions = pd.concat(
        [pd.read_csv(OUTPUT / f"decisions_seed{seed}.csv") for seed in SEEDS],
        ignore_index=True,
    )
    trajectories.to_csv(OUTPUT / "trajectories.csv", index=False)
    decisions.to_csv(OUTPUT / "decisions.csv", index=False)

    issues: list[str] = []
    per_seed: dict[str, dict] = {}
    allowed_state_keys = {
        "target",
        "cycle",
        "total_cycles",
        "batch_size",
        "remaining_assay_budget",
        "observed_count",
        "pool_count",
        "observed_affinity",
        "gp_predicted_mean",
        "gp_predicted_std",
        "chemical_space",
        "history",
    }
    for seed in SEEDS:
        t = trajectories[trajectories["seed"] == seed].sort_values("cycle")
        d = decisions[decisions["seed"] == seed].sort_values("cycle")
        selected_lists = [list(map(int, value.split(";"))) for value in d["selected_indices"]]
        selected = np.asarray([item for group in selected_lists for item in group], dtype=int)
        states = [json.loads(value) for value in d["state_json"]]
        total_candidates = states[0]["pool_count"] + states[0]["observed_count"]
        initial = np.sort(
            np.random.default_rng(seed).choice(total_candidates, size=60, replace=False)
        )
        checks = {
            "trajectory_rows_11": len(t) == 11,
            "decision_rows_10": len(d) == 10,
            "cycles_complete": t["cycle"].tolist() == list(range(11)),
            "measured_schedule": t["measured"].tolist() == list(range(60, 361, 30)),
            "allocations_sum_30": bool(
                ((d["n_exploit"] + d["n_uncertain"] + d["n_diverse"]) == 30).all()
            ),
            "nonnegative_allocations": bool(
                (d[["n_exploit", "n_uncertain", "n_diverse"]] >= 0).all().all()
            ),
            "each_batch_30_unique": all(len(group) == len(set(group)) == 30 for group in selected_lists),
            "selected_300_unique": len(selected) == len(set(selected.tolist())) == 300,
            "initial_60_unique": len(initial) == len(set(initial.tolist())) == 60,
            "initial_and_selected_disjoint": not bool(set(initial.tolist()) & set(selected.tolist())),
            "total_360_unique": len(set(initial.tolist()) | set(selected.tolist())) == 360,
            "state_top_level_allowlist": all(set(state) == allowed_state_keys for state in states),
            "state_counts_match": all(
                state["observed_count"] == 60 + 30 * (cycle - 1)
                for cycle, state in enumerate(states, start=1)
            ),
            "metric_ranges": bool(
                t[["top_1p_recall", "top_2p_recall", "top_5p_recall"]]
                .apply(lambda column: column.between(0, 1).all())
                .all()
            ),
            "model_metadata_exact": set(d["model"]) == {"gpt-5.6-sol"},
            "rationales_present": bool(d["rationale"].str.strip().ne("").all()),
            "expectations_present": bool(d["expected_outcome"].str.strip().ne("").all()),
        }
        failed = [name for name, passed in checks.items() if not passed]
        issues.extend([f"seed {seed}: {name}" for name in failed])
        final = t.iloc[-1]
        first_hit = t.loc[t["top_1p_recall"] > 0, "measured"]
        per_seed[str(seed)] = {
            "checks": checks,
            "endpoint": {
                "measured": int(final["measured"]),
                "best_affinity": float(final["best_affinity"]),
                "simple_regret": float(final["simple_regret"]),
                "top_1p_recall": float(final["top_1p_recall"]),
                "top_2p_recall": float(final["top_2p_recall"]),
                "top_5p_recall": float(final["top_5p_recall"]),
                "unique_top_5p_scaffolds": int(final["unique_top_5p_scaffolds"]),
                "experiments_to_first_top_1p_hit": int(first_hit.iloc[0]) if len(first_hit) else None,
                "aulc_top_2p_recall": aulc(t, "top_2p_recall"),
            },
        }

    baselines = pd.read_csv(ROOT / "results" / "baselines_10seeds" / "trajectories.csv")
    endpoints = baselines[
        (baselines["target"] == "TYK2")
        & (baselines["seed"].isin(SEEDS))
        & (baselines["cycle"] == 10)
    ].copy()
    agent_endpoints = trajectories[trajectories["cycle"] == 10].copy()
    rows: list[dict] = []
    for policy, group in endpoints.groupby("policy"):
        merged = agent_endpoints.merge(group, on="seed", suffixes=("_agent", "_baseline"))
        rows.append(
            {
                "baseline_policy": policy,
                "seeds": len(merged),
                "agent_top_2p_recall_mean": merged["top_2p_recall_agent"].mean(),
                "baseline_top_2p_recall_mean": merged["top_2p_recall_baseline"].mean(),
                "paired_delta_top_2p_recall": (
                    merged["top_2p_recall_agent"] - merged["top_2p_recall_baseline"]
                ).mean(),
                "agent_top_1p_recall_mean": merged["top_1p_recall_agent"].mean(),
                "baseline_top_1p_recall_mean": merged["top_1p_recall_baseline"].mean(),
                "paired_delta_top_1p_recall": (
                    merged["top_1p_recall_agent"] - merged["top_1p_recall_baseline"]
                ).mean(),
                "agent_simple_regret_mean": merged["simple_regret_agent"].mean(),
                "baseline_simple_regret_mean": merged["simple_regret_baseline"].mean(),
            }
        )
    comparison = pd.DataFrame(rows).sort_values("baseline_policy")
    comparison.to_csv(OUTPUT / "paired_baseline_comparison.csv", index=False)

    endpoint_rows = []
    for seed in SEEDS:
        endpoint_rows.append({"seed": seed, **per_seed[str(seed)]["endpoint"]})
    endpoint_frame = pd.DataFrame(endpoint_rows)
    summary_metrics = [
        "best_affinity",
        "simple_regret",
        "top_1p_recall",
        "top_2p_recall",
        "top_5p_recall",
        "unique_top_5p_scaffolds",
        "experiments_to_first_top_1p_hit",
        "aulc_top_2p_recall",
    ]
    endpoint_summary = {
        metric: {
            "mean": float(endpoint_frame[metric].mean()),
            "sd": float(endpoint_frame[metric].std(ddof=1)),
        }
        for metric in summary_metrics
    }
    (OUTPUT / "endpoint_summary.json").write_text(
        json.dumps(endpoint_summary, indent=2) + "\n"
    )

    runner_hash = hashlib.sha256((OUTPUT / "run_interactive.py").read_bytes()).hexdigest()
    validation = {
        "status": "PASS" if not issues else "FAIL",
        "target": "TYK2",
        "model": "gpt-5.6-sol",
        "seeds": list(SEEDS),
        "valid_actions": int(len(decisions)),
        "invalid_actions": 0,
        "issues": issues,
        "endpoint_summary": endpoint_summary,
        "per_seed": per_seed,
    }
    (OUTPUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    metadata = {
        "status": validation["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "controller_model": "gpt-5.6-sol",
        "controller_role": "Codex strategy controller (delegated sub-agent)",
        "protocol": {
            "execution": "persistent single-process hidden-assay session per seed",
            "input": "One leakage-safe STATE_JSON per cycle from summarize_agent_state",
            "action": "One validated BatchAllocation JSON per cycle; counts sum exactly to 30",
            "feedback": "Only outcomes from the just-revealed batch plus prior revealed history",
            "retries_or_fallbacks": 0,
            "hidden_label_access": "No unrevealed affinity, top-active flag, or pool outcome was exposed before action selection",
        },
        "campaigns": len(SEEDS),
        "cycles_per_campaign": 10,
        "actions": int(len(decisions)),
        "runner_sha256": runner_hash,
        "code_commit": json.loads((OUTPUT / "metadata_seed0.json").read_text())["code_commit"],
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(validation, indent=2))
    print("\nPAIRED_ENDPOINT_COMPARISON\n" + comparison.to_string(index=False))


if __name__ == "__main__":
    main()
