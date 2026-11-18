#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEEDS = tuple(range(10))
MODEL = "gpt-5.6-sol"


def aulc(frame: pd.DataFrame, metric: str) -> float:
    ordered = frame.sort_values("measured")
    span = ordered["measured"].max() - ordered["measured"].min()
    return float(np.trapz(ordered[metric], ordered["measured"]) / span)


def contains_forbidden(value) -> bool:
    if isinstance(value, dict):
        if {"hidden_labels", "pool_affinity", "unrevealed_affinity"} & set(value):
            return True
        return any(contains_forbidden(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_forbidden(item) for item in value)
    return False


def main() -> None:
    trajectories, decisions, issues, per_seed = [], [], [], {}
    for seed in SEEDS:
        session = json.loads((OUT / f"export_seed{seed}.json").read_text())
        t = pd.DataFrame(session["trajectory"])
        t.insert(0, "seed", seed)
        t.insert(0, "model", MODEL)
        t.insert(0, "policy", "codex-agent-v1")
        t.insert(0, "target", "USP7")
        d = pd.DataFrame(session["history"])
        d.insert(0, "seed", seed)
        d.insert(0, "model", MODEL)
        d.insert(0, "policy", "codex-agent-v1")
        d.insert(0, "target", "USP7")
        d["selected_indices"] = d["selected_indices"].map(lambda x: ";".join(map(str, x)))
        selected_lists = [list(map(int, x.split(";"))) for x in d["selected_indices"]]
        selected = [item for group in selected_lists for item in group]
        initial = session["observed_indices"][:60]
        view_files = sorted(OUT.glob(f"view_seed{seed}_cycle*.json"))
        views = [json.loads(path.read_text()) for path in view_files]
        checks = {
            "session_complete": session["cycle"] == 10,
            "trajectory_rows_11": len(t) == 11,
            "decision_rows_10": len(d) == 10,
            "measured_schedule": t["measured"].tolist() == list(range(60, 361, 30)),
            "allocations_sum_30": bool(((d.n_exploit + d.n_uncertain + d.n_diverse) == 30).all()),
            "allocations_nonnegative": bool((d[["n_exploit", "n_uncertain", "n_diverse"]] >= 0).all().all()),
            "each_batch_30_unique": all(len(x) == len(set(x)) == 30 for x in selected_lists),
            "selected_300_unique": len(selected) == len(set(selected)) == 300,
            "initial_60_unique": len(initial) == len(set(initial)) == 60,
            "initial_selected_disjoint": not bool(set(initial) & set(selected)),
            "total_360_unique": len(set(session["observed_indices"])) == len(session["observed_indices"]) == 360,
            "model_metadata": session["controller_model"] == MODEL and set(d.model) == {MODEL},
            "rationales_present": bool(d.rationale.str.strip().ne("").all()),
            "expectations_present": bool(d.expected_outcome.str.strip().ne("").all()),
            "no_forbidden_view_keys": not any(contains_forbidden(view) for view in views),
            "metric_ranges": bool(t[["top_1p_recall", "top_2p_recall", "top_5p_recall"]].apply(lambda c: c.between(0, 1).all()).all()),
        }
        issues.extend(f"seed {seed}: {name}" for name, passed in checks.items() if not passed)
        final = t.iloc[-1]
        first_hit = t.loc[t.top_1p_recall > 0, "measured"]
        per_seed[str(seed)] = {
            "checks": checks,
            "endpoint": {
                "measured": int(final.measured),
                "best_affinity": float(final.best_affinity),
                "simple_regret": float(final.simple_regret),
                "top_1p_recall": float(final.top_1p_recall),
                "top_2p_recall": float(final.top_2p_recall),
                "top_5p_recall": float(final.top_5p_recall),
                "unique_top_5p_scaffolds": int(final.unique_top_5p_scaffolds),
                "experiments_to_first_top_1p_hit": int(first_hit.iloc[0]) if len(first_hit) else None,
                "aulc_top_2p_recall": aulc(t, "top_2p_recall"),
            },
        }
        trajectories.append(t)
        decisions.append(d)

    trajectories = pd.concat(trajectories, ignore_index=True)
    decisions = pd.concat(decisions, ignore_index=True)
    trajectories.to_csv(OUT / "trajectories.csv", index=False)
    decisions.to_csv(OUT / "decisions.csv", index=False)

    base = pd.read_csv(ROOT / "results/baselines_10seeds/trajectories.csv")
    base = base[(base.target == "USP7") & base.seed.isin(SEEDS) & (base.cycle == 10)]
    agent = trajectories[trajectories.cycle == 10]
    rows = []
    for policy, group in base.groupby("policy"):
        paired = agent.merge(group, on="seed", suffixes=("_agent", "_baseline"))
        rows.append({
            "baseline_policy": policy,
            "agent_top_2p_recall_mean": paired.top_2p_recall_agent.mean(),
            "baseline_top_2p_recall_mean": paired.top_2p_recall_baseline.mean(),
            "paired_delta_top_2p_recall": (paired.top_2p_recall_agent - paired.top_2p_recall_baseline).mean(),
        })
    pd.DataFrame(rows).sort_values("baseline_policy").to_csv(OUT / "paired_baseline_comparison.csv", index=False)

    validation = {"status": "PASS" if not issues else "FAIL", "target": "USP7", "model": MODEL, "valid_actions": len(decisions), "invalid_actions": 0, "issues": issues, "per_seed": per_seed}
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    runner = ROOT / "scripts/codex_agent_session.py"
    metadata = {
        "status": validation["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": "USP7",
        "controller_model": MODEL,
        "controller_role": "Codex GPT-5.6 Sol strategy controller (delegated sub-agent)",
        "protocol": "codex-agent-v1; 60 initial + 10x30; actions use leakage-safe summaries only",
        "campaigns": len(SEEDS),
        "actions": 10 * len(SEEDS),
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
