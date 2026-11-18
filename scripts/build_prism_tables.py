#!/usr/bin/env python3
"""Build tidy, GraphPad Prism-ready CSV tables and a workbook JSON payload."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "neurips_draft" / "prism_tables"
TARGETS = ["D2R", "Mpro", "TYK2", "USP7"]
LEADERS = {"D2R": "balanced", "Mpro": "explore-heavy", "TYK2": "exploit", "USP7": "explore-heavy"}
CONSTRAINED_LEADERS = {"D2R": "balanced", "Mpro": "exploit-heavy", "TYK2": "exploit", "USP7": "explore-heavy"}
VARIANT_ORDER = ["full_reference", "minus_uncertainty", "minus_diversity", "minus_memory", "shuffled_evidence"]


def save(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame.to_csv(OUT / name, index=False, lineterminator="\n")
    return frame


def records(frame: pd.DataFrame) -> list[list[object]]:
    cleaned = frame.where(pd.notna(frame), None)
    return [list(cleaned.columns)] + cleaned.values.tolist()


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    agent_traj = pd.read_csv(ROOT / "results/codex_agent_dev/trajectories.csv")
    decisions = pd.read_csv(ROOT / "results/codex_agent_dev/decisions.csv")
    fixed_traj = pd.read_csv(ROOT / "results/baselines_10seeds/trajectories.csv")
    paired = pd.read_csv(ROOT / "figures/technical_report/paired_top2_summary.csv")
    calibration = pd.read_csv(ROOT / "results/calibration/legacy_scoring_summary.csv")
    ablations = pd.read_csv(ROOT / "results/information_ablations/full/endpoint_metrics.csv")
    constrained_agent = pd.read_csv(ROOT / "results/availability_constrained/agent_full/canonical_40_endpoints.csv")
    constrained_fixed = pd.read_csv(ROOT / "results/availability_constrained/baselines_canonical/trajectories.csv")

    agent_end = agent_traj[agent_traj.cycle.eq(10)].copy()
    fixed_end = fixed_traj[fixed_traj.cycle.eq(10)].copy()
    constrained_fixed_end = constrained_fixed[constrained_fixed.cycle.eq(10)].copy()

    endpoint_rows = []
    for target in TARGETS:
        for seed in range(10):
            a = agent_end[(agent_end.target == target) & (agent_end.seed == seed)].iloc[0]
            r = fixed_end[(fixed_end.target == target) & (fixed_end.seed == seed) & (fixed_end.policy == "random")].iloc[0]
            leader = LEADERS[target]
            b = fixed_end[(fixed_end.target == target) & (fixed_end.seed == seed) & (fixed_end.policy == leader)].iloc[0]
            endpoint_rows.append([target, seed, a.top_2p_recall, r.top_2p_recall, b.top_2p_recall, leader])
    endpoint = save(pd.DataFrame(endpoint_rows, columns=["target", "seed", "agent_top2_recall", "random_top2_recall", "best_fixed_top2_recall", "best_fixed_policy"]), "01_core_endpoint_top2.csv")

    recall_levels = save(agent_end[["target", "seed", "top_1p_recall", "top_2p_recall", "top_5p_recall", "simple_regret", "unique_top_5p_scaffolds"]].sort_values(["target", "seed"]), "02_agent_endpoint_metrics.csv")

    leader_parts = []
    for target, leader in LEADERS.items():
        leader_parts.append(fixed_traj[(fixed_traj.target == target) & (fixed_traj.policy == leader)])
    leader_traj = pd.concat(leader_parts, ignore_index=True)
    learning = pd.concat([
        agent_traj.assign(series="Agent")[["target", "seed", "cycle", "measured", "series", "top_2p_recall"]],
        fixed_traj[fixed_traj.policy.eq("random")].assign(series="Random")[["target", "seed", "cycle", "measured", "series", "top_2p_recall"]],
        leader_traj.assign(series="Best fixed")[["target", "seed", "cycle", "measured", "series", "top_2p_recall"]],
    ], ignore_index=True).sort_values(["target", "series", "seed", "cycle"])
    learning = save(learning, "03_top2_learning_curves.csv")

    allocations = save(decisions[["target", "seed", "cycle", "n_exploit", "n_uncertain", "n_diverse"]].sort_values(["target", "seed", "cycle"]), "04_agent_batch_allocations.csv")

    agent_means = agent_end.groupby("target").top_2p_recall.mean()
    fixed_means = fixed_end.groupby(["target", "policy"]).top_2p_recall.mean()
    heat_rows = []
    for target in TARGETS:
        for policy in sorted(fixed_end.policy.unique()):
            heat_rows.append([target, policy, agent_means[target], fixed_means[target, policy], agent_means[target] - fixed_means[target, policy]])
    heatmap = save(pd.DataFrame(heat_rows, columns=["target", "fixed_policy", "agent_mean_top2", "policy_mean_top2", "agent_minus_policy"]), "05_agent_vs_all_policies.csv")

    cal = calibration[calibration.scope.isin(["overall", "target"])][["scope", "group", "decisions", "scorable_decisions", "unscorable_rate", "atomic_claims", "atomic_success_rate", "mean_fraction_clauses_satisfied", "conjunction_success_rate"]].copy()
    cal = save(cal, "06_forecast_testability.csv")

    abl = ablations[["target", "variant", "seed", "top_1p_recall", "top_2p_recall", "top_5p_recall", "simple_regret", "unique_top_5p_scaffolds"]].copy()
    abl["variant"] = pd.Categorical(abl.variant, VARIANT_ORDER, ordered=True)
    abl = save(abl.sort_values(["target", "variant", "seed"]), "07_information_ablations.csv")

    availability_rows = []
    for target in TARGETS:
        leader = CONSTRAINED_LEADERS[target]
        for seed in range(10):
            full = agent_end[(agent_end.target == target) & (agent_end.seed == seed)].iloc[0]
            ca = constrained_agent[(constrained_agent.target == target) & (constrained_agent.seed == seed)].iloc[0]
            cf = constrained_fixed_end[(constrained_fixed_end.target == target) & (constrained_fixed_end.seed == seed) & (constrained_fixed_end.policy == leader)].iloc[0]
            availability_rows.append([target, seed, full.top_2p_recall, ca.top_2p_recall, cf.top_2p_recall, leader, ca.unavailable_post_initial])
    availability = save(pd.DataFrame(availability_rows, columns=["target", "seed", "full_agent_top2", "constrained_agent_top2", "constrained_best_fixed_top2", "constrained_fixed_policy", "unavailable_selections"]), "08_availability_constraints.csv")

    paired_out = save(paired.copy(), "09_paired_bootstrap_summary.csv")

    readme = pd.DataFrame([
        ["Purpose", "Numerical tables arranged for GraphPad Prism import and independent plotting."],
        ["Primary endpoint", "Top-2% recall at 360 measured compounds."],
        ["Replicates", "Ten paired seeds per target and method."],
        ["Import on Prism Mac", "Import an individual CSV into the matching Prism table type; Prism Mac does not directly import Excel workbooks."],
        ["Import on Prism Windows", "CSV is recommended; the companion XLSX is a readable index but Prism reads only the active worksheet during direct Excel import."],
        ["Missing values", "Blank cells mean not applicable; zero is a measured numeric zero."],
        ["Color scheme", "#CAA9F3, #B37AD4, #7997E6, #206ABC, #0E155E."],
        ["Important caveat", "Compare information-ablation arms only with their paired full_reference rerun, not with the earlier core benchmark."],
    ], columns=["item", "description"])
    save(readme, "00_READ_ME.csv")

    payload = {
        "Read me": records(readme),
        "Core endpoint": records(endpoint),
        "Agent metrics": records(recall_levels),
        "Learning curves": records(learning),
        "Allocations": records(allocations),
        "Policy comparison": records(heatmap),
        "Forecast scoring": records(cal),
        "Ablations": records(abl),
        "Availability": records(availability),
        "Paired statistics": records(paired_out),
    }
    (OUT.parent / "prism_workbook_data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    build()
