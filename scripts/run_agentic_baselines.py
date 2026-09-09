#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agentic_al.campaign import CampaignConfig, run_campaign


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe deterministic AgenticAL baselines")
    parser.add_argument("--targets", nargs="+", default=["TYK2", "USP7", "D2R", "Mpro"])
    parser.add_argument("--policies", nargs="+", default=["random", "explore", "exploit", "balanced", "explore-heavy", "heuristic"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--initial-size", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.output / "trajectories.csv"
    decision_path = args.output / "decisions.csv"
    trajectories = [pd.read_csv(trajectory_path)] if trajectory_path.exists() else []
    decisions = [pd.read_csv(decision_path)] if decision_path.exists() else []
    completed: set[tuple[str, str, int]] = set()
    if trajectories:
        existing = trajectories[0]
        finished = existing[existing["cycle"] == args.cycles]
        completed = set(zip(finished["target"], finished["policy"], finished["seed"].astype(int)))
    matrix = []
    for target in args.targets:
        for policy in args.policies:
            for seed in args.seeds:
                key = (target, policy, seed)
                config = CampaignConfig(
                    target=target,
                    policy=policy,
                    seed=seed,
                    initial_size=args.initial_size,
                    batch_size=args.batch_size,
                    cycles=args.cycles,
                )
                matrix.append(config.__dict__)
                if key in completed:
                    print(f"skipping completed target={target} policy={policy} seed={seed}", flush=True)
                    continue
                print(f"running target={target} policy={policy} seed={seed}", flush=True)
                trajectory, decision = run_campaign(root, config)
                trajectories.append(trajectory)
                decisions.append(decision)
                pd.concat(trajectories, ignore_index=True).to_csv(trajectory_path, index=False)
                pd.concat(decisions, ignore_index=True).to_csv(decision_path, index=False)
    if trajectories:
        pd.concat(trajectories, ignore_index=True).to_csv(trajectory_path, index=False)
    if decisions:
        pd.concat(decisions, ignore_index=True).to_csv(decision_path, index=False)
    (args.output / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")


if __name__ == "__main__":
    main()
