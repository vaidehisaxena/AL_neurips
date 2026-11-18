#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agentic_al.campaign import CampaignConfig, run_campaign


POLICIES = [
    "random",
    "explore",
    "heuristic",
    "sandwich",
    "exploit",
    "balanced",
    "explore-heavy",
    "exploit-heavy",
    "alternating",
]
CAMPAIGN_KEY = ["target", "policy", "seed"]


def validate_resume_outputs(
    trajectory_path: Path,
    decision_path: Path,
    *,
    initial_size: int,
    batch_size: int,
    cycles: int,
    availability_fraction: float,
    availability_salt: str,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], set[tuple[str, str, int]]]:
    """Load only complete, schema-compatible campaigns for safe append/resume."""
    if trajectory_path.exists() != decision_path.exists():
        raise ValueError("Refusing resume: trajectories.csv and decisions.csv must both exist")
    if not trajectory_path.exists():
        return [], [], set()

    trajectory = pd.read_csv(trajectory_path)
    decision = pd.read_csv(decision_path)
    required = set(CAMPAIGN_KEY + [
        "cycle",
        "initial_size",
        "batch_size",
        "cycles",
        "availability_fraction",
        "availability_salt",
        "availability_mask_digest",
        "initial_set_digest",
    ])
    for name, frame in (("trajectories", trajectory), ("decisions", decision)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Refusing resume: {name} missing columns {sorted(missing)}")
        expected = {
            "initial_size": initial_size,
            "batch_size": batch_size,
            "cycles": cycles,
            "availability_fraction": availability_fraction,
            "availability_salt": availability_salt,
        }
        for column, value in expected.items():
            if not frame[column].eq(value).all():
                raise ValueError(
                    f"Refusing resume: {name}.{column} does not match requested {value!r}"
                )

    trajectory_keys: set[tuple[str, str, int]] = set()
    for key_values, group in trajectory.groupby(CAMPAIGN_KEY, sort=False):
        key = (str(key_values[0]), str(key_values[1]), int(key_values[2]))
        cycle_values = group["cycle"].astype(int).tolist()
        if sorted(cycle_values) != list(range(cycles + 1)):
            raise ValueError(f"Refusing resume: incomplete or duplicate trajectory {key}")
        expected_measured = [initial_size + batch_size * cycle for cycle in range(cycles + 1)]
        measured_by_cycle = group.sort_values("cycle")["measured"].astype(int).tolist()
        if measured_by_cycle != expected_measured:
            raise ValueError(f"Refusing resume: incorrect measured schedule for {key}")
        if group["availability_mask_digest"].nunique() != 1 or group["initial_set_digest"].nunique() != 1:
            raise ValueError(f"Refusing resume: inconsistent provenance for {key}")
        trajectory_keys.add(key)

    decision_keys: set[tuple[str, str, int]] = set()
    for key_values, group in decision.groupby(CAMPAIGN_KEY, sort=False):
        key = (str(key_values[0]), str(key_values[1]), int(key_values[2]))
        cycle_values = group["cycle"].astype(int).tolist()
        if sorted(cycle_values) != list(range(1, cycles + 1)):
            raise ValueError(f"Refusing resume: incomplete or duplicate decisions {key}")
        if group["availability_mask_digest"].nunique() != 1 or group["initial_set_digest"].nunique() != 1:
            raise ValueError(f"Refusing resume: inconsistent provenance for {key}")
        decision_keys.add(key)

    if trajectory_keys != decision_keys:
        raise ValueError("Refusing resume: trajectory and decision campaign keys differ")
    return [trajectory], [decision], trajectory_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe deterministic AgenticAL baselines")
    parser.add_argument("--targets", nargs="+", default=["TYK2", "USP7", "D2R", "Mpro"])
    parser.add_argument("--policies", nargs="+", default=POLICIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--initial-size", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--availability-fraction", type=float, default=1.0)
    parser.add_argument("--availability-salt", default="agentic-al-availability-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.output / "trajectories.csv"
    decision_path = args.output / "decisions.csv"
    trajectories, decisions, completed = validate_resume_outputs(
        trajectory_path,
        decision_path,
        initial_size=args.initial_size,
        batch_size=args.batch_size,
        cycles=args.cycles,
        availability_fraction=args.availability_fraction,
        availability_salt=args.availability_salt,
    )
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
                    availability_fraction=args.availability_fraction,
                    availability_salt=args.availability_salt,
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
