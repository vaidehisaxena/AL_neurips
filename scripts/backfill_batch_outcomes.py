#!/usr/bin/env python3
"""Reconstruct auditable next-batch outcomes for the completed legacy benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_al.campaign import load_target, scaffolds


KEYS = ["target", "seed", "cycle"]
TOLERANCE = 1e-9


def parse_selected_indices(value: object) -> list[int]:
    """Accept the two historical encodings and return one validated list."""
    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = [int(item) for item in value]
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("selected_indices is empty")
        if text.startswith("["):
            parsed = [int(item) for item in json.loads(text)]
        else:
            parsed = [int(item) for item in text.split(";") if item.strip()]
    if len(parsed) != len(set(parsed)):
        raise ValueError("selected_indices contains duplicates")
    return parsed


def _optional_float(row: pd.Series, name: str) -> float | None:
    value = row.get(name)
    return None if value is None or pd.isna(value) else float(value)


def _state_from_row_or_disk(root: Path, row: pd.Series) -> dict[str, Any] | None:
    inline = row.get("state_json")
    if inline is not None and not pd.isna(inline) and str(inline).strip():
        return json.loads(str(inline))

    target = str(row["target"])
    seed = int(row["seed"])
    cycle = int(row["cycle"])
    target_dir = root / "results" / "codex_agent_dev" / target
    candidates = [
        target_dir / f"view_seed{seed}_cycle{cycle}.json",
        target_dir / "states" / f"seed_{seed}_cycle_{cycle}.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())
    return None


def _initial_indices(root: Path, target: str, seed: int, candidate_count: int) -> np.ndarray:
    target_dir = root / "results" / "codex_agent_dev" / target
    candidates = [
        target_dir / f"session_seed{seed}.json",
        target_dir / f"session_seed_{seed}.json",
    ]
    generated = np.sort(
        np.random.default_rng(seed).choice(candidate_count, size=60, replace=False)
    )
    for path in candidates:
        if path.exists():
            saved = np.asarray(json.loads(path.read_text())["observed_indices"][:60], dtype=int)
            if not np.array_equal(saved, generated):
                raise ValueError(f"Initial-set mismatch for {target} seed {seed}")
            return saved
    return generated


def _nested_float(state: dict[str, Any] | None, *keys: str) -> float | None:
    value: Any = state
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value)


def reconstruct_outcomes(root: Path, decisions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "target",
        "policy",
        "model",
        "seed",
        "cycle",
        "expected_outcome",
        "rationale",
        "selected_indices",
    }
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(f"Decision table is missing columns: {sorted(missing)}")
    if decisions.duplicated(KEYS).any():
        raise ValueError("Decision table contains duplicate target/seed/cycle keys")

    ordered = decisions.sort_values(KEYS).reset_index(drop=True)
    states = {
        tuple(row[key] for key in KEYS): _state_from_row_or_disk(root, row)
        for _, row in ordered.iterrows()
    }
    reconstructed: list[dict[str, Any]] = []
    stored_checks = 0
    state_rows = 0

    for (target, seed), campaign in ordered.groupby(["target", "seed"], sort=True):
        campaign = campaign.sort_values("cycle")
        if campaign["cycle"].astype(int).tolist() != list(range(1, 11)):
            raise ValueError(f"Incomplete cycle sequence for {target} seed {seed}")

        frame = load_target(root, str(target))
        labels = frame["affinity"].to_numpy(dtype=float)
        scaffold_ids = scaffolds(frame)
        observed = _initial_indices(root, str(target), int(seed), len(frame)).tolist()

        for _, row in campaign.iterrows():
            cycle = int(row["cycle"])
            chosen = parse_selected_indices(row["selected_indices"])
            if len(chosen) != 30:
                raise ValueError(f"Expected 30 selected compounds for {target} seed {seed} cycle {cycle}")
            if set(chosen).intersection(observed):
                raise ValueError(f"Repeated compound for {target} seed {seed} cycle {cycle}")
            if min(chosen) < 0 or max(chosen) >= len(frame):
                raise ValueError(f"Out-of-range compound for {target} seed {seed} cycle {cycle}")

            selected = np.asarray(chosen, dtype=int)
            before = np.asarray(observed, dtype=int)
            selected_y = labels[selected]
            before_y = labels[before]
            before_scaffolds = set(scaffold_ids[before].tolist())
            new_scaffolds = len(set(scaffold_ids[selected].tolist()) - before_scaffolds)
            observed.extend(chosen)
            after = np.asarray(observed, dtype=int)
            after_y = labels[after]

            pre_state = states.get((target, seed, cycle))
            post_state = states.get((target, seed, cycle + 1)) if cycle < 10 else None
            state_rows += int(pre_state is not None)
            best_improvement = float(after_y.max() - before_y.max())

            checks = {
                "improvement": best_improvement,
                "outcome_best_improvement": best_improvement,
                "observed_best_before": float(before_y.max()),
                "observed_best_after": float(after_y.max()),
                "outcome_batch_affinity_mean": float(selected_y.mean()),
                "outcome_batch_affinity_max": float(selected_y.max()),
                "outcome_new_scaffolds_in_batch": float(new_scaffolds),
            }
            for name, expected in checks.items():
                stored = _optional_float(row, name)
                if stored is not None:
                    stored_checks += 1
                    if not np.isclose(stored, expected, atol=TOLERANCE, rtol=0):
                        raise ValueError(
                            f"Stored {name} mismatch for {target} seed {seed} cycle {cycle}: "
                            f"{stored} != {expected}"
                        )

            reconstructed.append(
                {
                    "target": target,
                    "policy": row["policy"],
                    "model": row["model"],
                    "seed": int(seed),
                    "cycle": cycle,
                    "expected_outcome": row["expected_outcome"],
                    "rationale": row["rationale"],
                    "n_exploit": int(row["n_exploit"]),
                    "n_uncertain": int(row["n_uncertain"]),
                    "n_diverse": int(row["n_diverse"]),
                    "selected_indices": ";".join(map(str, chosen)),
                    "selected_affinities_json": json.dumps(selected_y.tolist()),
                    "batch_affinity_mean": float(selected_y.mean()),
                    "batch_affinity_max": float(selected_y.max()),
                    "incumbent_improvement": best_improvement,
                    "incumbent_improved": bool(best_improvement > TOLERANCE),
                    "new_scaffolds_in_batch": int(new_scaffolds),
                    "cumulative_affinity_mean_after": float(after_y.mean()),
                    "observed_unique_scaffolds_before": int(len(before_scaffolds)),
                    "observed_unique_scaffolds_after": int(len(set(scaffold_ids[after].tolist()))),
                    "pre_gp_predicted_mean_max": _nested_float(
                        pre_state, "gp_predicted_mean", "max"
                    ),
                    "pre_gp_predicted_std_median": _nested_float(
                        pre_state, "gp_predicted_std", "median"
                    ),
                    "pre_gp_predicted_std_max": _nested_float(
                        pre_state, "gp_predicted_std", "max"
                    ),
                    "post_gp_predicted_mean_max": _nested_float(
                        post_state, "gp_predicted_mean", "max"
                    ),
                    "post_gp_predicted_std_median": _nested_float(
                        post_state, "gp_predicted_std", "median"
                    ),
                    "post_gp_predicted_std_max": _nested_float(
                        post_state, "gp_predicted_std", "max"
                    ),
                    "pre_state_available": bool(pre_state is not None),
                    "post_state_available": bool(post_state is not None),
                }
            )

        if len(set(observed)) != 360:
            raise ValueError(f"Final unique count is not 360 for {target} seed {seed}")

    outcome_frame = pd.DataFrame(reconstructed).sort_values(KEYS).reset_index(drop=True)
    validation = {
        "status": "PASS",
        "decision_rows": int(len(ordered)),
        "outcome_rows": int(len(outcome_frame)),
        "campaigns": int(outcome_frame.groupby(["target", "seed"]).ngroups),
        "rows_with_pre_state": int(state_rows),
        "rows_with_post_state": int(outcome_frame["post_state_available"].sum()),
        "stored_values_cross_checked": int(stored_checks),
        "duplicate_keys": int(outcome_frame.duplicated(KEYS).sum()),
        "wrong_batch_sizes": 0,
        "wrong_final_unique_counts": 0,
    }
    return outcome_frame, validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/calibration_v1.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text())
    input_path = root / config["legacy"]["input_decisions"]
    output_path = root / config["legacy"]["outcomes_csv"]
    validation_path = root / config["legacy"]["backfill_validation_json"]

    outcomes, validation = reconstruct_outcomes(root, pd.read_csv(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(output_path, index=False)
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
