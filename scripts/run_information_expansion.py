#!/usr/bin/env python3
"""Checkpointed information-study expansion driven only by sanitized agent views."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.codex_agent_session import (
    _read,
    _write,
    agent_view,
    apply_action,
    initialize,
)


VARIANTS = (
    "full_reference",
    "minus_uncertainty",
    "minus_diversity",
    "minus_memory",
    "shuffled_evidence",
)


def controller_action(view: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen controller judgment using only its serialized public view."""
    cycle = int(view["cycle"])
    best = float(view["observed_affinity"]["best"])
    predicted = view["gp_predicted_mean"]
    median = float(predicted["median"])
    q75 = float(predicted["q75"])

    n_exploit = 10 + (2 if cycle >= 4 else 0) + (2 if cycle >= 8 else 0)
    history = view.get("history", [])
    if history and float(history[-1].get("improvement", 0.0)) > 0:
        n_exploit += 2

    uncertainty = view.get("gp_predicted_std")
    if uncertainty is None:
        n_uncertain = 8
        uncertainty_reason = "uncertainty statistics are withheld"
    else:
        uncertainty_span = float(uncertainty["q75"]) - float(uncertainty["q25"])
        n_uncertain = 11 if uncertainty_span >= 0.2 else 9
        uncertainty_reason = f"the uncertainty IQR is {uncertainty_span:.3f}"

    n_exploit = min(n_exploit, 19)
    n_diverse = 30 - n_exploit - n_uncertain
    if n_diverse < 4:
        transfer = 4 - n_diverse
        n_exploit -= transfer
        n_diverse = 4

    space = view.get("chemical_space")
    if space is None:
        diversity_reason = "chemical-space counts are withheld"
        expected = f"The next-batch affinity mean will be at least {median:.3f}."
    else:
        pool_scaffolds = int(space["pool_unique_scaffolds"])
        diversity_reason = f"the eligible pool contains {pool_scaffolds} unique scaffolds"
        expected = (
            f"The next-batch affinity mean will be at least {median:.3f} and "
            f"at least {max(1, n_diverse // 3)} new scaffolds will be observed."
        )

    rationale = (
        f"Cycle {cycle}: allocate {n_exploit} to exploitation because the GP median/q75 "
        f"are {median:.3f}/{q75:.3f} versus observed best {best:.3f}; allocate "
        f"{n_uncertain} to uncertainty because {uncertainty_reason}; retain {n_diverse} "
        f"for diversity because {diversity_reason}."
    )
    return {
        "n_exploit": n_exploit,
        "n_uncertain": n_uncertain,
        "n_diverse": n_diverse,
        "rationale": rationale,
        "expected_outcome": expected,
    }


def run_one(args: tuple[str, str, int, str, str]) -> dict[str, Any]:
    root_value, target, seed, variant, config_value = args
    root = Path(root_value)
    config_path = Path(config_value)
    directory = root / "results" / "information_ablations" / "full" / target / variant
    session_path = directory / f"session_seed{seed}.json"
    if session_path.exists():
        session = _read(session_path)
        if (session["target"], session["seed"], session["variant"]) != (target, seed, variant):
            raise ValueError(f"Collision at {session_path}")
        view = agent_view(root, session)
    else:
        view = initialize(
            root,
            session_path,
            target,
            seed,
            config_path=config_path,
            variant=variant,
        )

    while view.get("status") != "complete":
        cycle = int(view["cycle"])
        view_path = directory / f"view_seed{seed}_cycle{cycle}.json"
        if view_path.exists() and _read(view_path) != view:
            raise ValueError(f"View collision at {view_path}")
        _write(view_path, view)
        action_path = directory / f"action_seed{seed}_cycle{cycle}.json"
        action = controller_action(view)
        if action_path.exists() and _read(action_path) != action:
            raise ValueError(f"Action collision at {action_path}")
        _write(action_path, action)
        view = apply_action(root, session_path, action_path)

    export_path = directory / f"export_seed{seed}.json"
    session = _read(session_path)
    _write(export_path, session)
    endpoint = session["trajectory"][-1]
    return {
        "target": target,
        "variant": variant,
        "seed": seed,
        "decisions": len(session["history"]),
        "measured": endpoint["measured"],
        **{key: endpoint[key] for key in (
            "best_affinity",
            "simple_regret",
            "top_1p_recall",
            "top_2p_recall",
            "top_5p_recall",
            "unique_top_5p_scaffolds",
        )},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["TYK2", "USP7"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "agent_information_ablations.json",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    jobs = [
        (str(root), target, seed, variant, str(args.config))
        for target in args.targets
        for variant in VARIANTS
        for seed in args.seeds
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"complete {row['target']} {row['variant']} seed={row['seed']}",
                flush=True,
            )
    report_scope = "_" + "_".join(args.targets)
    report_dir = root / "results" / "information_ablations" / "full" / report_scope
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "endpoint_metrics.csv"
    frame = pd.DataFrame(rows).sort_values(["target", "variant", "seed"])
    frame.to_csv(output, index=False)
    summary = (
        frame.groupby(["target", "variant"], as_index=False)
        .agg(
            campaigns=("seed", "size"),
            top_2p_recall_mean=("top_2p_recall", "mean"),
            top_2p_recall_std=("top_2p_recall", "std"),
            top_1p_recall_mean=("top_1p_recall", "mean"),
            top_5p_recall_mean=("top_5p_recall", "mean"),
            simple_regret_mean=("simple_regret", "mean"),
            unique_top_5p_scaffolds_mean=("unique_top_5p_scaffolds", "mean"),
        )
    )
    summary.to_csv(output.with_name("summary_by_target_variant.csv"), index=False)


if __name__ == "__main__":
    main()
