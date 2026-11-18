#!/usr/bin/env python3
"""Stepwise, leakage-controlled Codex controller runner for the Mpro dev gate.

The public stdout/state summary contains only observed assay outcomes, GP prediction
summaries, scaffold counts, and prior revealed-batch outcomes. Full ground-truth
evaluation metrics are written during execution but are not printed until the final
cycle, so they cannot influence controller decisions.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agentic_al.agent import (
    BatchAllocation,
    select_portfolio_batch,
    summarize_agent_state,
)
from agentic_al.campaign import (
    CampaignConfig,
    ExactTanimotoGP,
    HiddenAssay,
    _metrics,
    featurize,
    load_target,
    scaffolds,
)


TARGET = "Mpro"
CONTROLLER_MODEL = "gpt-5.6-sol"


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def _seed_dir(seed: int) -> Path:
    return OUTPUT / f"seed_{seed}"


def _load_resources(config: CampaignConfig):
    frame = load_target(ROOT, TARGET)
    features = featurize(
        frame,
        OUTPUT / "feature_cache" / f"{TARGET}_r{config.radius}_{config.n_bits}.npz",
        config.radius,
        config.n_bits,
    )
    scaffold_ids = scaffolds(frame)
    labels = frame["affinity"].to_numpy(dtype=np.float64)
    return frame, features, scaffold_ids, labels


def _public_summary(
    config: CampaignConfig,
    internal: dict,
    features: np.ndarray,
    scaffold_ids: np.ndarray,
) -> dict:
    observed = np.asarray(internal["observed_indices"], dtype=int)
    observed_y = np.asarray(internal["observed_affinities"], dtype=float)
    pool = np.setdiff1d(np.arange(len(features)), observed, assume_unique=False)
    gp = ExactTanimotoGP(config.noise, config.jitter).fit(features[observed], observed_y)
    mean, std = gp.predict(features[pool])
    return summarize_agent_state(
        target=TARGET,
        cycle=int(internal["cycles_completed"]) + 1,
        cycles=config.cycles,
        batch_size=config.batch_size,
        observed_affinities=observed_y,
        predicted_mean=mean,
        predicted_std=std,
        observed_scaffolds=scaffold_ids[observed],
        pool_scaffolds=scaffold_ids[pool],
        history=internal["history"],
    )


def start(seed: int) -> None:
    run_dir = _seed_dir(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "internal_state.json"
    if state_path.exists():
        raise SystemExit(f"Seed {seed} is already initialized; use step or inspect.")

    config = CampaignConfig(target=TARGET, policy="codex-agent", seed=seed)
    frame, features, scaffold_ids, labels = _load_resources(config)
    rng = np.random.default_rng(seed)
    initial = np.sort(rng.choice(len(frame), size=config.initial_size, replace=False))
    assay = HiddenAssay(labels)
    observed_y = assay.reveal(initial)
    internal = {
        "config": asdict(config),
        "cycles_completed": 0,
        "observed_indices": initial.tolist(),
        "observed_affinities": observed_y.tolist(),
        "history": [],
    }
    _write_json(state_path, internal)

    initial_metrics = _metrics(
        labels,
        frame["top_2p"].to_numpy(dtype=bool),
        frame["top_5p"].to_numpy(dtype=bool),
        scaffold_ids,
        initial.tolist(),
    )
    pd.DataFrame(
        [{**asdict(config), "cycle": 0, "method": "initial-random", "measured": len(initial), **initial_metrics}]
    ).to_csv(run_dir / "trajectory.private_until_complete.csv", index=False)

    metadata = {
        "status": "RUNNING",
        "target": TARGET,
        "seed": seed,
        "controller": "Codex strategy controller",
        "controller_model": CONTROLLER_MODEL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "leakage_contract": (
            "Controller receives observed labels, GP mean/std quantiles, scaffold counts, "
            "and prior revealed-batch outcomes only; hidden pool labels and interim global "
            "evaluation metrics are withheld."
        ),
        "config": asdict(config),
    }
    _write_json(run_dir / "metadata.json", metadata)
    summary = _public_summary(config, internal, features, scaffold_ids)
    _write_json(run_dir / "next_state_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def step(seed: int, allocation: BatchAllocation) -> None:
    run_dir = _seed_dir(seed)
    state_path = run_dir / "internal_state.json"
    if not state_path.exists():
        raise SystemExit(f"Seed {seed} is not initialized; run start first.")
    internal = json.loads(state_path.read_text())
    config = CampaignConfig(**internal["config"])
    allocation.validate(config.batch_size)
    cycle = int(internal["cycles_completed"]) + 1
    if cycle > config.cycles:
        raise SystemExit(f"Seed {seed} is already complete.")

    frame, features, scaffold_ids, labels = _load_resources(config)
    observed = np.asarray(internal["observed_indices"], dtype=int)
    observed_y = np.asarray(internal["observed_affinities"], dtype=float)

    # Reconstitute the assay boundary without exposing labels to stdout.
    assay = HiddenAssay(labels)
    reconstructed = assay.reveal(observed)
    if not np.allclose(reconstructed, observed_y):
        raise RuntimeError("Persisted observed labels do not match the assay record")

    pool = np.setdiff1d(np.arange(len(frame)), observed, assume_unique=False)
    gp = ExactTanimotoGP(config.noise, config.jitter).fit(features[observed], observed_y)
    mean, std = gp.predict(features[pool])
    chosen = select_portfolio_batch(allocation, pool, mean, std, features, observed)
    selected_mask = np.isin(pool, chosen)
    revealed = assay.reveal(chosen)

    previous_best = float(observed_y.max())
    batch_best = float(revealed.max())
    batch_mean = float(revealed.mean())
    new_best = max(previous_best, batch_best)
    outcome = {
        "cycle": cycle,
        "allocation": {
            "n_exploit": allocation.n_exploit,
            "n_uncertain": allocation.n_uncertain,
            "n_diverse": allocation.n_diverse,
        },
        "revealed_batch_best": batch_best,
        "revealed_batch_mean": batch_mean,
        "incumbent_improvement": float(new_best - previous_best),
        "observed_unique_scaffolds_after": int(
            len(set(scaffold_ids[np.concatenate([observed, chosen])].tolist()))
        ),
    }
    internal["observed_indices"].extend(chosen.tolist())
    internal["observed_affinities"].extend(revealed.tolist())
    internal["cycles_completed"] = cycle
    internal["history"].append(outcome)
    _write_json(state_path, internal)

    decision = {
        "target": TARGET,
        "seed": seed,
        "cycle": cycle,
        **allocation.to_dict(),
        "selected_indices": ";".join(map(str, chosen.tolist())),
        "observed_count_before": int(len(observed)),
        "remaining_budget_before": int((config.cycles - cycle + 1) * config.batch_size),
        "mean_score_selected": float(mean[selected_mask].mean()),
        "mean_uncertainty_selected": float(std[selected_mask].mean()),
        **{key: value for key, value in outcome.items() if key not in {"cycle", "allocation"}},
    }
    decisions_path = run_dir / "decisions.csv"
    pd.DataFrame([decision]).to_csv(
        decisions_path,
        mode="a",
        header=not decisions_path.exists(),
        index=False,
    )

    all_observed = internal["observed_indices"]
    metrics = _metrics(
        labels,
        frame["top_2p"].to_numpy(dtype=bool),
        frame["top_5p"].to_numpy(dtype=bool),
        scaffold_ids,
        all_observed,
    )
    trajectory_path = run_dir / "trajectory.private_until_complete.csv"
    pd.DataFrame(
        [{**asdict(config), "cycle": cycle, "method": "codex-agent", "measured": len(all_observed), **metrics}]
    ).to_csv(trajectory_path, mode="a", header=False, index=False)

    if cycle < config.cycles:
        summary = _public_summary(config, internal, features, scaffold_ids)
        _write_json(run_dir / "next_state_summary.json", summary)
        print(json.dumps({"revealed_outcome": outcome, "next_state": summary}, indent=2, sort_keys=True))
        return

    final_path = run_dir / "trajectory.csv"
    trajectory_path.replace(final_path)
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update({
        "status": "COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "cycles_completed": cycle,
        "measured": len(all_observed),
        "valid_action_rate": 1.0,
    })
    _write_json(metadata_path, metadata)
    (run_dir / "next_state_summary.json").unlink(missing_ok=True)
    print(json.dumps({"revealed_outcome": outcome, "status": "COMPLETE", "final_metrics": metrics}, indent=2, sort_keys=True))


def inspect(seed: int) -> None:
    summary_path = _seed_dir(seed) / "next_state_summary.json"
    if summary_path.exists():
        print(summary_path.read_text(), end="")
        return
    metadata_path = _seed_dir(seed) / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"Seed {seed} has not been initialized.")
    metadata = json.loads(metadata_path.read_text())
    print(json.dumps({"status": metadata["status"], "cycles_completed": metadata.get("cycles_completed")}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("start", "inspect"):
        child = subparsers.add_parser(command)
        child.add_argument("--seed", type=int, required=True)
    step_parser = subparsers.add_parser("step")
    step_parser.add_argument("--seed", type=int, required=True)
    step_parser.add_argument("--exploit", type=int, required=True)
    step_parser.add_argument("--uncertain", type=int, required=True)
    step_parser.add_argument("--diverse", type=int, required=True)
    step_parser.add_argument("--rationale", required=True)
    step_parser.add_argument("--expected", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "start":
        start(args.seed)
    elif args.command == "inspect":
        inspect(args.seed)
    else:
        step(
            args.seed,
            BatchAllocation(
                n_exploit=args.exploit,
                n_uncertain=args.uncertain,
                n_diverse=args.diverse,
                rationale=args.rationale,
                expected_outcome=args.expected,
            ),
        )


if __name__ == "__main__":
    main()
