#!/usr/bin/env python3
"""Stepwise, file-backed campaign harness for a human/Codex strategy controller."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agentic_al.agent import BatchAllocation, select_portfolio_batch, summarize_agent_state
from agentic_al.campaign import ExactTanimotoGP, _metrics, featurize, load_target, scaffolds


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _assets(root: Path, session: dict):
    target = session["target"]
    frame = load_target(root, target)
    features = featurize(
        frame,
        root / "artifacts" / "feature_cache" / f"{target}_r4_4096.npz",
        radius=4,
        n_bits=4096,
    )
    return frame, features, scaffolds(frame)


def initialize(root: Path, session_path: Path, target: str, seed: int) -> dict:
    frame = load_target(root, target)
    rng = np.random.default_rng(seed)
    initial = np.sort(rng.choice(len(frame), size=60, replace=False))
    labels = frame["affinity"].to_numpy(dtype=float)
    scaffold_ids = scaffolds(frame)
    initial_metrics = _metrics(
        labels,
        frame["top_2p"].to_numpy(bool),
        frame["top_5p"].to_numpy(bool),
        scaffold_ids,
        initial.tolist(),
    )
    session = {
        "protocol": "codex-agent-v1",
        "controller_model": "gpt-5.6-sol",
        "target": target,
        "seed": seed,
        "cycle": 0,
        "cycles": 10,
        "batch_size": 30,
        "observed_indices": initial.tolist(),
        "observed_affinities": labels[initial].tolist(),
        "history": [],
        "trajectory": [{"cycle": 0, "measured": 60, **initial_metrics}],
    }
    _write(session_path, session)
    return agent_view(root, session)


def agent_view(root: Path, session: dict) -> dict:
    if session["cycle"] >= session["cycles"]:
        return {"status": "complete", "target": session["target"], "seed": session["seed"]}
    frame, features, scaffold_ids = _assets(root, session)
    observed = np.asarray(session["observed_indices"], dtype=int)
    pool = np.setdiff1d(np.arange(len(frame)), observed)
    gp = ExactTanimotoGP().fit(features[observed], np.asarray(session["observed_affinities"]))
    mean, std = gp.predict(features[pool])
    summary = summarize_agent_state(
        target=session["target"],
        cycle=session["cycle"] + 1,
        cycles=session["cycles"],
        batch_size=session["batch_size"],
        observed_affinities=np.asarray(session["observed_affinities"]),
        predicted_mean=mean,
        predicted_std=std,
        observed_scaffolds=scaffold_ids[observed],
        pool_scaffolds=scaffold_ids[pool],
        history=session["history"],
    )
    summary["instruction"] = (
        "Choose non-negative integer n_exploit, n_uncertain, n_diverse summing to 30, "
        "plus a rationale and falsifiable expected_outcome."
    )
    return summary


def apply_action(root: Path, session_path: Path, action_path: Path) -> dict:
    session = _read(session_path)
    if session["cycle"] >= session["cycles"]:
        raise ValueError("Campaign is already complete")
    raw = _read(action_path)
    action = BatchAllocation(**raw).validate(session["batch_size"])
    frame, features, scaffold_ids = _assets(root, session)
    labels = frame["affinity"].to_numpy(dtype=float)
    observed = np.asarray(session["observed_indices"], dtype=int)
    pool = np.setdiff1d(np.arange(len(frame)), observed)
    gp = ExactTanimotoGP().fit(features[observed], np.asarray(session["observed_affinities"]))
    mean, std = gp.predict(features[pool])
    chosen = select_portfolio_batch(action, pool, mean, std, features, observed)
    before_best = max(session["observed_affinities"])
    revealed = labels[chosen]
    session["observed_indices"].extend(chosen.tolist())
    session["observed_affinities"].extend(revealed.tolist())
    session["cycle"] += 1
    metrics = _metrics(
        labels,
        frame["top_2p"].to_numpy(bool),
        frame["top_5p"].to_numpy(bool),
        scaffold_ids,
        session["observed_indices"],
    )
    decision = {
        "cycle": session["cycle"],
        **action.to_dict(),
        "selected_indices": chosen.tolist(),
        "selected_predicted_mean": float(mean[np.isin(pool, chosen)].mean()),
        "selected_predicted_std": float(std[np.isin(pool, chosen)].mean()),
        "observed_best_before": float(before_best),
        "observed_best_after": float(metrics["best_affinity"]),
        "improvement": float(metrics["best_affinity"] - before_best),
    }
    session["history"].append(decision)
    session["trajectory"].append(
        {"cycle": session["cycle"], "measured": len(session["observed_indices"]), **metrics}
    )
    _write(session_path, session)
    return agent_view(root, session)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "view", "act", "export"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--target", choices=("D2R", "Mpro", "TYK2", "USP7"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--action", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        if args.target is None or args.seed is None:
            parser.error("init requires --target and --seed")
        result = initialize(args.root, args.session, args.target, args.seed)
    elif args.command == "view":
        result = agent_view(args.root, _read(args.session))
    elif args.command == "act":
        if args.action is None:
            parser.error("act requires --action")
        result = apply_action(args.root, args.session, args.action)
    else:
        session = _read(args.session)
        if session["cycle"] != session["cycles"]:
            raise ValueError("Cannot export an incomplete campaign")
        result = session
    if args.output:
        _write(args.output, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
