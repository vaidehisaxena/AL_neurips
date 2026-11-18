#!/usr/bin/env python3
"""Target-local interactive harness for a Codex-controlled TYK2 campaign."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_al.agent import BatchAllocation, select_portfolio_batch, summarize_agent_state
from agentic_al.campaign import (
    CampaignConfig,
    ExactTanimotoGP,
    HiddenAssay,
    _metrics,
    featurize,
    load_target,
    scaffolds,
)


def git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = CampaignConfig(target="TYK2", policy="codex-agent", seed=args.seed)
    frame = load_target(root, config.target)
    cache = root / "artifacts" / "feature_cache" / f"{config.target}_r{config.radius}_{config.n_bits}.npz"
    features = featurize(frame, cache, config.radius, config.n_bits)
    scaffold_ids = scaffolds(frame)

    # Complete pool labels stay private to the assay/evaluation environment.
    labels = frame["affinity"].to_numpy(dtype=np.float64)
    top_2p = frame["top_2p"].to_numpy(dtype=bool)
    top_5p = frame["top_5p"].to_numpy(dtype=bool)
    assay = HiddenAssay(labels)
    rng = np.random.default_rng(config.seed)
    initial = np.sort(rng.choice(len(frame), size=config.initial_size, replace=False))
    observed = initial.tolist()
    observed_y = assay.reveal(initial).tolist()
    history: list[dict] = []
    decisions: list[dict] = []
    trajectory: list[dict] = []

    initial_metrics = _metrics(labels, top_2p, top_5p, scaffold_ids, observed)
    trajectory.append(
        {
            **asdict(config),
            "model": "gpt-5.6-sol",
            "cycle": 0,
            "method": "initial-random",
            "measured": len(observed),
            **initial_metrics,
        }
    )

    for cycle in range(1, config.cycles + 1):
        gp = ExactTanimotoGP(config.noise, config.jitter).fit(
            features[observed], np.asarray(observed_y)
        )
        pool = np.setdiff1d(np.arange(len(frame)), np.asarray(observed), assume_unique=False)
        mean, std = gp.predict(features[pool])
        state = summarize_agent_state(
            target=config.target,
            cycle=cycle,
            cycles=config.cycles,
            batch_size=config.batch_size,
            observed_affinities=np.asarray(observed_y),
            predicted_mean=mean,
            predicted_std=std,
            observed_scaffolds=scaffold_ids[np.asarray(observed)],
            pool_scaffolds=scaffold_ids[pool],
            history=history,
        )
        print("STATE_JSON " + json.dumps(state, sort_keys=True), flush=True)
        line = input().strip()
        payload = json.loads(line)
        action = BatchAllocation(**payload).validate(config.batch_size)
        chosen = select_portfolio_batch(
            action,
            pool,
            mean,
            std,
            features,
            np.asarray(observed),
        )
        before_best = float(np.max(observed_y))
        before_scaffolds = set(scaffold_ids[np.asarray(observed)].tolist())
        revealed = assay.reveal(chosen)
        observed.extend(chosen.tolist())
        observed_y.extend(revealed.tolist())
        after_best = float(np.max(observed_y))
        selected_new_scaffolds = len(set(scaffold_ids[chosen].tolist()) - before_scaffolds)
        outcome = {
            "cycle": cycle,
            "allocation": {
                "n_exploit": action.n_exploit,
                "n_uncertain": action.n_uncertain,
                "n_diverse": action.n_diverse,
            },
            "batch_affinity_mean": float(np.mean(revealed)),
            "batch_affinity_max": float(np.max(revealed)),
            "best_improvement": after_best - before_best,
            "new_scaffolds_in_batch": selected_new_scaffolds,
        }
        history.append(outcome)
        decisions.append(
            {
                "target": config.target,
                "policy": config.policy,
                "model": "gpt-5.6-sol",
                "seed": config.seed,
                "cycle": cycle,
                **action.to_dict(),
                "state_json": json.dumps(state, sort_keys=True),
                "selected_indices": ";".join(map(str, chosen.tolist())),
                **{f"outcome_{key}": value for key, value in outcome.items() if key not in {"cycle", "allocation"}},
            }
        )
        metrics = _metrics(labels, top_2p, top_5p, scaffold_ids, observed)
        trajectory.append(
            {
                **asdict(config),
                "model": "gpt-5.6-sol",
                "cycle": cycle,
                "method": "portfolio",
                "measured": len(observed),
                **metrics,
            }
        )
        print("OUTCOME_JSON " + json.dumps(outcome, sort_keys=True), flush=True)

    pd.DataFrame(trajectory).to_csv(output / f"trajectory_seed{config.seed}.csv", index=False)
    pd.DataFrame(decisions).to_csv(output / f"decisions_seed{config.seed}.csv", index=False)
    metadata = {
        "status": "complete",
        "controller": "Codex sub-agent",
        "model": "gpt-5.6-sol",
        "config": asdict(config),
        "code_commit": git_commit(root),
        "python": platform.python_version(),
        "hidden_label_contract": "Only STATE_JSON and prior revealed outcomes were used for decisions.",
        "valid_actions": len(decisions),
        "invalid_actions": 0,
    }
    (output / f"metadata_seed{config.seed}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("COMPLETE " + json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
