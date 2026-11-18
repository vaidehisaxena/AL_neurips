#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_al.agent import SYSTEM_PROMPT
from agentic_al.campaign import (
    _index_digest,
    _mask_digest,
    deterministic_availability_mask,
    load_target,
)
from agentic_al.explanations import canonical_digest

OUT = ROOT / "results" / "availability_constrained" / "agent_dev_gate"
CONFIG_PATH = ROOT / "configs" / "agent_availability_constrained.json"


def read(path: Path):
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_seed(seed: int, config: dict, frame) -> dict:
    session = read(OUT / f"session_seed{seed}.json")
    exported = read(OUT / f"export_seed{seed}.json")
    require(session == exported, f"seed {seed}: export differs from session")
    require(session["target"] == "D2R", f"seed {seed}: wrong target")
    require(session["seed"] == seed, f"seed {seed}: wrong seed")
    require(session["variant"] == "full_reference", f"seed {seed}: wrong variant")
    require(session["cycle"] == 10, f"seed {seed}: wrong terminal cycle")
    require(len(session["history"]) == 10, f"seed {seed}: decision count")
    require(len(session["predecision_views"]) == 10, f"seed {seed}: view count")
    require(len(session["trajectory"]) == 11, f"seed {seed}: trajectory count")

    config_digest = canonical_digest(config)
    prompt_digest = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    initial = np.asarray(session["initial_indices"], dtype=int)
    initial_json_digest = canonical_digest(initial.tolist())
    initial_set_digest = _index_digest(initial)
    base_mask = deterministic_availability_mask(
        frame["SMILES"].astype(str),
        float(config["availability"]["fraction"]),
        str(config["availability"]["salt"]),
    )
    mask_digest = _mask_digest(base_mask)

    require(session["config_sha256"] == config_digest, f"seed {seed}: config digest")
    require(session["system_prompt_sha256"] == prompt_digest, f"seed {seed}: prompt digest")
    require(session["initial_indices_sha256"] == initial_json_digest, f"seed {seed}: initial JSON digest")
    require(session["initial_set_digest"] == initial_set_digest, f"seed {seed}: initial set digest")
    require(session["availability_mask_digest"] == mask_digest, f"seed {seed}: mask digest")
    require(len(initial) == 60 and len(set(initial.tolist())) == 60, f"seed {seed}: initial uniqueness")

    selected_all: list[int] = []
    for cycle, (decision, view_record) in enumerate(
        zip(session["history"], session["predecision_views"]), start=1
    ):
        action = read(OUT / f"action_seed{seed}_cycle{cycle}.json")
        saved_view = read(OUT / f"view_seed{seed}_cycle{cycle}.json")
        counts = [action[k] for k in ("n_exploit", "n_uncertain", "n_diverse")]
        require(all(type(value) is int and value >= 0 for value in counts), f"seed {seed} cycle {cycle}: allocation type/range")
        require(sum(counts) == 30, f"seed {seed} cycle {cycle}: allocation sum")
        require(bool(action["rationale"].strip()), f"seed {seed} cycle {cycle}: rationale")
        require(bool(action["expected_outcome"].strip()), f"seed {seed} cycle {cycle}: expected outcome")
        for key in ("n_exploit", "n_uncertain", "n_diverse", "rationale", "expected_outcome"):
            require(decision[key] == action[key], f"seed {seed} cycle {cycle}: action/history mismatch {key}")

        chosen = [int(value) for value in decision["selected_indices"]]
        require(len(chosen) == 30 and len(set(chosen)) == 30, f"seed {seed} cycle {cycle}: batch uniqueness")
        require(all(bool(base_mask[index]) for index in chosen), f"seed {seed} cycle {cycle}: unavailable selection")
        selected_all.extend(chosen)

        view_digest = canonical_digest(saved_view)
        require(view_record["cycle"] == cycle, f"seed {seed} cycle {cycle}: view cycle")
        require(view_record["view"] == saved_view, f"seed {seed} cycle {cycle}: saved/recorded view mismatch")
        require(view_record["view_sha256"] == view_digest, f"seed {seed} cycle {cycle}: recorded view digest")
        require(decision["predecision_view_sha256"] == view_digest, f"seed {seed} cycle {cycle}: decision view digest")

        evidence = saved_view["explanatory_evidence"]
        evidence_digest = canonical_digest({k: v for k, v in evidence.items() if k != "evidence_sha256"})
        require(evidence["evidence_sha256"] == evidence_digest, f"seed {seed} cycle {cycle}: evidence self-digest")
        require(view_record["evidence_sha256"] == evidence_digest, f"seed {seed} cycle {cycle}: recorded evidence digest")
        require(decision["evidence_sha256"] == evidence_digest, f"seed {seed} cycle {cycle}: decision evidence digest")

        for holder_name, holder in (("decision", decision), ("view", view_record)):
            require(holder["availability_mask_digest"] == mask_digest, f"seed {seed} cycle {cycle}: {holder_name} mask digest")
            require(holder["initial_set_digest"] == initial_set_digest, f"seed {seed} cycle {cycle}: {holder_name} initial digest")
        require(decision["config_sha256"] == config_digest, f"seed {seed} cycle {cycle}: decision config digest")
        require(decision["system_prompt_sha256"] == prompt_digest, f"seed {seed} cycle {cycle}: decision prompt digest")
        require(decision["initial_indices_sha256"] == initial_json_digest, f"seed {seed} cycle {cycle}: decision initial JSON digest")

    require(len(selected_all) == 300, f"seed {seed}: selected count")
    require(len(set(selected_all)) == 300, f"seed {seed}: cross-cycle selected uniqueness")
    require(set(initial.tolist()).isdisjoint(selected_all), f"seed {seed}: initial reselection")
    require(session["observed_indices"] == initial.tolist() + selected_all, f"seed {seed}: observed order")
    require(len(session["observed_indices"]) == 360, f"seed {seed}: measured endpoint")
    require([row["cycle"] for row in session["trajectory"]] == list(range(11)), f"seed {seed}: trajectory cycles")
    require([row["measured"] for row in session["trajectory"]] == [60 + 30 * c for c in range(11)], f"seed {seed}: trajectory budget")
    require(all(row["availability_mask_digest"] == mask_digest for row in session["trajectory"]), f"seed {seed}: trajectory mask digest")
    require(all(row["initial_set_digest"] == initial_set_digest for row in session["trajectory"]), f"seed {seed}: trajectory initial digest")

    endpoint = session["trajectory"][-1]
    return {
        "seed": seed,
        "decisions": len(session["history"]),
        "selected_unique": len(set(selected_all)),
        "unavailable_post_initial": sum(not bool(base_mask[index]) for index in selected_all),
        "measured": endpoint["measured"],
        "availability_mask_digest": mask_digest,
        "initial_set_digest": initial_set_digest,
        "config_sha256": config_digest,
        "system_prompt_sha256": prompt_digest,
        "endpoint_metrics": {
            key: endpoint[key]
            for key in (
                "best_affinity",
                "simple_regret",
                "top_1p_recall",
                "top_2p_recall",
                "top_5p_recall",
                "unique_top_5p_scaffolds",
            )
        },
    }


def main() -> None:
    config = read(CONFIG_PATH)
    frame = load_target(ROOT, "D2R")
    report = {
        "status": "pass",
        "protocol": config["protocol"],
        "variant": "full_reference",
        "target": "D2R",
        "seeds": [validate_seed(seed, config, frame) for seed in (0, 1)],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
