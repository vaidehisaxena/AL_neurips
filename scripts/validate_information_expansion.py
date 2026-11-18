#!/usr/bin/env python3
"""Validate checkpointed information-study campaigns and aggregate endpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_al.agent import SYSTEM_PROMPT
from agentic_al.campaign import _index_digest
from agentic_al.explanations import canonical_digest, evidence_shuffle_seed_digest


VARIANTS = (
    "full_reference",
    "minus_uncertainty",
    "minus_diversity",
    "minus_memory",
    "shuffled_evidence",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def repair_checkpoints(root: Path, targets: list[str], seeds: list[int]) -> int:
    """Restore view/action files from the authoritative append-only session records."""
    repaired = 0
    base = root / "results" / "information_ablations" / "full"
    for target in targets:
        for variant in VARIANTS:
            directory = base / target / variant
            for seed in seeds:
                session = _read(directory / f"session_seed{seed}.json")
                for decision, record in zip(
                    session.get("history", []), session.get("predecision_views", [])
                ):
                    cycle = int(decision["cycle"])
                    action = {
                        key: decision[key]
                        for key in (
                            "n_exploit",
                            "n_uncertain",
                            "n_diverse",
                            "rationale",
                            "expected_outcome",
                        )
                    }
                    paths = (
                        (directory / f"action_seed{seed}_cycle{cycle}.json", action),
                        (directory / f"view_seed{seed}_cycle{cycle}.json", record["view"]),
                    )
                    for path, value in paths:
                        if not path.exists() or _read(path) != value:
                            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
                            repaired += 1
    return repaired


def validate(root: Path, targets: list[str], seeds: list[int]) -> tuple[pd.DataFrame, dict]:
    config = _read(root / "configs" / "agent_information_ablations.json")
    config_digest = canonical_digest(config)
    prompt_digest = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    base = root / "results" / "information_ablations" / "full"
    endpoints: list[dict[str, Any]] = []
    initial_by_pair: dict[tuple[str, int], tuple[int, ...]] = {}
    checks = {
        "campaigns": 0,
        "actions": 0,
        "unique_selection_campaigns": 0,
        "budget_valid_campaigns": 0,
        "config_digest_matches": 0,
        "prompt_digest_matches": 0,
        "initial_digest_matches": 0,
        "view_digest_matches": 0,
        "evidence_digest_matches": 0,
        "shuffle_digest_matches": 0,
        "forbidden_view_key_violations": 0,
        "paired_initial_violations": 0,
        "anomalies": [],
    }

    def fail(message: str) -> None:
        checks["anomalies"].append(message)

    for target in targets:
        for variant in VARIANTS:
            directory = base / target / variant
            for seed in seeds:
                label = f"{target}/{variant}/seed{seed}"
                session_path = directory / f"session_seed{seed}.json"
                export_path = directory / f"export_seed{seed}.json"
                if not session_path.is_file() or not export_path.is_file():
                    fail(f"{label}: missing session or export")
                    continue
                session = _read(session_path)
                export = _read(export_path)
                checks["campaigns"] += 1
                if session != export:
                    fail(f"{label}: export differs from session")
                if (session.get("target"), session.get("variant"), session.get("seed")) != (
                    target,
                    variant,
                    seed,
                ):
                    fail(f"{label}: identity/path mismatch")
                if session.get("config_sha256") == config_digest:
                    checks["config_digest_matches"] += 1
                else:
                    fail(f"{label}: config digest mismatch")
                if session.get("system_prompt_sha256") == prompt_digest:
                    checks["prompt_digest_matches"] += 1
                else:
                    fail(f"{label}: prompt digest mismatch")

                initial = tuple(map(int, session.get("initial_indices", []) or session["observed_indices"][:60]))
                pair = (target, seed)
                if pair in initial_by_pair and initial_by_pair[pair] != initial:
                    checks["paired_initial_violations"] += 1
                    fail(f"{label}: initial set differs across variants")
                initial_by_pair.setdefault(pair, initial)
                if (
                    session.get("initial_indices_sha256") == canonical_digest(list(initial))
                    and session.get("initial_set_digest") == _index_digest(initial)
                ):
                    checks["initial_digest_matches"] += 1
                else:
                    fail(f"{label}: initial digest mismatch")

                history = session.get("history", [])
                checks["actions"] += len(history)
                if session.get("cycle") != 10 or len(history) != 10:
                    fail(f"{label}: expected cycle/history length 10")
                observed = list(map(int, session.get("observed_indices", [])))
                if len(observed) == len(set(observed)) == 360:
                    checks["unique_selection_campaigns"] += 1
                else:
                    fail(f"{label}: observed indices are duplicate or not length 360")
                trajectory = session.get("trajectory", [])
                if [row.get("measured") for row in trajectory] == list(range(60, 361, 30)):
                    checks["budget_valid_campaigns"] += 1
                else:
                    fail(f"{label}: measured trajectory is not 60..360 by 30")

                for cycle, decision in enumerate(history, start=1):
                    action_path = directory / f"action_seed{seed}_cycle{cycle}.json"
                    view_path = directory / f"view_seed{seed}_cycle{cycle}.json"
                    if not action_path.is_file() or not view_path.is_file():
                        fail(f"{label}/cycle{cycle}: missing action or view")
                        continue
                    action = _read(action_path)
                    view = _read(view_path)
                    expected_action = {
                        key: decision[key]
                        for key in (
                            "n_exploit",
                            "n_uncertain",
                            "n_diverse",
                            "rationale",
                            "expected_outcome",
                        )
                    }
                    if action != expected_action or sum(
                        action[key] for key in ("n_exploit", "n_uncertain", "n_diverse")
                    ) != 30:
                        fail(f"{label}/cycle{cycle}: action mismatch or bad allocation")
                    if canonical_digest(view) == decision.get("predecision_view_sha256"):
                        checks["view_digest_matches"] += 1
                    else:
                        fail(f"{label}/cycle{cycle}: view digest mismatch")
                    evidence = view.get("explanatory_evidence", {})
                    if evidence.get("evidence_sha256") == decision.get("evidence_sha256"):
                        checks["evidence_digest_matches"] += 1
                    else:
                        fail(f"{label}/cycle{cycle}: evidence digest mismatch")
                    expected_shuffle = None
                    if variant == "shuffled_evidence":
                        expected_shuffle = evidence_shuffle_seed_digest(
                            target=target,
                            seed=seed,
                            cycle=cycle,
                            protocol_version=config["evidence"]["shuffle_protocol_version"],
                        )
                    if decision.get("evidence_shuffle_seed_sha256") == expected_shuffle:
                        checks["shuffle_digest_matches"] += 1
                    else:
                        fail(f"{label}/cycle{cycle}: shuffle digest mismatch")
                    forbidden = {"hidden_labels", "pool_affinity", "top_2p", "top_5p"}
                    if forbidden.intersection(view):
                        checks["forbidden_view_key_violations"] += 1
                        fail(f"{label}/cycle{cycle}: forbidden view key")

                endpoint = trajectory[-1]
                endpoints.append(
                    {
                        "target": target,
                        "variant": variant,
                        "seed": seed,
                        **{
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
                )
    checks["status"] = "PASS" if not checks["anomalies"] else "FAIL"
    return pd.DataFrame(endpoints), checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--report-scope", required=True)
    parser.add_argument("--repair-checkpoints", action="store_true")
    args = parser.parse_args()
    repaired = 0
    if args.repair_checkpoints:
        repaired = repair_checkpoints(PROJECT_ROOT, args.targets, args.seeds)
    endpoints, checks = validate(PROJECT_ROOT, args.targets, args.seeds)
    checks["repaired_checkpoint_files"] = repaired
    output = PROJECT_ROOT / "results" / "information_ablations" / "full" / args.report_scope
    output.mkdir(parents=True, exist_ok=True)
    endpoints = endpoints.sort_values(["target", "variant", "seed"])
    endpoints.to_csv(output / "endpoint_metrics.csv", index=False)
    summary = (
        endpoints.groupby(["target", "variant"], as_index=False)
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
    summary.to_csv(output / "summary_by_target_variant.csv", index=False)
    (output / "validation_report.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    if checks["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
