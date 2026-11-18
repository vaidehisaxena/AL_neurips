#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import statistics
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

OUT = ROOT / "results" / "availability_constrained" / "agent_full"
CONFIG_PATH = ROOT / "configs" / "agent_availability_constrained.json"
CAMPAIGNS = {
    "D2R": tuple(range(2, 10)),
    "Mpro": tuple(range(10)),
    "TYK2": tuple(range(10)),
    "USP7": tuple(range(10)),
}
METRICS = (
    "best_affinity",
    "simple_regret",
    "top_1p_recall",
    "top_2p_recall",
    "top_5p_recall",
    "unique_top_5p_scaffolds",
)


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate(target: str, seed: int, config: dict, frame) -> dict:
    prefix = f"{target} seed {seed}"
    folder = OUT / target
    session = read(folder / f"session_seed{seed}.json")
    exported = read(folder / f"export_seed{seed}.json")
    require(session == exported, f"{prefix}: export differs from session")
    require(session["target"] == target and session["seed"] == seed, f"{prefix}: identity")
    require(session["protocol"] == config["protocol"], f"{prefix}: protocol")
    require(session["variant"] == "full_reference", f"{prefix}: variant")
    require(session["controller_model"] == "gpt-5.6-sol", f"{prefix}: controller model")
    require(session["cycle"] == 10, f"{prefix}: terminal cycle")
    require(len(session["history"]) == 10, f"{prefix}: history count")
    require(len(session["predecision_views"]) == 10, f"{prefix}: view count")
    require(len(session["trajectory"]) == 11, f"{prefix}: trajectory count")

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
    require(session["config_sha256"] == config_digest, f"{prefix}: config digest")
    require(session["system_prompt_sha256"] == prompt_digest, f"{prefix}: prompt digest")
    require(session["initial_indices_sha256"] == initial_json_digest, f"{prefix}: initial JSON digest")
    require(session["initial_set_digest"] == initial_set_digest, f"{prefix}: initial set digest")
    require(session["availability_mask_digest"] == mask_digest, f"{prefix}: mask digest")
    require(session["availability_fraction"] == 0.8, f"{prefix}: availability fraction")
    require(session["availability_salt"] == config["availability"]["salt"], f"{prefix}: availability salt")
    require(len(initial) == 60 and len(set(initial.tolist())) == 60, f"{prefix}: initial uniqueness")

    selected_all: list[int] = []
    for cycle, (decision, view_record) in enumerate(
        zip(session["history"], session["predecision_views"]), start=1
    ):
        action = read(folder / f"action_seed{seed}_cycle{cycle}.json")
        saved_view = read(folder / f"view_seed{seed}_cycle{cycle}.json")
        counts = [action[k] for k in ("n_exploit", "n_uncertain", "n_diverse")]
        require(all(type(value) is int and value >= 0 for value in counts), f"{prefix} cycle {cycle}: allocation")
        require(sum(counts) == 30, f"{prefix} cycle {cycle}: allocation sum")
        require(bool(action["rationale"].strip()) and bool(action["expected_outcome"].strip()), f"{prefix} cycle {cycle}: action text")
        for key in ("n_exploit", "n_uncertain", "n_diverse", "rationale", "expected_outcome"):
            require(decision[key] == action[key], f"{prefix} cycle {cycle}: action mismatch {key}")

        chosen = [int(value) for value in decision["selected_indices"]]
        require(len(chosen) == 30 and len(set(chosen)) == 30, f"{prefix} cycle {cycle}: batch uniqueness")
        require(all(bool(base_mask[index]) for index in chosen), f"{prefix} cycle {cycle}: unavailable selection")
        selected_all.extend(chosen)

        view_digest = canonical_digest(saved_view)
        require(view_record["view"] == saved_view, f"{prefix} cycle {cycle}: saved view mismatch")
        require(view_record["view_sha256"] == view_digest, f"{prefix} cycle {cycle}: view record digest")
        require(decision["predecision_view_sha256"] == view_digest, f"{prefix} cycle {cycle}: decision view digest")
        evidence = saved_view["explanatory_evidence"]
        evidence_digest = canonical_digest({k: v for k, v in evidence.items() if k != "evidence_sha256"})
        require(evidence["evidence_sha256"] == evidence_digest, f"{prefix} cycle {cycle}: evidence self-digest")
        require(view_record["evidence_sha256"] == evidence_digest, f"{prefix} cycle {cycle}: evidence record digest")
        require(decision["evidence_sha256"] == evidence_digest, f"{prefix} cycle {cycle}: decision evidence digest")
        for holder in (decision, view_record):
            require(holder["availability_mask_digest"] == mask_digest, f"{prefix} cycle {cycle}: mask provenance")
            require(holder["initial_set_digest"] == initial_set_digest, f"{prefix} cycle {cycle}: initial provenance")
        require(decision["config_sha256"] == config_digest, f"{prefix} cycle {cycle}: config provenance")
        require(decision["system_prompt_sha256"] == prompt_digest, f"{prefix} cycle {cycle}: prompt provenance")
        require(decision["initial_indices_sha256"] == initial_json_digest, f"{prefix} cycle {cycle}: initial JSON provenance")

    require(len(selected_all) == 300 and len(set(selected_all)) == 300, f"{prefix}: cross-cycle uniqueness")
    require(set(initial.tolist()).isdisjoint(selected_all), f"{prefix}: initial reselection")
    require(session["observed_indices"] == initial.tolist() + selected_all, f"{prefix}: observed order")
    require(len(session["observed_indices"]) == 360, f"{prefix}: endpoint budget")
    require([row["cycle"] for row in session["trajectory"]] == list(range(11)), f"{prefix}: trajectory cycles")
    require([row["measured"] for row in session["trajectory"]] == [60 + 30 * c for c in range(11)], f"{prefix}: trajectory budgets")
    require(all(row["availability_mask_digest"] == mask_digest for row in session["trajectory"]), f"{prefix}: trajectory masks")
    require(all(row["initial_set_digest"] == initial_set_digest for row in session["trajectory"]), f"{prefix}: trajectory initials")

    endpoint = session["trajectory"][-1]
    return {
        "target": target,
        "seed": seed,
        "decisions": 10,
        "selected_unique": 300,
        "unavailable_post_initial": sum(not bool(base_mask[index]) for index in selected_all),
        "measured": endpoint["measured"],
        "availability_mask_digest": mask_digest,
        "initial_set_digest": initial_set_digest,
        "config_sha256": config_digest,
        "system_prompt_sha256": prompt_digest,
        **{key: endpoint[key] for key in METRICS},
    }


def main() -> None:
    config = read(CONFIG_PATH)
    endpoints = []
    for target, seeds in CAMPAIGNS.items():
        frame = load_target(ROOT, target)
        target_rows = [validate(target, seed, config, frame) for seed in seeds]
        endpoints.extend(target_rows)
        target_summary = {
            "target": target,
            "campaigns": len(target_rows),
            "seeds": list(seeds),
            "endpoint_measured_per_campaign": 360,
            "metrics": {
                key: {
                    "mean": statistics.fmean(float(row[key]) for row in target_rows),
                    "population_std": statistics.pstdev(float(row[key]) for row in target_rows),
                    "min": min(float(row[key]) for row in target_rows),
                    "max": max(float(row[key]) for row in target_rows),
                }
                for key in METRICS
            },
        }
        write(OUT / target / "endpoint_summary.json", target_summary)

    validation = {
        "status": "pass",
        "protocol": config["protocol"],
        "variant": "full_reference",
        "campaigns": len(endpoints),
        "decisions": sum(row["decisions"] for row in endpoints),
        "selected_compounds": sum(row["selected_unique"] for row in endpoints),
        "unavailable_post_initial": sum(row["unavailable_post_initial"] for row in endpoints),
        "endpoint_budget_failures": sum(row["measured"] != 360 for row in endpoints),
        "anomalies": [],
        "campaign_results": endpoints,
    }
    write(OUT / "validation_report.json", validation)
    write(OUT / "endpoints.json", endpoints)
    with (OUT / "endpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoints[0]))
        writer.writeheader()
        writer.writerows(endpoints)
    summary = {
        "protocol": config["protocol"],
        "controller_model": config["controller_model"],
        "variant": "full_reference",
        "availability_fraction": config["availability"]["fraction"],
        "campaigns": len(endpoints),
        "targets": {target: read(OUT / target / "endpoint_summary.json") for target in CAMPAIGNS},
    }
    write(OUT / "summary.json", summary)

    # The two validated D2R development-gate runs are canonical study members.
    gate_report = read(OUT.parent / "agent_dev_gate" / "validation_report.json")
    require(gate_report["status"] == "pass", "development gate is not validated")
    gate_endpoints = []
    for row in gate_report["seeds"]:
        gate_endpoints.append({
            "target": "D2R",
            "seed": row["seed"],
            "decisions": row["decisions"],
            "selected_unique": row["selected_unique"],
            "unavailable_post_initial": row["unavailable_post_initial"],
            "measured": row["measured"],
            "availability_mask_digest": row["availability_mask_digest"],
            "initial_set_digest": row["initial_set_digest"],
            "config_sha256": row["config_sha256"],
            "system_prompt_sha256": row["system_prompt_sha256"],
            **row["endpoint_metrics"],
        })
    canonical_endpoints = sorted(gate_endpoints + endpoints, key=lambda row: (row["target"], row["seed"]))
    canonical_targets = {}
    for target in CAMPAIGNS:
        rows = [row for row in canonical_endpoints if row["target"] == target]
        canonical_targets[target] = {
            "campaigns": len(rows),
            "seeds": [row["seed"] for row in rows],
            "metrics": {
                key: {
                    "mean": statistics.fmean(float(row[key]) for row in rows),
                    "population_std": statistics.pstdev(float(row[key]) for row in rows),
                    "min": min(float(row[key]) for row in rows),
                    "max": max(float(row[key]) for row in rows),
                }
                for key in METRICS
            },
        }
    canonical_summary = {
        "status": "pass",
        "protocol": config["protocol"],
        "controller_model": config["controller_model"],
        "variant": "full_reference",
        "availability_fraction": config["availability"]["fraction"],
        "campaigns": len(canonical_endpoints),
        "decisions": sum(row["decisions"] for row in canonical_endpoints),
        "selected_compounds": sum(row["selected_unique"] for row in canonical_endpoints),
        "total_measured_endpoints": sum(row["measured"] for row in canonical_endpoints),
        "unavailable_post_initial": sum(row["unavailable_post_initial"] for row in canonical_endpoints),
        "anomalies": [],
        "targets": canonical_targets,
    }
    write(OUT / "canonical_40_endpoints.json", canonical_endpoints)
    with (OUT / "canonical_40_endpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(canonical_endpoints[0]))
        writer.writeheader()
        writer.writerows(canonical_endpoints)
    write(OUT / "canonical_40_summary.json", canonical_summary)
    print(json.dumps({k: validation[k] for k in ("status", "campaigns", "decisions", "selected_compounds", "unavailable_post_initial", "endpoint_budget_failures", "anomalies")}, indent=2))


if __name__ == "__main__":
    main()
