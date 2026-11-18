#!/usr/bin/env python3
"""Strictly parse and score explicit threshold claims in legacy forecasts."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


NUMBER = r"-?\d+(?:\.\d+)?"
COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass(frozen=True)
class ClaimSpec:
    metric: str
    operator: str
    threshold: float
    source_text: str
    parser_rule: str
    auxiliary_threshold: float | None = None
    auxiliary_operator: str | None = None


def _count(value: str) -> int:
    return COUNT_WORDS.get(value.lower(), int(value) if value.isdigit() else -1)


def parse_expected_outcome(text: str) -> list[ClaimSpec]:
    """Parse only explicit, objective thresholds; vague prose is intentionally ignored."""
    claims: list[ClaimSpec] = []
    occupied: list[tuple[int, int]] = []

    def add_matches(
        rule: str,
        pattern: str,
        builder: Callable[[re.Match[str]], ClaimSpec],
    ) -> None:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            claims.append(builder(match))
            occupied.append(span)

    add_matches(
        "cumulative_mean_at_least",
        rf"cumulative(?: observed)? mean will (?:be|remain) at least (?P<x>{NUMBER})",
        lambda m: ClaimSpec(
            "cumulative_affinity_mean_after", ">=", float(m["x"]), m.group(0),
            "cumulative_mean_at_least"
        ),
    )
    add_matches(
        "scaffold_after_from",
        r"observed unique(?:-| )scaffold(?: count|s)? will increase by at least "
        rf"(?P<n>\d+) from (?P<base>\d+)",
        lambda m: ClaimSpec(
            "observed_unique_scaffolds_after", ">=", float(int(m["base"]) + int(m["n"])),
            m.group(0), "scaffold_after_from", float(m["base"])
        ),
    )
    add_matches(
        "scaffold_after_beyond",
        r"(?:final batch will )?add at least (?P<n>\d+) observed unique scaffolds? beyond "
        r"(?P<base>\d+)",
        lambda m: ClaimSpec(
            "observed_unique_scaffolds_after", ">=", float(int(m["base"]) + int(m["n"])),
            m.group(0), "scaffold_after_beyond", float(m["base"])
        ),
    )
    add_matches(
        "affinity_count_at_least",
        rf"(?:contain|include) at least (?P<count>one|two|three|four|five|six|seven|eight|nine|ten|\d+) "
        rf"(?:(?:selected )?outcomes?|compounds?|affinit(?:y|ies))?(?: with affinity| of)?(?: above)? "
        rf"(?P<cutoff>{NUMBER})(?: or higher)?",
        lambda m: ClaimSpec(
            "batch_affinity_count_at_or_above", ">=", float(_count(m["count"])),
            m.group(0), "affinity_count_at_least", float(m["cutoff"]),
            ">" if " above " in m.group(0).lower() and "or higher" not in m.group(0).lower() else ">="
        ),
    )
    add_matches(
        "batch_mean_exceeds",
        rf"(?:final )?batch mean will exceed (?P<x>{NUMBER})",
        lambda m: ClaimSpec(
            "batch_affinity_mean", ">", float(m["x"]), m.group(0), "batch_mean_exceeds"
        ),
    )
    add_matches(
        "batch_max_exceeds",
        rf"(?:its |the )?maximum will exceed (?P<x>{NUMBER})",
        lambda m: ClaimSpec(
            "batch_affinity_max", ">", float(m["x"]), m.group(0), "batch_max_exceeds"
        ),
    )
    add_matches(
        "batch_max_above",
        rf"(?:achieve )?a maximum(?: affinity| observed outcome)? above (?P<x>{NUMBER})",
        lambda m: ClaimSpec(
            "batch_affinity_max", ">", float(m["x"]), m.group(0), "batch_max_above"
        ),
    )
    add_matches(
        "gp_mean_max_above",
        rf"(?:raise|keep|lift) (?:the )?(?:next|final-cycle) GP maximum above (?P<x>{NUMBER})",
        lambda m: ClaimSpec(
            "post_gp_predicted_mean_max", ">", float(m["x"]), m.group(0),
            "gp_mean_max_above"
        ),
    )
    add_matches(
        "gp_std_max_below",
        rf"(?:lower|reduce) maximum(?: predictive)? uncertainty below (?P<x>{NUMBER})",
        lambda m: ClaimSpec(
            "post_gp_predicted_std_max", "<", float(m["x"]), m.group(0),
            "gp_std_max_below"
        ),
    )
    add_matches(
        "gp_std_median_below",
        rf"(?:lower|reduce|decline in) (?:median GP uncertainty|median uncertainty)(?: from {NUMBER})? "
        rf"(?:to )?below (?P<x>{NUMBER})",
        lambda m: ClaimSpec(
            "post_gp_predicted_std_median", "<", float(m["x"]), m.group(0),
            "gp_std_median_below"
        ),
    )
    add_matches(
        "scaffolds_at_least",
        r"(?:(?:add|adds|adding|increase by|rise by) at least (?P<n>\d+) "
        r"(?:(?:previously unseen|previously unobserved|new|observed unique|observed) )?scaffolds?"
        r"|at least (?P<n2>\d+) "
        r"(?:(?:previously unseen|previously unobserved|new|observed unique|observed) )?scaffolds? "
        r"will be added)",
        lambda m: ClaimSpec(
            "new_scaffolds_in_batch", ">=", float(m["n"] or m["n2"]), m.group(0),
            "scaffolds_at_least"
        ),
    )
    add_matches(
        "scaffold_coverage_at_least",
        r"(?:scaffold coverage should increase|observed scaffold count should rise) by at least "
        r"(?P<n>\d+)",
        lambda m: ClaimSpec(
            "new_scaffolds_in_batch", ">=", float(m["n"]), m.group(0),
            "scaffold_coverage_at_least"
        ),
    )
    add_matches(
        "scaffolds_or_more",
        r"add at least (?P<n>\d+) scaffolds?|add (?P<n2>\d+) or more scaffolds?",
        lambda m: ClaimSpec(
            "new_scaffolds_in_batch", ">=", float(m["n"] or m["n2"]), m.group(0),
            "scaffolds_or_more"
        ),
    )
    return claims


def _observed_value(row: pd.Series, claim: ClaimSpec) -> float | None:
    if claim.metric == "batch_affinity_count_at_or_above":
        affinities = json.loads(row["selected_affinities_json"])
        cutoff = float(claim.auxiliary_threshold)
        if claim.auxiliary_operator == ">":
            return float(sum(float(value) > cutoff for value in affinities))
        return float(sum(float(value) >= cutoff for value in affinities))
    value = row.get(claim.metric)
    return None if value is None or pd.isna(value) else float(value)


def _evaluate(observed: float, operator: str, threshold: float, tolerance: float) -> tuple[bool, float]:
    if operator == ">":
        return observed > threshold + tolerance, observed - threshold
    if operator == ">=":
        return observed >= threshold - tolerance, observed - threshold
    if operator == "<":
        return observed < threshold - tolerance, threshold - observed
    if operator == "<=":
        return observed <= threshold + tolerance, threshold - observed
    raise ValueError(f"Unsupported operator: {operator}")


def score_outcomes(
    outcomes: pd.DataFrame, tolerance: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    claim_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    for _, row in outcomes.sort_values(["target", "seed", "cycle"]).iterrows():
        specs = parse_expected_outcome(str(row["expected_outcome"]))
        scored: list[bool] = []
        unavailable = 0
        for claim_number, spec in enumerate(specs, start=1):
            observed = _observed_value(row, spec)
            if observed is None:
                satisfied = None
                margin = None
                unavailable += 1
            else:
                satisfied, margin = _evaluate(observed, spec.operator, spec.threshold, tolerance)
                scored.append(satisfied)
            claim_rows.append(
                {
                    "target": row["target"],
                    "policy": row["policy"],
                    "model": row["model"],
                    "seed": int(row["seed"]),
                    "cycle": int(row["cycle"]),
                    "claim_id": claim_number,
                    "metric": spec.metric,
                    "operator": spec.operator,
                    "threshold": spec.threshold,
                    "auxiliary_threshold": spec.auxiliary_threshold,
                    "auxiliary_operator": spec.auxiliary_operator,
                    "observed_value": observed,
                    "satisfied": satisfied,
                    "signed_threshold_margin": margin,
                    "parser_rule": spec.parser_rule,
                    "source_text": spec.source_text,
                    "expected_outcome": row["expected_outcome"],
                }
            )
        scorable = len(scored)
        parsed = len(specs)
        decision_rows.append(
            {
                "target": row["target"],
                "policy": row["policy"],
                "model": row["model"],
                "seed": int(row["seed"]),
                "cycle": int(row["cycle"]),
                "expected_outcome": row["expected_outcome"],
                "parsed_claims": parsed,
                "scored_claims": scorable,
                "unavailable_claims": unavailable,
                "satisfied_claims": int(sum(scored)),
                "fraction_clauses_satisfied": float(np.mean(scored)) if scored else None,
                "conjunction_success": bool(all(scored)) if scored else None,
                "expectation_scorable": bool(scored),
                "unscorable_reason": "" if scored else (
                    "recognized threshold outcome unavailable" if parsed else "no supported explicit threshold clause"
                ),
            }
        )

    claims = pd.DataFrame(claim_rows)
    decisions = pd.DataFrame(decision_rows)
    unscorable = decisions.loc[~decisions["expectation_scorable"]].copy()

    summaries: list[dict[str, object]] = []
    groups: list[tuple[str, str, pd.DataFrame, pd.DataFrame]] = [
        ("overall", "all", decisions, claims)
    ]
    groups.extend(
        ("target", str(target), group, claims[claims["target"] == target])
        for target, group in decisions.groupby("target", sort=True)
    )
    if len(claims):
        groups.extend(
            ("metric", str(metric), decisions, group)
            for metric, group in claims.groupby("metric", sort=True)
        )
    for scope, group_name, decision_group, claim_group in groups:
        scored_claims = claim_group[claim_group["satisfied"].notna()] if len(claim_group) else claim_group
        scorable_decisions = decision_group[decision_group["expectation_scorable"]]
        summaries.append(
            {
                "scope": scope,
                "group": group_name,
                "decisions": int(len(decision_group)) if scope != "metric" else None,
                "scorable_decisions": int(len(scorable_decisions)) if scope != "metric" else None,
                "unscorable_decisions": int((~decision_group["expectation_scorable"]).sum()) if scope != "metric" else None,
                "unscorable_rate": float((~decision_group["expectation_scorable"]).mean()) if scope != "metric" else None,
                "atomic_claims": int(len(claim_group)),
                "scored_atomic_claims": int(len(scored_claims)),
                "atomic_success_rate": float(scored_claims["satisfied"].astype(bool).mean()) if len(scored_claims) else None,
                "mean_fraction_clauses_satisfied": float(scorable_decisions["fraction_clauses_satisfied"].mean()) if len(scorable_decisions) and scope != "metric" else None,
                "conjunction_success_rate": float(scorable_decisions["conjunction_success"].astype(bool).mean()) if len(scorable_decisions) and scope != "metric" else None,
                "mean_signed_threshold_margin": float(scored_claims["signed_threshold_margin"].mean()) if len(scored_claims) else None,
            }
        )
    summary = pd.DataFrame(summaries)
    validation = {
        "status": "PASS",
        "decision_rows": int(len(decisions)),
        "claim_rows": int(len(claims)),
        "scorable_decisions": int(decisions["expectation_scorable"].sum()),
        "unscorable_decisions": int((~decisions["expectation_scorable"]).sum()),
        "unscorable_rate": float((~decisions["expectation_scorable"]).mean()),
        "unavailable_claims": int(decisions["unavailable_claims"].sum()),
        "probabilistic_calibration_computed": False,
        "note": "Legacy expectations contain no probabilities; rates are threshold-claim audit results, not calibration.",
    }
    return claims, decisions, summary, unscorable, validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/calibration_v1.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text())
    legacy = config["legacy"]
    outcomes = pd.read_csv(root / legacy["outcomes_csv"])
    outputs = score_outcomes(outcomes, float(config["scoring"]["numeric_tolerance"]))
    claims, decisions, summary, unscorable, validation = outputs
    paths = {
        "claim_scores_csv": claims,
        "decision_scores_csv": decisions,
        "summary_csv": summary,
        "unscorable_csv": unscorable,
    }
    for config_key, frame in paths.items():
        path = root / legacy[config_key]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    validation_path = root / legacy["scoring_validation_json"]
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
