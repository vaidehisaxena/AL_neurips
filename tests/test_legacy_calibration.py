import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


backfill = _load("backfill_batch_outcomes", ROOT / "scripts" / "backfill_batch_outcomes.py")
scoring = _load("score_expected_outcomes", ROOT / "scripts" / "score_expected_outcomes.py")
parse_selected_indices = backfill.parse_selected_indices
parse_expected_outcome = scoring.parse_expected_outcome
score_outcomes = scoring.score_outcomes


def test_selected_index_parser_accepts_both_legacy_encodings():
    assert parse_selected_indices("1;2;3") == [1, 2, 3]
    assert parse_selected_indices("[1, 2, 3]") == [1, 2, 3]


def test_reconstruction_builds_one_complete_campaign(monkeypatch, tmp_path):
    frame = pd.DataFrame({"affinity": np.arange(360, dtype=float)})
    monkeypatch.setattr(backfill, "load_target", lambda root, target: frame)
    monkeypatch.setattr(
        backfill, "scaffolds", lambda value: np.asarray([f"s{index}" for index in range(360)])
    )
    monkeypatch.setattr(
        backfill, "_initial_indices", lambda root, target, seed, count: np.arange(60)
    )
    monkeypatch.setattr(backfill, "_state_from_row_or_disk", lambda root, row: None)
    rows = []
    for cycle in range(1, 11):
        chosen = list(range(60 + 30 * (cycle - 1), 60 + 30 * cycle))
        rows.append(
            {
                "target": "toy",
                "policy": "codex-agent",
                "model": "test",
                "seed": 0,
                "cycle": cycle,
                "expected_outcome": "The batch mean will exceed 0.",
                "rationale": "Test fixture.",
                "n_exploit": 10,
                "n_uncertain": 10,
                "n_diverse": 10,
                "selected_indices": ";".join(map(str, chosen)),
            }
        )
    outcomes, validation = backfill.reconstruct_outcomes(tmp_path, pd.DataFrame(rows))
    assert len(outcomes) == 10
    assert outcomes.loc[0, "batch_affinity_mean"] == 74.5
    assert outcomes.loc[0, "new_scaffolds_in_batch"] == 30
    assert validation["status"] == "PASS"


def test_parser_extracts_batch_mean_max_and_scaffold_thresholds():
    text = (
        "The batch mean will exceed 8.2, its maximum will exceed 10.5, "
        "and at least 5 new scaffolds will be added."
    )
    claims = parse_expected_outcome(text)
    assert [(claim.metric, claim.operator, claim.threshold) for claim in claims] == [
        ("batch_affinity_mean", ">", 8.2),
        ("batch_affinity_max", ">", 10.5),
        ("new_scaffolds_in_batch", ">=", 5.0),
    ]


def test_parser_extracts_affinity_count_with_cutoff():
    claims = parse_expected_outcome(
        "The batch will contain at least two compounds with affinity 9.0 or higher "
        "and add at least 5 new scaffolds."
    )
    count_claim = next(claim for claim in claims if claim.metric.startswith("batch_affinity_count"))
    assert count_claim.threshold == 2
    assert count_claim.auxiliary_threshold == 9.0
    assert count_claim.auxiliary_operator == ">="

    above_claim = parse_expected_outcome(
        "The batch should include at least two selected outcomes above 9.0."
    )[0]
    assert above_claim.auxiliary_operator == ">"


def test_vague_claim_is_unscorable():
    assert parse_expected_outcome(
        "Expand scaffold coverage and recover competitive high-affinity compounds."
    ) == []


def test_scoring_reports_atomic_and_conjunction_results():
    row = {
        "target": "TYK2",
        "policy": "codex-agent",
        "model": "gpt-5.6-sol",
        "seed": 0,
        "cycle": 1,
        "expected_outcome": (
            "The batch mean will exceed 8.0, its maximum will exceed 10.5, "
            "and at least 2 new scaffolds will be added."
        ),
        "selected_affinities_json": json.dumps([7.0, 9.0, 11.0]),
        "batch_affinity_mean": 9.0,
        "batch_affinity_max": 11.0,
        "new_scaffolds_in_batch": 1,
        "cumulative_affinity_mean_after": 8.0,
        "observed_unique_scaffolds_after": 10,
        "post_gp_predicted_mean_max": np.nan,
        "post_gp_predicted_std_median": np.nan,
        "post_gp_predicted_std_max": np.nan,
    }
    claims, decisions, _, _, validation = score_outcomes(pd.DataFrame([row]), 1e-9)
    assert claims["satisfied"].tolist() == [True, True, False]
    assert decisions.loc[0, "fraction_clauses_satisfied"] == 2 / 3
    assert not bool(decisions.loc[0, "conjunction_success"])
    assert validation["probabilistic_calibration_computed"] is False
