import inspect
import json
from pathlib import Path

import numpy as np

from agentic_al.agent import AgentViewSpec, project_history, summarize_agent_state
from agentic_al.explanations import (
    build_explanatory_evidence,
    canonical_digest,
    shuffle_explanatory_evidence,
)
from scripts.codex_agent_session import _initial_indices, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "agent_information_ablations.json"


def _evidence() -> dict:
    features = np.array(
        [
            [1, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 1],
            [1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    affinities = np.array([8.0, 7.5, 7.0, 4.0, 3.5, 3.0, 6.0, 5.0])
    return build_explanatory_evidence(features, affinities, top_k=5, min_support=2)


def _state(spec: AgentViewSpec) -> dict:
    history = [
        {
            "cycle": 1,
            "n_exploit": 10,
            "n_uncertain": 10,
            "n_diverse": 10,
            "rationale": "A reason.",
            "expected_outcome": "A testable outcome.",
            "selected_predicted_mean": 5.0,
            "selected_predicted_std": 1.2,
            "selected_new_scaffolds": 7,
            "observed_best_before": 7.0,
            "observed_best_after": 8.0,
            "improvement": 1.0,
            "selected_indices": [99],
        }
    ]
    return summarize_agent_state(
        target="D2R",
        cycle=2,
        cycles=10,
        batch_size=30,
        observed_affinities=np.array([4.0, 8.0]),
        predicted_mean=np.array([5.0, 6.0, 7.0]),
        predicted_std=np.array([0.1, 0.3, 0.2]),
        observed_scaffolds=np.array(["a", "b"]),
        pool_scaffolds=np.array(["c", "c", "d"]),
        history=history,
        view_spec=spec,
        explanatory_evidence=_evidence(),
    )


def test_frozen_information_config_has_paired_five_arm_matrix():
    config = load_config(CONFIG_PATH)
    assert set(config["variants"]) == {
        "full_reference",
        "minus_uncertainty",
        "minus_diversity",
        "minus_memory",
        "shuffled_evidence",
    }
    assert len(config["targets"]) * len(config["seeds"]) * len(config["variants"]) == 200
    assert config["initial_size"] + config["cycles"] * config["batch_size"] == 360


def test_information_variants_expose_only_the_prespecified_keys():
    config = json.loads(CONFIG_PATH.read_text())
    states = {
        name: _state(AgentViewSpec.from_dict({"name": name, **raw}))
        for name, raw in config["variants"].items()
        if name != "shuffled_evidence"
    }
    assert "gp_predicted_std" in states["full_reference"]
    assert "chemical_space" in states["full_reference"]
    assert "explanatory_evidence" in states["full_reference"]

    assert "gp_predicted_std" not in states["minus_uncertainty"]
    assert "selected_predicted_std" not in states["minus_uncertainty"]["history"][0]
    assert "chemical_space" not in states["minus_diversity"]
    assert "selected_new_scaffolds" not in states["minus_diversity"]["history"][0]
    assert states["minus_memory"]["history"] == []
    assert "selected_indices" not in states["full_reference"]["history"][0]


def test_observed_only_evidence_api_has_no_hidden_label_inputs():
    parameters = set(inspect.signature(build_explanatory_evidence).parameters)
    assert parameters == {
        "observed_features",
        "observed_affinities",
        "top_k",
        "min_support",
    }
    assert not {"pool_affinities", "labels", "top_2p", "top_5p", "frame"}.intersection(
        parameters
    )


def test_evidence_shuffle_is_deterministic_nonidentity_and_value_preserving():
    evidence = _evidence()
    kwargs = dict(target="D2R", seed=3, cycle=4, protocol_version="shuffle-v1")
    first = shuffle_explanatory_evidence(evidence, **kwargs)
    second = shuffle_explanatory_evidence(evidence, **kwargs)
    assert first == second
    original_rows = evidence["features"]
    shuffled_rows = first["features"]
    assert all(
        original["bit_index"] != shuffled["bit_index"]
        for original, shuffled in zip(original_rows, shuffled_rows)
    )
    assert sorted(row["bit_index"] for row in original_rows) == sorted(
        row["bit_index"] for row in shuffled_rows
    )
    preserved = (
        "rank",
        "score",
        "sign",
        "support",
        "prevalence",
        "affinity_difference",
        "standardized_enrichment",
    )
    assert [[row[key] for key in preserved] for row in original_rows] == [
        [row[key] for key in preserved] for row in shuffled_rows
    ]
    assert set(first) == set(evidence)
    assert first["method"] == evidence["method"]


def test_shuffle_seed_does_not_change_paired_initialization():
    expected = _initial_indices(pool_size=2502, initial_size=60, seed=7)
    expected_digest = canonical_digest(expected.tolist())
    for variant in json.loads(CONFIG_PATH.read_text())["variants"]:
        actual = _initial_indices(pool_size=2502, initial_size=60, seed=7)
        assert canonical_digest(actual.tolist()) == expected_digest, variant


def test_history_projection_does_not_mutate_internal_log():
    history = [{"cycle": 1, "selected_predicted_std": 2.0, "selected_indices": [4]}]
    projected = project_history(
        history, AgentViewSpec(name="minus_uncertainty", include_uncertainty=False)
    )
    assert "selected_predicted_std" not in projected[0]
    assert history[0]["selected_predicted_std"] == 2.0
