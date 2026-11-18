import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.codex_agent_session as session_module
from agentic_al.campaign import _mask_digest, deterministic_availability_mask
from agentic_al.explanations import EVIDENCE_METHOD, canonical_digest


ROOT = Path(__file__).resolve().parents[1]


def _frame(size: int = 30) -> pd.DataFrame:
    affinity = np.linspace(0.0, 10.0, size)
    return pd.DataFrame(
        {
            "SMILES": ["C" * length + "O" for length in range(1, size + 1)],
            "affinity": affinity,
            "top_2p": affinity >= np.quantile(affinity, 0.98),
            "top_5p": affinity >= np.quantile(affinity, 0.95),
            "Target": ["D2R"] * size,
        }
    )


def _config(seed: int, availability: dict | None) -> dict:
    variants = {
        name: {
            "include_uncertainty": name != "minus_uncertainty",
            "include_diversity": name != "minus_diversity",
            "include_memory": name != "minus_memory",
            "evidence_mode": "none",
        }
        for name in session_module.REQUIRED_VARIANTS
    }
    value = {
        "protocol": "test-agent-availability-v1",
        "controller_model": "gpt-5.6-sol",
        "targets": ["D2R"],
        "seeds": [seed],
        "initial_size": 4,
        "cycles": 2,
        "batch_size": 3,
        "radius": 4,
        "n_bits": 24,
        "noise": 0.05,
        "jitter": 0.000001,
        "prompt_version": "test-v1",
        "active_variants": ["full_reference"],
        "evidence": {
            "method": EVIDENCE_METHOD,
            "top_k": 3,
            "min_support": 1,
            "shuffle_protocol_version": "test-shuffle-v1",
        },
        "variants": variants,
    }
    if availability is not None:
        value["availability"] = availability
    return value


def _patch_assets(monkeypatch, frame: pd.DataFrame) -> None:
    features = np.random.default_rng(11).integers(
        0, 2, size=(len(frame), 24), dtype=np.uint8
    )
    monkeypatch.setattr(session_module, "load_target", lambda root, target: frame.copy())
    monkeypatch.setattr(
        session_module,
        "featurize",
        lambda frame, cache_path, radius, n_bits: features.copy(),
    )
    monkeypatch.setattr(
        session_module,
        "scaffolds",
        lambda frame: np.asarray([f"s{index // 2}" for index in range(len(frame))]),
    )


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def test_frozen_agent_availability_config_is_controller_only_and_budgeted():
    config = session_module.load_config(
        ROOT / "configs" / "agent_availability_constrained.json"
    )
    assert config["controller_model"] == "gpt-5.6-sol"
    assert config["active_variants"] == ["full_reference"]
    assert config["availability"] == {
        "fraction": 0.8,
        "salt": "agentic-al-availability-v1",
        "algorithm": "sha256-canonical-isomeric-smiles-u64-v1",
        "grandfather_initial": True,
    }
    assert len(config["targets"]) * len(config["seeds"]) == 40
    assert config["initial_size"] + config["cycles"] * config["batch_size"] == 360


def test_availability_session_preserves_pairing_and_filters_post_initial_pool(
    monkeypatch, tmp_path
):
    frame = _frame()
    _patch_assets(monkeypatch, frame)
    availability = {
        "fraction": 0.7,
        "salt": "test-salt",
        "algorithm": "sha256-canonical-isomeric-smiles-u64-v1",
        "grandfather_initial": True,
    }
    base_mask = deterministic_availability_mask(
        frame["SMILES"], availability["fraction"], availability["salt"]
    )
    seed = next(
        candidate
        for candidate in range(100)
        if (~base_mask[session_module._initial_indices(len(frame), 4, candidate)]).any()
    )
    constrained_config = tmp_path / "constrained.json"
    unconstrained_config = tmp_path / "unconstrained.json"
    _write_json(constrained_config, _config(seed, availability))
    _write_json(unconstrained_config, _config(seed, None))
    constrained_session = tmp_path / "constrained-session.json"
    unconstrained_session = tmp_path / "unconstrained-session.json"

    session_module.initialize(
        tmp_path, constrained_session, "D2R", seed, config_path=constrained_config
    )
    session_module.initialize(
        tmp_path, unconstrained_session, "D2R", seed, config_path=unconstrained_config
    )
    constrained = json.loads(constrained_session.read_text())
    unconstrained = json.loads(unconstrained_session.read_text())
    assert constrained["initial_indices"] == unconstrained["initial_indices"]
    assert any(not base_mask[index] for index in constrained["initial_indices"])

    action_path = tmp_path / "action.json"
    _write_json(
        action_path,
        {
            "n_exploit": 3,
            "n_uncertain": 0,
            "n_diverse": 0,
            "rationale": "Test the highest predicted means.",
            "expected_outcome": "The batch mean will exceed 1.0.",
        },
    )
    session_module.apply_action(tmp_path, constrained_session, action_path)
    updated = json.loads(constrained_session.read_text())
    decision = updated["history"][0]
    assert all(base_mask[index] for index in decision["selected_indices"])
    assert decision["availability_mask_digest"] == _mask_digest(base_mask)
    saved_view = updated["predecision_views"][0]
    assert decision["predecision_view_sha256"] == canonical_digest(saved_view["view"])
    assert saved_view["availability_mask_digest"] == decision["availability_mask_digest"]
    assert updated["trajectory"][-1]["availability_mask_digest"] == _mask_digest(base_mask)


def test_availability_session_rejects_insufficient_endpoint_capacity(
    monkeypatch, tmp_path
):
    frame = _frame()
    _patch_assets(monkeypatch, frame)
    config_path = tmp_path / "too-small.json"
    _write_json(
        config_path,
        _config(
            0,
            {
                "fraction": 0.000001,
                "salt": "capacity-test",
                "algorithm": "sha256-canonical-isomeric-smiles-u64-v1",
                "grandfather_initial": True,
            },
        ),
    )
    session_path = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="frozen endpoint budget"):
        session_module.initialize(
            tmp_path, session_path, "D2R", 0, config_path=config_path
        )
    assert not session_path.exists()
