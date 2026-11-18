from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import agentic_al.campaign as campaign_module
from agentic_al.campaign import CampaignConfig, run_campaign
from scripts.run_agentic_baselines import POLICIES, validate_resume_outputs


def _toy_frame(size: int = 25) -> pd.DataFrame:
    labels = np.linspace(0.0, 10.0, size)
    return pd.DataFrame(
        {
            "SMILES": ["C" * length + "O" for length in range(1, size + 1)],
            "affinity": labels,
            "top_2p": labels >= np.quantile(labels, 0.98),
            "top_5p": labels >= np.quantile(labels, 0.95),
            "Target": ["D2R"] * size,
        }
    )


def _patch_toy_campaign(monkeypatch: pytest.MonkeyPatch, base_mask: np.ndarray) -> None:
    frame = _toy_frame(len(base_mask))
    features = np.random.default_rng(3).integers(0, 2, size=(len(frame), 24), dtype=np.uint8)
    monkeypatch.setattr(campaign_module, "load_target", lambda root, target: frame.copy())
    monkeypatch.setattr(
        campaign_module,
        "featurize",
        lambda frame, cache_path, radius, n_bits: features.copy(),
    )
    monkeypatch.setattr(
        campaign_module,
        "scaffolds",
        lambda frame: np.asarray([f"s{index // 2}" for index in range(len(frame))]),
    )
    monkeypatch.setattr(
        campaign_module,
        "deterministic_availability_mask",
        lambda smiles, fraction, salt: base_mask.copy(),
    )


def test_availability_campaign_is_paired_budgeted_and_auditable(monkeypatch, tmp_path):
    base_mask = np.asarray([True] * 20 + [False] * 5)
    _patch_toy_campaign(monkeypatch, base_mask)
    config = CampaignConfig(
        target="D2R",
        policy="random",
        seed=4,
        initial_size=5,
        batch_size=4,
        cycles=2,
        availability_fraction=0.8,
        availability_salt="test-salt",
        n_bits=24,
    )
    trajectory, decisions = run_campaign(tmp_path, config)

    initial = set(np.sort(np.random.default_rng(4).choice(25, size=5, replace=False)))
    selected = {
        int(value)
        for packed in decisions["selected_indices"]
        for value in packed.split(";")
    }
    assert trajectory["measured"].tolist() == [5, 9, 13]
    assert len(selected) == 8
    assert selected.isdisjoint(initial)
    assert all(base_mask[index] for index in selected)
    assert trajectory["initial_set_digest"].nunique() == 1
    assert trajectory["availability_mask_digest"].nunique() == 1
    assert decisions["availability_fraction"].eq(0.8).all()
    assert decisions["availability_base_count"].eq(20).all()
    assert decisions["availability_effective_count"].ge(20).all()


def test_availability_campaign_fails_before_exceeding_endpoint_budget(monkeypatch, tmp_path):
    _patch_toy_campaign(monkeypatch, np.zeros(25, dtype=bool))
    config = CampaignConfig(
        target="D2R",
        policy="random",
        seed=0,
        initial_size=5,
        batch_size=4,
        cycles=2,
        availability_fraction=0.2,
        n_bits=24,
    )
    with pytest.raises(ValueError, match="frozen endpoint budget"):
        run_campaign(tmp_path, config)


def _resume_frames(cycles: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = {
        "target": "D2R",
        "policy": "random",
        "seed": 0,
        "initial_size": 5,
        "batch_size": 4,
        "cycles": cycles,
        "availability_fraction": 0.8,
        "availability_salt": "test-salt",
        "availability_mask_digest": "mask",
        "initial_set_digest": "initial",
    }
    trajectory = pd.DataFrame(
        [{**common, "cycle": cycle, "measured": 5 + 4 * cycle} for cycle in range(cycles + 1)]
    )
    decisions = pd.DataFrame(
        [{**common, "cycle": cycle} for cycle in range(1, cycles + 1)]
    )
    return trajectory, decisions


def _validate(tmp_path: Path):
    return validate_resume_outputs(
        tmp_path / "trajectories.csv",
        tmp_path / "decisions.csv",
        initial_size=5,
        batch_size=4,
        cycles=2,
        availability_fraction=0.8,
        availability_salt="test-salt",
    )


def test_resume_validation_accepts_only_complete_matching_campaigns(tmp_path):
    trajectory, decisions = _resume_frames()
    trajectory.to_csv(tmp_path / "trajectories.csv", index=False)
    decisions.to_csv(tmp_path / "decisions.csv", index=False)
    _, _, completed = _validate(tmp_path)
    assert completed == {("D2R", "random", 0)}
    assert len(POLICIES) == 9

    decisions.loc[0, "availability_salt"] = "different"
    decisions.to_csv(tmp_path / "decisions.csv", index=False)
    with pytest.raises(ValueError, match="availability_salt"):
        _validate(tmp_path)


def test_resume_validation_rejects_partial_or_unpaired_output(tmp_path):
    trajectory, decisions = _resume_frames()
    trajectory.to_csv(tmp_path / "trajectories.csv", index=False)
    with pytest.raises(ValueError, match="must both exist"):
        _validate(tmp_path)

    decisions.iloc[:1].to_csv(tmp_path / "decisions.csv", index=False)
    with pytest.raises(ValueError, match="incomplete or duplicate decisions"):
        _validate(tmp_path)
