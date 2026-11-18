from pathlib import Path

import numpy as np

from agentic_al.campaign import (
    CampaignState,
    ExactTanimotoGP,
    HiddenAssay,
    deterministic_availability_mask,
    select_batch,
)
from agentic_al.agent import (
    BatchAllocation,
    allocation_json_schema,
    select_portfolio_batch,
    summarize_agent_state,
)


def test_hidden_assay_rejects_duplicate_and_repeated_requests():
    assay = HiddenAssay(np.array([1.0, 2.0, 3.0]))
    assert assay.reveal([1]).tolist() == [2.0]
    try:
        assay.reveal([1])
    except ValueError:
        pass
    else:
        raise AssertionError("Repeated labels must not be revealed")


def test_agent_state_contains_only_observed_affinities():
    state = CampaignState(1, (0, 2), (1.0, 3.0), 30, 0.5)
    assert not hasattr(state, "hidden_labels")
    assert state.observed_affinities == (1.0, 3.0)


def test_tanimoto_kernel_and_gp_are_finite():
    x = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    kernel = ExactTanimotoGP.kernel(x[:1], x[1:2])
    assert np.isclose(kernel.item(), 1 / 3)
    mean, std = ExactTanimotoGP().fit(x[:2], np.array([1.0, 2.0])).predict(x)
    assert np.isfinite(mean).all()
    assert (std > 0).all()


def test_selection_is_deterministic_and_respects_pool():
    pool = np.array([3, 4, 5, 6])
    mean = np.array([0.2, 0.9, 0.1, 0.8])
    std = np.array([0.9, 0.1, 0.8, 0.2])
    first = select_batch("balanced", pool, mean, std, 2, np.random.default_rng(7))
    second = select_batch("balanced", pool, mean, std, 2, np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert set(first).issubset(set(pool))
    assert len(set(first)) == 2


def test_agent_allocation_validation_and_schema():
    action = BatchAllocation(10, 10, 10, "Balance the queues.", "At least one new hit.")
    assert action.validate(30) is action
    assert allocation_json_schema(30)["additionalProperties"] is False
    try:
        BatchAllocation(10, 10, 9, "Bad total.", "No expectation.").validate(30)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid batch totals must be rejected")


def test_agent_summary_cannot_accept_hidden_pool_labels():
    summary = summarize_agent_state(
        target="D2R",
        cycle=1,
        cycles=10,
        batch_size=30,
        observed_affinities=np.array([1.0, 2.0]),
        predicted_mean=np.array([1.5, 1.8, 2.1]),
        predicted_std=np.array([0.2, 0.4, 0.3]),
        observed_scaffolds=np.array(["a", "b"]),
        pool_scaffolds=np.array(["c", "c", "d"]),
    )
    assert summary["observed_affinity"]["best"] == 2.0
    assert "hidden_labels" not in summary
    assert "pool_affinity" not in summary


def test_portfolio_selection_is_unique_and_deterministic():
    features = np.array(
        [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]], dtype=np.uint8
    )
    allocation = BatchAllocation(1, 1, 1, "Use all queues.", "Increase coverage.")
    kwargs = dict(
        allocation=allocation,
        pool=np.array([1, 2, 3, 4]),
        mean=np.array([0.9, 0.2, 0.3, 0.8]),
        std=np.array([0.1, 0.9, 0.8, 0.2]),
        features=features,
        observed_indices=np.array([0]),
    )
    first = select_portfolio_batch(**kwargs)
    second = select_portfolio_batch(**kwargs)
    assert np.array_equal(first, second)
    assert len(first) == len(set(first)) == 3
    assert set(first).issubset({1, 2, 3, 4})


def test_availability_mask_is_deterministic_and_label_independent():
    smiles = ["C" * length + "O" for length in range(1, 80)]
    first = deterministic_availability_mask(smiles, 0.8, "frozen-salt")
    second = deterministic_availability_mask(smiles, 0.8, "frozen-salt")
    assert np.array_equal(first, second)
    assert first.dtype == bool
    assert 0 < first.sum() < len(first)
    try:
        deterministic_availability_mask(smiles, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid availability fractions must be rejected")


def test_availability_mask_uses_canonical_smiles():
    mask = deterministic_availability_mask(["CCO", "OCC"], 0.8, "frozen-salt")
    assert mask[0] == mask[1]
