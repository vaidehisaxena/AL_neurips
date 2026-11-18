from pathlib import Path

import pytest

from scripts.run_versioned_api_check import (
    build_call_manifest,
    build_payload,
    load_config,
    require_paid_gate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_api_config_has_exact_twenty_call_matrix_and_existing_inputs():
    config = load_config(ROOT / "configs" / "versioned_api_check.json")
    manifest = build_call_manifest(config)
    assert len(manifest) == len(set(manifest)) == 20
    assert manifest[0] == ("D2R", 0, 1)
    assert manifest[-1] == ("D2R", 1, 10)
    for _, seed, cycle in manifest:
        path = ROOT / config["state_path_template"].format(seed=seed, cycle=cycle)
        assert path.exists()


def test_paid_gate_requires_both_flag_and_key():
    with pytest.raises(RuntimeError, match="--execute-paid-check"):
        require_paid_gate(False, {"OPENAI_API_KEY": "dummy"})
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        require_paid_gate(True, {})
    assert require_paid_gate(True, {"OPENAI_API_KEY": " dummy "}) == "dummy"


def test_payload_freezes_reasoning_structured_output_and_has_no_secret():
    config = load_config(ROOT / "configs" / "versioned_api_check.json")
    payload = build_payload(config, {"batch_size": 30}, 0, 1)
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["store"] is False
    assert "OPENAI_API_KEY" not in repr(payload)
