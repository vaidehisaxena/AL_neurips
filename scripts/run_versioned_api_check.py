#!/usr/bin/env python3
"""Run the explicitly authorized, paid 20-call versioned Responses API check."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping


FROZEN_MODEL = "gpt-5.6-sol"
FROZEN_TARGET = "D2R"
FROZEN_SEEDS = [0, 1]
FROZEN_CYCLES = list(range(1, 11))


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    expected = {
        "model": FROZEN_MODEL,
        "target": FROZEN_TARGET,
        "seeds": FROZEN_SEEDS,
        "cycles": FROZEN_CYCLES,
        "expected_calls": 20,
        "reasoning_effort": "medium",
        "store": False,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"Frozen API-check field {field!r} must equal {value!r}")
    output_format = config.get("output_format", {})
    if output_format.get("type") != "json_schema" or output_format.get("strict") is not True:
        raise ValueError("The API check requires strict Structured Outputs")
    manifest = build_call_manifest(config)
    if len(manifest) != config["expected_calls"] or len(set(manifest)) != len(manifest):
        raise ValueError("Frozen API-check matrix must contain exactly 20 unique calls")
    return config


def build_call_manifest(config: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    return [
        (str(config["target"]), int(seed), int(cycle))
        for seed in config["seeds"]
        for cycle in config["cycles"]
    ]


def require_paid_gate(execute_paid_check: bool, environ: Mapping[str, str]) -> str:
    if not execute_paid_check:
        raise RuntimeError("Refusing paid API call without explicit --execute-paid-check")
    api_key = environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Refusing paid API call because OPENAI_API_KEY is not set")
    return api_key


def build_payload(config: Mapping[str, Any], state: Any, seed: int, cycle: int) -> dict[str, Any]:
    return {
        "model": config["model"],
        "instructions": config["instructions"],
        "input": json.dumps(state, sort_keys=True, separators=(",", ":")),
        "reasoning": {"effort": config["reasoning_effort"]},
        "text": {"format": config["output_format"]},
        "store": config["store"],
        "max_output_tokens": config["max_output_tokens"],
        "metadata": {
            "protocol": config["protocol"],
            "target": config["target"],
            "seed": str(seed),
            "cycle": str(cycle),
        },
    }


def _request(endpoint: str, api_key: str, payload: Mapping[str, Any], timeout: int):
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read())
        request_id = response.headers.get("x-request-id")
    return value, request_id, time.monotonic() - started, hashlib.sha256(body).hexdigest()


def _output_text(response: Mapping[str, Any]) -> str:
    for item in response.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return str(content["text"])
    raise ValueError("Response did not contain output_text")


def _validate_allocation(value: Mapping[str, Any]) -> None:
    required = {"n_exploit", "n_uncertain", "n_diverse", "rationale", "expected_outcome"}
    if set(value) != required:
        raise ValueError("Structured allocation fields do not match the frozen schema")
    counts = [value[name] for name in ("n_exploit", "n_uncertain", "n_diverse")]
    if any(isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 30 for count in counts):
        raise ValueError("Structured allocation counts are invalid")
    if sum(counts) != 30:
        raise ValueError("Structured allocation counts must sum to 30")
    if not all(isinstance(value[name], str) and value[name].strip() for name in ("rationale", "expected_outcome")):
        raise ValueError("Structured allocation text fields cannot be blank")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "versioned_api_check.json",
    )
    parser.add_argument("--execute-paid-check", action="store_true")
    args = parser.parse_args()

    # The authorization gate intentionally precedes all filesystem output and network work.
    api_key = require_paid_gate(args.execute_paid_check, os.environ)
    config = load_config(args.config)
    root = Path(__file__).resolve().parents[1]
    output_path = root / config["output_jsonl"]
    manifest = build_call_manifest(config)
    state_paths = {
        (target, seed, cycle): root / config["state_path_template"].format(seed=seed, cycle=cycle)
        for target, seed, cycle in manifest
    }
    missing = [str(path) for path in state_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Refusing paid run: missing frozen inputs: {missing}")

    completed: set[tuple[str, int, int]] = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("protocol") != config["protocol"] or row.get("requested_model") != config["model"]:
                raise RuntimeError("Refusing resume: output contains a different protocol or model")
            key = (str(row["target"]), int(row["seed"]), int(row["cycle"]))
            if key in completed or key not in set(manifest):
                raise RuntimeError("Refusing resume: output has duplicate or foreign call records")
            # Invalid structured output still consumed a paid call and must never be repeated.
            completed.add(key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for target, seed, cycle in manifest:
        if (target, seed, cycle) in completed:
            continue
        state_path = state_paths[(target, seed, cycle)]
        state_bytes = state_path.read_bytes()
        state = json.loads(state_bytes)
        payload = build_payload(config, state, seed, cycle)
        response, request_id, latency, request_hash = _request(
            config["endpoint"], api_key, payload, int(config["timeout_seconds"])
        )
        allocation = json.loads(_output_text(response))
        status = "completed"
        try:
            _validate_allocation(allocation)
        except (TypeError, ValueError):
            status = "invalid_structured_output"
        usage = response.get("usage", {})
        record = {
                "protocol": config["protocol"],
                "target": target,
                "seed": seed,
                "cycle": cycle,
                "requested_model": config["model"],
                "returned_model": response.get("model"),
                "response_id": response.get("id"),
                "x_request_id": request_id,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "usage": usage,
                "latency_seconds": latency,
                "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
                "request_sha256": request_hash,
                "structured_output": allocation,
                "status": status,
                "retry_count": 0,
            }
        with output_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if status != "completed":
            raise ValueError(f"Paid response for seed={seed}, cycle={cycle} failed schema validation")
    print(f"recorded {len(manifest)} paid-call outcomes in {output_path}")


if __name__ == "__main__":
    main()
