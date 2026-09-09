#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


TARGETS = ("D2R", "Mpro", "TYK2", "USP7")
VARIANTS = ("agent_gp_chem", "full_agent", "full_minus_shap", "full_shuffled_shap", "full_minus_memory")


@dataclass(frozen=True)
class PlannedCall:
    target: str
    variant: str
    seed: int
    cycle: int
    model: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/agent_dry_run"))
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--input-tokens", type=int, default=1800)
    parser.add_argument("--output-tokens", type=int, default=180)
    parser.add_argument("--input-price-per-million", type=float, default=2.50)
    parser.add_argument("--output-price-per-million", type=float, default=15.00)
    parser.add_argument("--baseline-runtime-seconds", type=float, default=380.53)
    args = parser.parse_args()

    calls = [
        PlannedCall(target, variant, seed, cycle, args.model)
        for target in TARGETS
        for variant in VARIANTS
        for seed in range(10)
        for cycle in range(1, 11)
    ]
    count = len(calls)
    input_tokens = count * args.input_tokens
    output_tokens = count * args.output_tokens
    cost = (
        input_tokens / 1_000_000 * args.input_price_per_million
        + output_tokens / 1_000_000 * args.output_price_per_million
    )
    development_calls = 1 * 1 * 2 * 10
    report = {
        "status": "DRY_RUN_ONLY_NO_API_CALLS",
        "model": args.model,
        "targets": list(TARGETS),
        "variants": list(VARIANTS),
        "seeds": 10,
        "cycles": 10,
        "campaigns": len(TARGETS) * len(VARIANTS) * 10,
        "gp_fits": count,
        "llm_calls": count,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_api_cost_usd": round(cost, 2),
        "development_gate": {
            "target_count": 1,
            "variant_count": 1,
            "seeds": 2,
            "llm_calls": development_calls,
            "estimated_api_cost_usd": round(cost * development_calls / count, 2),
        },
        "compute_runtime_lower_bound_seconds": round(
            args.baseline_runtime_seconds * count / 3600, 2
        ),
        "assumptions": {
            "input_tokens_per_call": args.input_tokens,
            "output_tokens_per_call": args.output_tokens,
            "input_price_usd_per_million": args.input_price_per_million,
            "output_price_usd_per_million": args.output_price_per_million,
            "pricing_checked_on": "2026-09-09",
            "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5.4",
            "excludes_retries": True,
            "excludes_prompt_cache_discount": True,
            "runtime_excludes_network_latency": True,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.jsonl").write_text(
        "".join(json.dumps(asdict(call), sort_keys=True) + "\n" for call in calls)
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

