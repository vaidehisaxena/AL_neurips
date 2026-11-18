#!/usr/bin/env python3
"""Config-driven, file-backed campaign harness for a strategy controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_al.agent import (
    SYSTEM_PROMPT,
    AgentViewSpec,
    BatchAllocation,
    select_portfolio_batch,
    summarize_agent_state,
)
from agentic_al.campaign import (
    ExactTanimotoGP,
    _index_digest,
    _mask_digest,
    _metrics,
    deterministic_availability_mask,
    featurize,
    load_target,
    scaffolds,
)
from agentic_al.explanations import (
    EVIDENCE_METHOD,
    build_explanatory_evidence,
    canonical_digest,
    evidence_shuffle_seed_digest,
    shuffle_explanatory_evidence,
)


DEFAULT_CONFIG = Path("configs/agent_information_ablations.json")
REQUIRED_VARIANTS = {
    "full_reference",
    "minus_uncertainty",
    "minus_diversity",
    "minus_memory",
    "shuffled_evidence",
}
DEFAULT_AVAILABILITY_SALT = "agentic-al-availability-v1"
AVAILABILITY_ALGORITHM = "sha256-canonical-isomeric-smiles-u64-v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _initial_indices(pool_size: int, initial_size: int, seed: int) -> np.ndarray:
    """Return the paired initialization; variant names never enter this RNG."""
    if initial_size > pool_size:
        raise ValueError("Initial size exceeds the target library")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool_size, size=initial_size, replace=False))


def _validate_config(config: dict[str, Any]) -> None:
    required = (
        "protocol",
        "controller_model",
        "targets",
        "seeds",
        "initial_size",
        "cycles",
        "batch_size",
        "radius",
        "n_bits",
        "noise",
        "jitter",
        "prompt_version",
        "evidence",
        "variants",
    )
    for field in required:
        if field not in config:
            raise ValueError(f"Experiment config is missing {field!r}")
    if set(config["variants"]) != REQUIRED_VARIANTS:
        raise ValueError("Information-study variant names do not match the frozen protocol")
    if config["evidence"].get("method") != EVIDENCE_METHOD:
        raise ValueError("Unknown explanatory-evidence method")
    for name, raw in config["variants"].items():
        AgentViewSpec.from_dict({"name": name, **raw})
    active_variants = config.get("active_variants", list(config["variants"]))
    if not active_variants or not set(active_variants).issubset(config["variants"]):
        raise ValueError("active_variants must be a non-empty subset of variants")
    availability = config.get("availability", {})
    fraction = float(availability.get("fraction", 1.0))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("availability fraction must be in (0, 1]")
    if availability and not str(availability.get("salt", "")).strip():
        raise ValueError("availability salt is required when availability is configured")
    if availability and availability.get("algorithm") != AVAILABILITY_ALGORITHM:
        raise ValueError("Availability algorithm does not match the frozen implementation")
    if availability and availability.get("grandfather_initial") is not True:
        raise ValueError("Availability-constrained agent campaigns must grandfather initialization")
    if min(config["initial_size"], config["batch_size"], config["cycles"]) < 1:
        raise ValueError("Campaign sizes must be positive")


def load_config(path: Path) -> dict[str, Any]:
    config = _read(path)
    _validate_config(config)
    return config


def _assets(root: Path, session: dict[str, Any]):
    target = session["target"]
    frame = load_target(root, target)
    radius = int(session["radius"])
    n_bits = int(session["n_bits"])
    features = featurize(
        frame,
        root / "artifacts" / "feature_cache" / f"{target}_r{radius}_{n_bits}.npz",
        radius=radius,
        n_bits=n_bits,
    )
    return frame, features, scaffolds(frame)


def _view_spec(session: dict[str, Any]) -> AgentViewSpec:
    return AgentViewSpec.from_dict(session["view_spec"])


def _evidence_for_view(
    session: dict[str, Any], features: np.ndarray, observed: np.ndarray
) -> dict[str, Any] | None:
    spec = _view_spec(session)
    if spec.evidence_mode == "none":
        return None
    evidence_config = session["evidence_config"]
    evidence = build_explanatory_evidence(
        features[observed],
        np.asarray(session["observed_affinities"], dtype=float),
        top_k=int(evidence_config["top_k"]),
        min_support=int(evidence_config["min_support"]),
    )
    if spec.evidence_mode == "shuffled":
        evidence = shuffle_explanatory_evidence(
            evidence,
            target=session["target"],
            seed=int(session["seed"]),
            cycle=int(session["cycle"]) + 1,
            protocol_version=evidence_config["shuffle_protocol_version"],
        )
    return evidence


def _availability_settings(config: dict[str, Any]) -> tuple[float, str]:
    value = config.get("availability", {})
    return (
        float(value.get("fraction", 1.0)),
        str(value.get("salt", DEFAULT_AVAILABILITY_SALT)),
    )


def _availability(
    frame, session: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rebuild and verify the label-independent mask for a saved session."""
    fraction = float(session.get("availability_fraction", 1.0))
    salt = str(session.get("availability_salt", DEFAULT_AVAILABILITY_SALT))
    base = deterministic_availability_mask(frame["SMILES"].astype(str), fraction, salt)
    base_digest = _mask_digest(base)
    expected_digest = session.get("availability_mask_digest")
    if expected_digest is not None and base_digest != expected_digest:
        raise ValueError("Availability mask digest does not match the initialized session")
    initial = np.asarray(
        session.get("initial_indices", session["observed_indices"][: session["initial_size"]]),
        dtype=int,
    )
    effective = base.copy()
    effective[initial] = True
    provenance = {
        "availability_algorithm": AVAILABILITY_ALGORITHM,
        "availability_fraction": fraction,
        "availability_salt": salt,
        "availability_mask_digest": base_digest,
        "availability_base_count": int(base.sum()),
        "availability_effective_count": int(effective.sum()),
        "availability_excluded_fraction": float(1.0 - base.mean()),
        "initial_set_digest": _index_digest(initial),
    }
    return effective, provenance


def _available_pool(frame, session: dict[str, Any], observed: np.ndarray) -> np.ndarray:
    available, _ = _availability(frame, session)
    return np.setdiff1d(np.flatnonzero(available), observed, assume_unique=False)


def initialize(
    root: Path,
    session_path: Path,
    target: str,
    seed: int,
    *,
    config_path: Path | None = None,
    variant: str = "full_reference",
) -> dict[str, Any]:
    resolved_config = config_path or root / DEFAULT_CONFIG
    config = load_config(resolved_config)
    if target not in config["targets"]:
        raise ValueError(f"Target {target!r} is outside the frozen matrix")
    if seed not in config["seeds"]:
        raise ValueError(f"Seed {seed} is outside the frozen matrix")
    if variant not in config.get("active_variants", config["variants"]):
        raise ValueError(f"Unknown information variant: {variant}")

    frame = load_target(root, target)
    initial = _initial_indices(len(frame), int(config["initial_size"]), seed)
    availability_fraction, availability_salt = _availability_settings(config)
    base_available = deterministic_availability_mask(
        frame["SMILES"].astype(str), availability_fraction, availability_salt
    )
    effective_available = base_available.copy()
    effective_available[initial] = True
    endpoint_size = int(config["initial_size"]) + int(config["cycles"]) * int(config["batch_size"])
    if int(effective_available.sum()) < endpoint_size:
        raise ValueError("Availability mask cannot support the frozen endpoint budget")
    labels = frame["affinity"].to_numpy(dtype=float)
    scaffold_ids = scaffolds(frame)
    initial_metrics = _metrics(
        labels,
        frame["top_2p"].to_numpy(bool),
        frame["top_5p"].to_numpy(bool),
        scaffold_ids,
        initial.tolist(),
    )
    view_spec = AgentViewSpec.from_dict(
        {"name": variant, **config["variants"][variant]}
    )
    session: dict[str, Any] = {
        "protocol": config["protocol"],
        "prompt_version": config["prompt_version"],
        "controller_model": config["controller_model"],
        "system_prompt_sha256": _text_digest(SYSTEM_PROMPT),
        "config_sha256": canonical_digest(config),
        "variant": variant,
        "view_spec": asdict(view_spec),
        "evidence_config": config["evidence"],
        "target": target,
        "seed": int(seed),
        "cycle": 0,
        "cycles": int(config["cycles"]),
        "initial_size": int(config["initial_size"]),
        "batch_size": int(config["batch_size"]),
        "radius": int(config["radius"]),
        "n_bits": int(config["n_bits"]),
        "noise": float(config["noise"]),
        "jitter": float(config["jitter"]),
        "availability_algorithm": AVAILABILITY_ALGORITHM,
        "availability_fraction": availability_fraction,
        "availability_salt": availability_salt,
        "availability_mask_digest": _mask_digest(base_available),
        "availability_base_count": int(base_available.sum()),
        "availability_effective_count": int(effective_available.sum()),
        "availability_excluded_fraction": float(1.0 - base_available.mean()),
        "initial_set_digest": _index_digest(initial),
        "initial_indices_sha256": canonical_digest(initial.tolist()),
        "initial_indices": initial.tolist(),
        "observed_indices": initial.tolist(),
        "observed_affinities": labels[initial].tolist(),
        "history": [],
        "predecision_views": [],
        "trajectory": [
            {
                "cycle": 0,
                "measured": len(initial),
                "availability_mask_digest": _mask_digest(base_available),
                "initial_set_digest": _index_digest(initial),
                **initial_metrics,
            }
        ],
    }
    _write(session_path, session)
    return agent_view(root, session)


def agent_view(root: Path, session: dict[str, Any]) -> dict[str, Any]:
    if session["cycle"] >= session["cycles"]:
        return {
            "status": "complete",
            "target": session["target"],
            "seed": session["seed"],
        }
    frame, features, scaffold_ids = _assets(root, session)
    observed = np.asarray(session["observed_indices"], dtype=int)
    pool = _available_pool(frame, session, observed)
    gp = ExactTanimotoGP(session["noise"], session["jitter"]).fit(
        features[observed], np.asarray(session["observed_affinities"])
    )
    mean, std = gp.predict(features[pool])
    summary = summarize_agent_state(
        target=session["target"],
        cycle=session["cycle"] + 1,
        cycles=session["cycles"],
        batch_size=session["batch_size"],
        observed_affinities=np.asarray(session["observed_affinities"]),
        predicted_mean=mean,
        predicted_std=std,
        observed_scaffolds=scaffold_ids[observed],
        pool_scaffolds=scaffold_ids[pool],
        history=session["history"],
        view_spec=_view_spec(session),
        explanatory_evidence=_evidence_for_view(session, features, observed),
    )
    summary["instruction"] = (
        "Choose non-negative integer n_exploit, n_uncertain, n_diverse summing to "
        f"{session['batch_size']}, plus a rationale and falsifiable expected_outcome."
    )
    return summary


def apply_action(
    root: Path, session_path: Path, action_path: Path
) -> dict[str, Any]:
    session = _read(session_path)
    if session["cycle"] >= session["cycles"]:
        raise ValueError("Campaign is already complete")
    predecision_view = agent_view(root, session)
    predecision_view_sha256 = canonical_digest(predecision_view)
    evidence_sha256 = predecision_view.get("explanatory_evidence", {}).get(
        "evidence_sha256"
    )
    shuffle_seed_sha256 = None
    if session["view_spec"]["evidence_mode"] == "shuffled":
        shuffle_seed_sha256 = evidence_shuffle_seed_digest(
            target=session["target"],
            seed=session["seed"],
            cycle=session["cycle"] + 1,
            protocol_version=session["evidence_config"]["shuffle_protocol_version"],
        )

    raw = _read(action_path)
    action = BatchAllocation(**raw).validate(session["batch_size"])
    frame, features, scaffold_ids = _assets(root, session)
    labels = frame["affinity"].to_numpy(dtype=float)
    observed = np.asarray(session["observed_indices"], dtype=int)
    pool = _available_pool(frame, session, observed)
    _, availability_provenance = _availability(frame, session)
    if len(pool) < session["batch_size"]:
        raise ValueError("Availability mask cannot supply the next assay batch")
    gp = ExactTanimotoGP(session["noise"], session["jitter"]).fit(
        features[observed], np.asarray(session["observed_affinities"])
    )
    mean, std = gp.predict(features[pool])
    chosen = select_portfolio_batch(action, pool, mean, std, features, observed)
    if np.intersect1d(chosen, observed).size:
        raise ValueError("A selected compound was already observed")

    before_best = max(session["observed_affinities"])
    revealed = labels[chosen]
    selected_new_scaffolds = len(
        set(scaffold_ids[chosen].tolist()).difference(scaffold_ids[observed].tolist())
    )
    session["observed_indices"].extend(chosen.tolist())
    session["observed_affinities"].extend(revealed.tolist())
    session["cycle"] += 1
    metrics = _metrics(
        labels,
        frame["top_2p"].to_numpy(bool),
        frame["top_5p"].to_numpy(bool),
        scaffold_ids,
        session["observed_indices"],
    )
    decision = {
        "protocol": session["protocol"],
        "variant": session["variant"],
        "target": session["target"],
        "seed": session["seed"],
        "cycle": session["cycle"],
        **action.to_dict(),
        "selected_indices": chosen.tolist(),
        "selected_predicted_mean": float(mean[np.isin(pool, chosen)].mean()),
        "selected_predicted_std": float(std[np.isin(pool, chosen)].mean()),
        "selected_new_scaffolds": int(selected_new_scaffolds),
        "batch_affinity_mean": float(revealed.mean()),
        "batch_affinity_max": float(revealed.max()),
        "observed_best_before": float(before_best),
        "observed_best_after": float(metrics["best_affinity"]),
        "improvement": float(metrics["best_affinity"] - before_best),
        "incumbent_improved": bool(metrics["best_affinity"] > before_best),
        "config_sha256": session["config_sha256"],
        "system_prompt_sha256": session["system_prompt_sha256"],
        "initial_indices_sha256": session["initial_indices_sha256"],
        **availability_provenance,
        "predecision_view_sha256": predecision_view_sha256,
        "evidence_sha256": evidence_sha256,
        "evidence_shuffle_seed_sha256": shuffle_seed_sha256,
    }
    session["predecision_views"].append(
        {
            "cycle": session["cycle"],
            "view_sha256": predecision_view_sha256,
            "availability_mask_digest": availability_provenance["availability_mask_digest"],
            "initial_set_digest": availability_provenance["initial_set_digest"],
            "evidence_sha256": evidence_sha256,
            "evidence_shuffle_seed_sha256": shuffle_seed_sha256,
            "view": predecision_view,
        }
    )
    session["history"].append(decision)
    session["trajectory"].append(
        {
            "cycle": session["cycle"],
            "measured": len(session["observed_indices"]),
            "availability_mask_digest": availability_provenance["availability_mask_digest"],
            "initial_set_digest": availability_provenance["initial_set_digest"],
            **metrics,
        }
    )
    _write(session_path, session)
    return agent_view(root, session)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "view", "act", "export"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--variant", choices=sorted(REQUIRED_VARIANTS), default="full_reference"
    )
    parser.add_argument("--target", choices=("D2R", "Mpro", "TYK2", "USP7"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--action", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        if args.target is None or args.seed is None:
            parser.error("init requires --target and --seed")
        result = initialize(
            args.root,
            args.session,
            args.target,
            args.seed,
            config_path=args.config,
            variant=args.variant,
        )
    elif args.command == "view":
        result = agent_view(args.root, _read(args.session))
    elif args.command == "act":
        if args.action is None:
            parser.error("act requires --action")
        result = apply_action(args.root, args.session, args.action)
    else:
        session = _read(args.session)
        if session["cycle"] != session["cycles"]:
            raise ValueError("Cannot export an incomplete campaign")
        result = session
    if args.output:
        _write(args.output, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
