from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

import numpy as np


EVIDENCE_METHOD = "observed-fingerprint-enrichment-v1"
SHUFFLE_METHOD = "within-packet-bit-identity-derangement-v1"


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_explanatory_evidence(
    observed_features: np.ndarray,
    observed_affinities: np.ndarray,
    *,
    top_k: int = 10,
    min_support: int = 3,
) -> dict[str, Any]:
    """Rank fingerprint bits using labels from already observed compounds only.

    The public interface deliberately accepts neither pool labels, top-active flags,
    nor a full dataset frame. Scores are absolute standardized differences between
    observed compounds with and without each sufficiently supported bit.
    """
    features = np.asarray(observed_features)
    affinities = np.asarray(observed_affinities, dtype=float)
    if features.ndim != 2 or affinities.ndim != 1 or len(features) != len(affinities):
        raise ValueError("Observed features and affinities must be aligned")
    if len(affinities) < 2 or not np.isfinite(affinities).all():
        raise ValueError("At least two finite observed affinities are required")
    if isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if isinstance(min_support, bool) or min_support < 1:
        raise ValueError("min_support must be a positive integer")

    present = features.astype(bool, copy=False)
    support = present.sum(axis=0).astype(int)
    eligible = (support >= min_support) & ((len(features) - support) >= min_support)
    feature_indices = np.flatnonzero(eligible)
    if not len(feature_indices):
        raise ValueError("No fingerprint bits satisfy the observed-only support threshold")

    label_scale = float(affinities.std()) or 1.0
    rows: list[dict[str, Any]] = []
    for bit_index in feature_indices.tolist():
        mask = present[:, bit_index]
        difference = float(affinities[mask].mean() - affinities[~mask].mean())
        standardized = difference / label_scale
        rows.append(
            {
                "bit_index": int(bit_index),
                "support": int(support[bit_index]),
                "prevalence": float(support[bit_index] / len(features)),
                "affinity_difference": difference,
                "standardized_enrichment": float(standardized),
                "score": float(abs(standardized)),
                "sign": 1 if standardized > 0 else (-1 if standardized < 0 else 0),
            }
        )
    rows.sort(key=lambda row: (-row["score"], row["bit_index"]))
    selected = rows[: min(top_k, len(rows))]
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank

    packet = {
        "method": EVIDENCE_METHOD,
        "observed_count": int(len(features)),
        "eligible_bit_count": int(len(rows)),
        "features": selected,
    }
    packet["evidence_sha256"] = canonical_digest(packet)
    return packet


def shuffle_explanatory_evidence(
    evidence: dict[str, Any],
    *,
    target: str,
    seed: int,
    cycle: int,
    protocol_version: str,
) -> dict[str, Any]:
    """Deterministically derange bit identities without touching evidence values."""
    shuffled = deepcopy(evidence)
    rows = shuffled.get("features")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("At least two evidence rows are required for a non-identity shuffle")
    identities = [int(row["bit_index"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Evidence bit identities must be unique")

    seed_digest = evidence_shuffle_seed_digest(
        target=target,
        seed=seed,
        cycle=cycle,
        protocol_version=protocol_version,
    )
    rng = np.random.default_rng(int(seed_digest[:16], 16))
    shift = int(rng.integers(1, len(identities)))
    permuted = identities[shift:] + identities[:shift]
    if any(original == replacement for original, replacement in zip(identities, permuted)):
        raise AssertionError("Evidence shuffle must be a derangement")
    for row, replacement in zip(rows, permuted):
        row["bit_index"] = replacement

    shuffled["evidence_sha256"] = canonical_digest(
        {key: value for key, value in shuffled.items() if key != "evidence_sha256"}
    )
    return shuffled


def evidence_shuffle_seed_digest(
    *, target: str, seed: int, cycle: int, protocol_version: str
) -> str:
    """Return auditable shuffle provenance without exposing it in the agent view."""
    seed_material = {
        "protocol_version": protocol_version,
        "target": target,
        "seed": int(seed),
        "cycle": int(cycle),
        "method": SHUFFLE_METHOD,
    }
    return canonical_digest(seed_material)
