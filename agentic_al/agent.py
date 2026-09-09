from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from agentic_al.campaign import ExactTanimotoGP


SYSTEM_PROMPT = """You are the strategy controller for a virtual ligand-discovery campaign.
Choose how many compounds in the next assay batch should come from three deterministic
candidate queues: exploit (highest GP mean), uncertain (highest GP uncertainty), and
diverse (farthest from assayed and already selected compounds). Use only the supplied
summary. Never infer or request hidden labels. Return exactly one JSON object matching
the schema. The three non-negative integer allocations must sum to batch_size. State a
short, falsifiable expectation that can be checked after the assay."""


@dataclass(frozen=True)
class BatchAllocation:
    n_exploit: int
    n_uncertain: int
    n_diverse: int
    rationale: str
    expected_outcome: str

    def validate(self, batch_size: int) -> "BatchAllocation":
        counts = (self.n_exploit, self.n_uncertain, self.n_diverse)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise TypeError("Allocation counts must be integers")
        if any(value < 0 for value in counts):
            raise ValueError("Allocation counts must be non-negative")
        if sum(counts) != batch_size:
            raise ValueError(f"Allocation must sum to batch_size={batch_size}")
        if not self.rationale.strip() or not self.expected_outcome.strip():
            raise ValueError("Rationale and expected outcome are required")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def allocation_json_schema(batch_size: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "n_exploit": {"type": "integer", "minimum": 0, "maximum": batch_size},
            "n_uncertain": {"type": "integer", "minimum": 0, "maximum": batch_size},
            "n_diverse": {"type": "integer", "minimum": 0, "maximum": batch_size},
            "rationale": {"type": "string", "minLength": 1, "maxLength": 600},
            "expected_outcome": {"type": "string", "minLength": 1, "maxLength": 300},
        },
        "required": [
            "n_exploit",
            "n_uncertain",
            "n_diverse",
            "rationale",
            "expected_outcome",
        ],
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(np.asarray(values, dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0])
    return dict(zip(("min", "q25", "median", "q75", "max"), map(float, q)))


def summarize_agent_state(
    *,
    target: str,
    cycle: int,
    cycles: int,
    batch_size: int,
    observed_affinities: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    observed_scaffolds: np.ndarray,
    pool_scaffolds: np.ndarray,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create the only state passed to an agent; hidden pool labels are not accepted."""
    observed = np.asarray(observed_affinities, dtype=float)
    means = np.asarray(predicted_mean, dtype=float)
    stds = np.asarray(predicted_std, dtype=float)
    if not len(observed) or len(means) != len(stds):
        raise ValueError("Observed labels and aligned pool predictions are required")
    return {
        "target": target,
        "cycle": int(cycle),
        "total_cycles": int(cycles),
        "batch_size": int(batch_size),
        "remaining_assay_budget": int((cycles - cycle + 1) * batch_size),
        "observed_count": int(len(observed)),
        "pool_count": int(len(means)),
        "observed_affinity": {
            "best": float(observed.max()),
            "mean": float(observed.mean()),
            "quantiles": _quantiles(observed),
        },
        "gp_predicted_mean": _quantiles(means),
        "gp_predicted_std": _quantiles(stds),
        "chemical_space": {
            "observed_unique_scaffolds": int(len(set(observed_scaffolds.tolist()))),
            "pool_unique_scaffolds": int(len(set(pool_scaffolds.tolist()))),
        },
        "history": list(history or []),
    }


def _rank(score: np.ndarray, pool: np.ndarray) -> np.ndarray:
    return np.lexsort((pool, -np.asarray(score, dtype=float)))


def select_portfolio_batch(
    allocation: BatchAllocation,
    pool: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    features: np.ndarray,
    observed_indices: np.ndarray,
) -> np.ndarray:
    """Resolve an allocation into unique compounds with deterministic tie-breaking."""
    pool = np.asarray(pool, dtype=int)
    observed_indices = np.asarray(observed_indices, dtype=int)
    allocation.validate(allocation.n_exploit + allocation.n_uncertain + allocation.n_diverse)
    if len(mean) != len(pool) or len(std) != len(pool):
        raise ValueError("Pool predictions must align with pool indices")
    if sum((allocation.n_exploit, allocation.n_uncertain, allocation.n_diverse)) > len(pool):
        raise ValueError("Allocation exceeds remaining pool")

    selected: list[int] = []

    def add_ranked(order: np.ndarray) -> None:
        for position in order:
            candidate = int(pool[position])
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= target_length:
                break

    target_length = allocation.n_exploit
    add_ranked(_rank(mean, pool))
    target_length += allocation.n_uncertain
    add_ranked(_rank(std, pool))

    target_length += allocation.n_diverse
    while len(selected) < target_length:
        remaining = np.asarray([index for index in pool if index not in selected], dtype=int)
        reference = np.concatenate([observed_indices, np.asarray(selected, dtype=int)])
        if len(reference):
            similarity = ExactTanimotoGP.kernel(features[remaining], features[reference]).max(axis=1)
        else:
            similarity = np.zeros(len(remaining), dtype=float)
        choice = np.lexsort((remaining, similarity))[0]
        selected.append(int(remaining[choice]))

    return np.asarray(selected, dtype=int)
