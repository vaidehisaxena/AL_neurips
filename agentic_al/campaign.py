from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


_AVAILABILITY_CACHE: dict[tuple[str, float, str], np.ndarray] = {}
_SCAFFOLD_CACHE: dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class CampaignConfig:
    target: str
    policy: str
    seed: int
    initial_size: int = 60
    batch_size: int = 30
    cycles: int = 10
    radius: int = 4
    n_bits: int = 4096
    noise: float = 0.05
    jitter: float = 1e-6
    availability_fraction: float = 1.0
    availability_salt: str = "agentic-al-availability-v1"


@dataclass(frozen=True)
class CampaignState:
    cycle: int
    observed_indices: tuple[int, ...]
    observed_affinities: tuple[float, ...]
    remaining_budget: int
    last_improvement: float


class HiddenAssay:
    """Owns hidden labels and reveals only explicitly requested indices."""

    def __init__(self, labels: np.ndarray):
        self.__labels = np.asarray(labels, dtype=np.float64).copy()
        self.__revealed: set[int] = set()

    def reveal(self, indices: Iterable[int]) -> np.ndarray:
        requested = np.asarray(list(indices), dtype=int)
        if requested.ndim != 1 or len(set(requested.tolist())) != len(requested):
            raise ValueError("Assay requests must contain unique one-dimensional indices")
        if any(index < 0 or index >= len(self.__labels) for index in requested):
            raise IndexError("Assay request is outside the candidate pool")
        overlap = self.__revealed.intersection(requested.tolist())
        if overlap:
            raise ValueError(f"Compounds already assayed: {sorted(overlap)}")
        self.__revealed.update(requested.tolist())
        return self.__labels[requested].copy()


class ExactTanimotoGP:
    """Small exact GP with a fixed Tanimoto kernel and deterministic solves."""

    def __init__(self, noise: float = 0.05, jitter: float = 1e-6):
        self.noise = float(noise)
        self.jitter = float(jitter)

    @staticmethod
    def kernel(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        a = np.asarray(x1, dtype=np.float32)
        b = np.asarray(x2, dtype=np.float32)
        dot = a @ b.T
        denom = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - dot
        return np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ExactTanimotoGP":
        self.x_train = np.asarray(x, dtype=np.uint8)
        y = np.asarray(y, dtype=np.float64)
        self.y_mean = float(y.mean())
        self.y_scale = float(y.std()) or 1.0
        centered = (y - self.y_mean) / self.y_scale
        covariance = self.kernel(self.x_train, self.x_train).astype(np.float64)
        covariance.flat[:: len(covariance) + 1] += self.noise**2 + self.jitter
        self.cholesky = np.linalg.cholesky(covariance)
        self.alpha = np.linalg.solve(
            self.cholesky.T, np.linalg.solve(self.cholesky, centered)
        )
        return self

    def predict(self, x: np.ndarray, chunk_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
        means: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        x = np.asarray(x, dtype=np.uint8)
        for start in range(0, len(x), chunk_size):
            cross = self.kernel(x[start : start + chunk_size], self.x_train).astype(np.float64)
            mean = cross @ self.alpha
            solved = np.linalg.solve(self.cholesky, cross.T)
            variance = np.maximum(1.0 - np.square(solved).sum(axis=0), self.jitter)
            means.append(self.y_mean + self.y_scale * mean)
            stds.append(self.y_scale * np.sqrt(variance + self.noise**2))
        return np.concatenate(means), np.concatenate(stds)


def deterministic_availability_mask(
    smiles: Iterable[str],
    fraction: float = 0.8,
    salt: str = "agentic-al-availability-v1",
) -> np.ndarray:
    """Return a label-independent mask hashed from canonical isomeric SMILES."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("availability fraction must be in (0, 1]")
    raw_values = tuple(map(str, smiles))
    source_digest = hashlib.sha256("\0".join(raw_values).encode("utf-8")).hexdigest()
    cache_key = (source_digest, float(fraction), str(salt))
    if cache_key in _AVAILABILITY_CACHE:
        return _AVAILABILITY_CACHE[cache_key].copy()
    threshold = int(fraction * (1 << 64))
    available = []
    for index, value in enumerate(raw_values):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"Invalid SMILES at availability row {index}: {value}")
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        digest = hashlib.sha256(f"{salt}\0{canonical}".encode("utf-8")).digest()
        available.append(int.from_bytes(digest[:8], "big") < threshold)
    result = np.asarray(available, dtype=bool)
    _AVAILABILITY_CACHE[cache_key] = result
    return result.copy()


def _index_digest(indices: Iterable[int]) -> str:
    ordered = sorted(map(int, indices))
    payload = ",".join(map(str, ordered)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _mask_digest(mask: np.ndarray) -> str:
    values = np.asarray(mask, dtype=bool)
    payload = len(values).to_bytes(8, "big") + np.packbits(values, bitorder="little").tobytes()
    return hashlib.sha256(payload).hexdigest()


def dataset_paths(root: Path) -> dict[str, Path]:
    base = root / "artifacts" / "zenodo" / "17935028"
    return {
        "D2R": base / "composite_d2r.csv",
        "Mpro": base / "mpro_sorted.csv",
        "TYK2": base / "TYK2_sorted.csv",
        "USP7": base / "USP7_sorted.csv",
    }


def load_target(root: Path, target: str) -> pd.DataFrame:
    paths = dataset_paths(root)
    if target not in paths:
        raise KeyError(f"Unknown target {target!r}; choose from {sorted(paths)}")
    frame = pd.read_csv(paths[target]).loc[:, lambda x: ~x.columns.str.startswith("Unnamed")]
    required = {"SMILES", "affinity", "top_2p", "top_5p", "Target"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{target} is missing required columns: {sorted(missing)}")
    if frame["SMILES"].isna().any() or frame["SMILES"].duplicated().any():
        raise ValueError(f"{target} has missing or duplicate SMILES")
    frame = frame.reset_index(drop=True)
    frame["affinity"] = pd.to_numeric(frame["affinity"], errors="raise")
    for column in ("top_2p", "top_5p"):
        frame[column] = frame[column].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if frame[column].isna().any():
            raise ValueError(f"{target}.{column} contains non-boolean values")
    return frame


def featurize(frame: pd.DataFrame, cache_path: Path, radius: int, n_bits: int) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        features = cached["features"]
        if features.shape == (len(frame), n_bits):
            return features
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    features = np.zeros((len(frame), n_bits), dtype=np.uint8)
    for index, smiles in enumerate(frame["SMILES"]):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES at row {index}: {smiles}")
        features[index] = generator.GetFingerprintAsNumPy(molecule)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features)
    return features


def scaffolds(frame: pd.DataFrame) -> np.ndarray:
    raw_values = tuple(frame["SMILES"].astype(str))
    source_digest = hashlib.sha256("\0".join(raw_values).encode("utf-8")).hexdigest()
    if source_digest in _SCAFFOLD_CACHE:
        return _SCAFFOLD_CACHE[source_digest].copy()
    values = []
    for smiles in raw_values:
        scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles)
        values.append(scaffold or smiles)
    result = np.asarray(values, dtype=object)
    _SCAFFOLD_CACHE[source_digest] = result
    return result.copy()


def _zscore(values: np.ndarray) -> np.ndarray:
    scale = values.std()
    return (values - values.mean()) / (scale if scale > 0 else 1.0)


def _method_for_cycle(policy: str, cycle: int, last_improvement: float) -> str:
    if policy in {"random", "explore", "exploit", "balanced"}:
        return policy
    if policy == "explore-heavy":
        return "explore" if cycle < 7 else "exploit"
    if policy == "exploit-heavy":
        return "explore" if cycle < 3 else "exploit"
    if policy == "alternating":
        return "explore" if cycle % 2 == 0 else "exploit"
    if policy == "sandwich":
        return "explore" if cycle < 2 or cycle >= 8 else "exploit"
    if policy == "heuristic":
        return "explore" if last_improvement <= 0 else "balanced"
    raise ValueError(f"Unknown policy: {policy}")


def select_batch(
    method: str,
    pool: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if batch_size > len(pool):
        raise ValueError("Batch exceeds the remaining candidate pool")
    if method == "random":
        return np.sort(rng.choice(pool, size=batch_size, replace=False))
    if method == "explore":
        score = std
    elif method == "exploit":
        score = mean
    elif method == "balanced":
        score = 0.5 * _zscore(mean) + 0.5 * _zscore(std)
    else:
        raise ValueError(f"Unknown acquisition method: {method}")
    order = np.lexsort((pool, -score))
    return pool[order[:batch_size]]


def _metrics(
    labels: np.ndarray,
    top_2p: np.ndarray,
    top_5p: np.ndarray,
    scaffold_ids: np.ndarray,
    observed: list[int],
) -> dict[str, float | int]:
    selected = np.asarray(observed, dtype=int)
    selected_labels = labels[selected]
    top_1_cutoff = np.quantile(labels, 0.99)
    discovered_1 = int(np.sum(selected_labels >= top_1_cutoff))
    selected_top_5 = selected[top_5p[selected]]
    return {
        "best_affinity": float(selected_labels.max()),
        "simple_regret": float(labels.max() - selected_labels.max()),
        "top_1p_recall": float(discovered_1 / max(1, np.sum(labels >= top_1_cutoff))),
        "top_2p_recall": float(top_2p[selected].sum() / max(1, top_2p.sum())),
        "top_5p_recall": float(top_5p[selected].sum() / max(1, top_5p.sum())),
        "unique_top_5p_scaffolds": int(len(set(scaffold_ids[selected_top_5].tolist()))),
    }


def run_campaign(root: Path, config: CampaignConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_target(root, config.target)
    cache = root / "artifacts" / "feature_cache" / f"{config.target}_r{config.radius}_{config.n_bits}.npz"
    features = featurize(frame, cache, config.radius, config.n_bits)
    scaffold_ids = scaffolds(frame)
    labels = frame["affinity"].to_numpy(dtype=np.float64)
    top_2p = frame["top_2p"].to_numpy(dtype=bool)
    top_5p = frame["top_5p"].to_numpy(dtype=bool)
    assay = HiddenAssay(labels)
    rng = np.random.default_rng(config.seed)
    initial = np.sort(rng.choice(len(frame), size=config.initial_size, replace=False))
    base_available = deterministic_availability_mask(
        frame["SMILES"].astype(str), config.availability_fraction, config.availability_salt
    )
    available = base_available.copy()
    # Preserve the paired initialization; availability constrains later choices only.
    available[initial] = True
    if int(available.sum()) < config.initial_size + config.batch_size * config.cycles:
        raise ValueError("Availability mask cannot support the frozen endpoint budget")
    observed = initial.tolist()
    observed_y = assay.reveal(initial).tolist()
    availability_provenance = {
        "availability_algorithm": "sha256-canonical-isomeric-smiles-u64-v1",
        "availability_mask_digest": _mask_digest(base_available),
        "availability_base_count": int(base_available.sum()),
        "availability_effective_count": int(available.sum()),
        "availability_excluded_fraction": float(1.0 - base_available.mean()),
        "initial_set_digest": _index_digest(initial),
    }
    trajectory: list[dict] = []
    decisions: list[dict] = []
    initial_metrics = _metrics(labels, top_2p, top_5p, scaffold_ids, observed)
    trajectory.append({**asdict(config), **availability_provenance, "cycle": 0, "method": "initial-random", "measured": len(observed), **initial_metrics})
    previous_best = initial_metrics["best_affinity"]
    last_improvement = float("inf")

    for cycle in range(config.cycles):
        gp = ExactTanimotoGP(config.noise, config.jitter).fit(features[observed], np.asarray(observed_y))
        pool = np.setdiff1d(
            np.flatnonzero(available), np.asarray(observed), assume_unique=False
        )
        mean, std = gp.predict(features[pool])
        method = _method_for_cycle(config.policy, cycle, last_improvement)
        chosen = select_batch(method, pool, mean, std, min(config.batch_size, len(pool)), rng)
        state = CampaignState(
            cycle=cycle + 1,
            observed_indices=tuple(observed),
            observed_affinities=tuple(observed_y),
            remaining_budget=(config.cycles - cycle) * config.batch_size,
            last_improvement=last_improvement,
        )
        decisions.append({
            "target": config.target,
            "policy": config.policy,
            "seed": config.seed,
            "initial_size": config.initial_size,
            "batch_size": config.batch_size,
            "cycles": config.cycles,
            "availability_fraction": config.availability_fraction,
            "availability_salt": config.availability_salt,
            **availability_provenance,
            "cycle": cycle + 1,
            "method": method,
            "selected_indices": ";".join(map(str, chosen.tolist())),
            "observed_count_before": len(state.observed_indices),
            "remaining_budget_before": state.remaining_budget,
            "last_improvement": state.last_improvement,
            "mean_score_selected": float(mean[np.isin(pool, chosen)].mean()),
            "mean_uncertainty_selected": float(std[np.isin(pool, chosen)].mean()),
        })
        revealed = assay.reveal(chosen)
        observed.extend(chosen.tolist())
        observed_y.extend(revealed.tolist())
        metrics = _metrics(labels, top_2p, top_5p, scaffold_ids, observed)
        last_improvement = float(metrics["best_affinity"] - previous_best)
        previous_best = metrics["best_affinity"]
        trajectory.append({**asdict(config), **availability_provenance, "cycle": cycle + 1, "method": method, "measured": len(observed), **metrics})

    return pd.DataFrame(trajectory), pd.DataFrame(decisions)
