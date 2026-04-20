"""Prototype construction for normal-regime stabilisation.

Purpose
-------
Build M prototypes in latent space from D_normal embeddings.
Each prototype represents a cluster of similar normal market states.
Prototype weights = empirical mass of assigned observations.

Role
----
NOT generative — we do NOT sample new normal scenarios.
The prototype layer reduces redundancy and stabilises the bulk distribution
representation, improving robustness of final VaR/ES estimation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class PrototypeModel:
    """Fitted prototype model.

    Attributes
    ----------
    centers       : shape (M, latent_dim) — prototype locations in latent space
    weights       : shape (M,) — empirical mass per prototype (sum to 1)
    assignments   : shape (N_normal,) — prototype index per observation
    M             : number of prototypes
    inertia       : K-means inertia (reconstruction quality proxy)
    cluster_sizes : shape (M,) — count per prototype
    """
    centers: np.ndarray
    weights: np.ndarray
    assignments: np.ndarray
    M: int
    inertia: float
    cluster_sizes: np.ndarray

    def __post_init__(self) -> None:
        if abs(self.weights.sum() - 1.0) > 1e-5:
            raise ValueError(
                f"Prototype weights sum to {self.weights.sum():.6f}, expected 1.0"
            )
        assert len(self.weights) == self.M


def fit_prototypes(
    embeddings: np.ndarray,
    M: int,
    method: str = "kmeans",
    seed: int = 42,
) -> PrototypeModel:
    """Fit M prototypes to normal-state latent embeddings.

    Parameters
    ----------
    embeddings : shape (N, latent_dim)
    M          : number of prototypes
    method     : 'kmeans' (default) or 'random' (ablation)
    """
    N = len(embeddings)
    if M > N:
        logger.warning(f"M={M} > N={N}; capping M at N.")
        M = N

    if method == "kmeans":
        km = KMeans(n_clusters=M, random_state=seed, n_init="auto")
        km.fit(embeddings)
        assignments = km.labels_
        centers = km.cluster_centers_
        inertia = float(km.inertia_)
    elif method == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=M, replace=False)
        centers = embeddings[idx].copy()
        # Assign each obs to nearest center
        diffs = embeddings[:, None, :] - centers[None, :, :]  # (N, M, D)
        dists = np.sum(diffs ** 2, axis=2)
        assignments = np.argmin(dists, axis=1)
        inertia = float(np.sum(np.min(dists, axis=1)))
    else:
        raise ValueError(f"Unknown prototype method: '{method}'")

    cluster_sizes = np.bincount(assignments, minlength=M).astype(float)
    weights = cluster_sizes / cluster_sizes.sum()

    logger.info(
        f"Prototypes fitted: M={M}, method={method}, inertia={inertia:.4f}, "
        f"min_weight={weights.min():.4f}, max_weight={weights.max():.4f}"
    )
    return PrototypeModel(
        centers=centers,
        weights=weights,
        assignments=assignments,
        M=M,
        inertia=inertia,
        cluster_sizes=cluster_sizes,
    )


def assign_to_prototypes(
    embeddings: np.ndarray, model: PrototypeModel
) -> np.ndarray:
    """Assign new embeddings to the nearest prototype centre."""
    diffs = embeddings[:, None, :] - model.centers[None, :, :]
    dists = np.sum(diffs ** 2, axis=2)
    return np.argmin(dists, axis=1).astype(int)


def compute_prototype_reconstruction_quality(
    embeddings: np.ndarray,
    model: PrototypeModel,
) -> dict:
    """Reconstruction quality metrics for the prototype model."""
    assignments = assign_to_prototypes(embeddings, model)
    reconstructed = model.centers[assignments]
    errors = np.linalg.norm(embeddings - reconstructed, axis=1)

    total_var = np.var(embeddings, axis=0).sum()
    within_var = sum(
        np.var(embeddings[assignments == j], axis=0).sum() if (assignments == j).sum() > 1 else 0.0
        for j in range(model.M)
    )
    explained_var_ratio = 1.0 - within_var / total_var if total_var > 0 else 0.0

    return {
        "mean_reconstruction_error": float(errors.mean()),
        "max_reconstruction_error": float(errors.max()),
        "within_cluster_variance": float(within_var),
        "explained_variance_ratio": float(explained_var_ratio),
        "inertia": model.inertia,
    }


def select_M_by_reconstruction(
    embeddings: np.ndarray,
    M_values: list[int],
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate reconstruction quality for multiple M values.

    Returns DataFrame (M as index) with reconstruction quality metrics.
    Used for data-driven M selection, not visual guessing.
    """
    rows = []
    for M in M_values:
        model = fit_prototypes(embeddings, M=M, method="kmeans", seed=seed)
        quality = compute_prototype_reconstruction_quality(embeddings, model)
        quality["M"] = M
        rows.append(quality)
    df = pd.DataFrame(rows).set_index("M")
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_prototypes(model: PrototypeModel, path: Path | str) -> None:
    """Save prototype model to .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        centers=model.centers,
        weights=model.weights,
        assignments=model.assignments,
        cluster_sizes=model.cluster_sizes,
        M=np.array([model.M]),
        inertia=np.array([model.inertia]),
    )
    logger.info(f"Saved prototypes -> {path}")


def load_prototypes(path: Path | str) -> PrototypeModel:
    """Load prototype model from .npz file."""
    d = np.load(Path(path))
    return PrototypeModel(
        centers=d["centers"],
        weights=d["weights"],
        assignments=d["assignments"],
        cluster_sizes=d["cluster_sizes"],
        M=int(d["M"][0]),
        inertia=float(d["inertia"][0]),
    )
