"""Stability analysis for the prototype model.

Bootstrap and rolling-window robustness checks confirm that the
number of prototypes M does not introduce instability in VaR/ES
estimates.  These utilities are used in calibration and structural
evaluation.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from src.normal_model.prototypes import PrototypeModel, fit_prototypes

logger = logging.getLogger(__name__)


def bootstrap_var_es_stability(
    embeddings: np.ndarray,
    prototype_model: PrototypeModel,
    loss_fn: Callable[[np.ndarray], tuple[float, float]],
    n_bootstrap: int = 200,
    confidence_level: float = 0.99,
    seed: int = 42,
) -> dict:
    """Bootstrap stability of VaR/ES estimates under resampled prototype weights.

    Procedure
    ---------
    1. For each bootstrap draw: resample normal observations with replacement.
    2. Re-assign resampled observations to existing prototype centres.
    3. Recompute prototype weights from resampled assignments.
    4. Apply ``loss_fn(weights)`` → (VaR, ES).
    5. Return bootstrap distribution statistics.

    Parameters
    ----------
    loss_fn : callable(weights) → (var, es)
        Given prototype weights (np.ndarray summing to 1), returns (VaR, ES).
    """
    rng = np.random.default_rng(seed)
    N = len(embeddings)
    M = prototype_model.M

    var_samples, es_samples = [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        sampled = embeddings[idx]

        # Assign to nearest existing centre
        diffs = sampled[:, None, :] - prototype_model.centers[None, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        assignments = np.argmin(dists, axis=1)

        cluster_sizes = np.bincount(assignments, minlength=M).astype(float)
        weights = cluster_sizes / cluster_sizes.sum()

        var_val, es_val = loss_fn(weights)
        var_samples.append(var_val)
        es_samples.append(es_val)

    var_arr = np.array(var_samples)
    es_arr = np.array(es_samples)

    return {
        "var_mean": float(var_arr.mean()),
        "var_std": float(var_arr.std()),
        "var_cv": float(var_arr.std() / abs(var_arr.mean())) if var_arr.mean() != 0 else np.nan,
        "es_mean": float(es_arr.mean()),
        "es_std": float(es_arr.std()),
        "es_cv": float(es_arr.std() / abs(es_arr.mean())) if es_arr.mean() != 0 else np.nan,
        "var_bootstrap": var_arr,
        "es_bootstrap": es_arr,
    }


def rolling_retrain_stability(
    embeddings_sequence: list[np.ndarray],
    M: int,
    loss_fn: Callable[[np.ndarray], tuple[float, float]],
    confidence_level: float = 0.99,
    seed: int = 42,
) -> pd.DataFrame:
    """VaR/ES stability across rolling retrain windows.

    For each window in *embeddings_sequence*: fit prototypes, compute risk.
    Returns DataFrame with columns [var, es, n_prototypes_used].
    """
    rows = []
    for i, emb in enumerate(embeddings_sequence):
        m_actual = min(M, len(emb))
        proto = fit_prototypes(emb, M=m_actual, method="kmeans", seed=seed)
        var_val, es_val = loss_fn(proto.weights)
        rows.append({"window": i, "var": var_val, "es": es_val,
                     "n_prototypes_used": m_actual})

    return pd.DataFrame(rows).set_index("window")


def compare_M_values_stability(
    embeddings: np.ndarray,
    M_values: list[int],
    loss_fn: Callable[[np.ndarray], tuple[float, float]],
    n_bootstrap: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare prototype stability across M values.

    For each M: compute bootstrap CV of VaR/ES.
    Returns DataFrame to support data-driven M selection.
    """
    rows = []
    for M in M_values:
        proto = fit_prototypes(embeddings, M=M, seed=seed)
        stab = bootstrap_var_es_stability(
            embeddings, proto, loss_fn, n_bootstrap=n_bootstrap, seed=seed
        )
        rows.append({
            "M": M,
            "var_mean": stab["var_mean"],
            "var_cv": stab["var_cv"],
            "es_mean": stab["es_mean"],
            "es_cv": stab["es_cv"],
        })
    return pd.DataFrame(rows).set_index("M")
