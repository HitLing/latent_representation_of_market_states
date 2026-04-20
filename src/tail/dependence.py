"""Dependence structure analysis for stress scenarios.

Used in structural evaluation (plausibility checks) to verify that
synthetic stress scenarios preserve the correlation structure of
historical stress observations.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix."""
    return returns.corr()


def compute_tail_correlation(
    returns: pd.DataFrame, quantile: float = 0.10
) -> pd.DataFrame:
    """Tail correlation: probability that both assets simultaneously fall
    below their individual q-quantile.

    Returns a matrix in [0, 1].
    """
    n = len(returns)
    assets = returns.columns.tolist()
    k = len(assets)
    mat = np.zeros((k, k))

    thresholds = {col: returns[col].quantile(quantile) for col in assets}

    for i, a in enumerate(assets):
        for j, b in enumerate(assets):
            below_i = returns[a] <= thresholds[a]
            below_j = returns[b] <= thresholds[b]
            mat[i, j] = float((below_i & below_j).mean())

    return pd.DataFrame(mat, index=assets, columns=assets)


def compare_dependence_structures(
    normal_returns: pd.DataFrame,
    stress_returns: pd.DataFrame,
    synthetic_returns: pd.DataFrame | None = None,
) -> dict:
    """Compare dependence structures across regimes."""
    normal_corr = compute_correlation_matrix(normal_returns)
    stress_corr = compute_correlation_matrix(stress_returns)

    def _frobenius(A: pd.DataFrame, B: pd.DataFrame) -> float:
        diff = A.values - B.reindex(index=A.index, columns=A.columns).values
        return float(np.linalg.norm(diff, "fro"))

    result = {
        "normal_correlation": normal_corr,
        "stress_correlation": stress_corr,
        "normal_tail_corr": compute_tail_correlation(normal_returns),
        "stress_tail_corr": compute_tail_correlation(stress_returns),
        "frobenius_distance_stress_vs_normal": _frobenius(stress_corr, normal_corr),
    }

    if synthetic_returns is not None:
        syn_corr = compute_correlation_matrix(synthetic_returns)
        result["synthetic_correlation"] = syn_corr
        result["frobenius_distance_synthetic_vs_stress"] = _frobenius(syn_corr, stress_corr)

    return result


def test_correlation_change(
    normal_returns: pd.DataFrame, stress_returns: pd.DataFrame
) -> dict:
    """Statistical summary of correlation change between regimes.

    Uses Fisher z-transformation to compare pairwise correlations.
    Returns per-pair statistics.
    """
    assets = normal_returns.columns.tolist()
    pairs = [(a, b) for i, a in enumerate(assets) for b in assets[i + 1:]]

    rows = []
    for a, b in pairs:
        r_n = float(normal_returns[[a, b]].corr().iloc[0, 1])
        r_s = float(stress_returns[[a, b]].corr().iloc[0, 1])
        z_n = np.arctanh(np.clip(r_n, -0.9999, 0.9999))
        z_s = np.arctanh(np.clip(r_s, -0.9999, 0.9999))
        n_n = len(normal_returns)
        n_s = len(stress_returns)
        se = np.sqrt(1.0 / (n_n - 3) + 1.0 / (n_s - 3))
        z_stat = (z_s - z_n) / se if se > 0 else 0.0
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        rows.append({
            "asset_a": a,
            "asset_b": b,
            "corr_normal": r_n,
            "corr_stress": r_s,
            "corr_change": r_s - r_n,
            "z_stat": z_stat,
            "p_value": p_val,
        })

    return {"pair_stats": pd.DataFrame(rows)}
