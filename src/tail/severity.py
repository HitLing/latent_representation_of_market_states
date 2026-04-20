"""Severity bucket assignment for stress observations.

Severity buckets partition the stress regime by intensity:
  bucket 0 : mild stress
  bucket 1 : moderate stress
  bucket 2 : severe stress   (default: 3 buckets)

These are used as conditioning variables in the CVAE.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def assign_severity_buckets(
    stress_score: pd.Series,
    stress_dates: pd.DatetimeIndex,
    n_buckets: int = 3,
    method: str = "quantile",
) -> pd.Series:
    """Assign severity buckets to stress observations.

    Parameters
    ----------
    stress_score : full stress score series (DatetimeIndex)
    stress_dates : dates belonging to the stress regime
    n_buckets    : number of severity levels
    method       : 'quantile' (equal-count) | 'fixed' (equal-range)
    """
    score_at_stress = stress_score.loc[stress_score.index.intersection(stress_dates)]
    vals = score_at_stress.values
    n = len(vals)

    if n == 0:
        return pd.Series(dtype=int, name="severity_bucket")

    if method == "quantile":
        quantiles = np.linspace(0, 100, n_buckets + 1)
        thresholds = np.percentile(vals, quantiles)
    elif method == "fixed":
        lo, hi = vals.min(), vals.max()
        thresholds = np.linspace(lo, hi, n_buckets + 1)
    else:
        raise ValueError(f"Unknown severity method: '{method}'")

    buckets = np.digitize(vals, thresholds[1:-1]).astype(int)
    # Ensure range [0, n_buckets-1]
    buckets = np.clip(buckets, 0, n_buckets - 1)

    result = pd.Series(buckets, index=score_at_stress.index, name="severity_bucket")
    counts = {b: int((buckets == b).sum()) for b in range(n_buckets)}
    logger.info(f"Severity buckets (method={method}): {counts}")
    return result


def compute_severity_stats(
    returns: pd.DataFrame,
    weights: pd.Series,
    severity_labels: pd.Series,
) -> pd.DataFrame:
    """Statistics per severity bucket."""
    port = returns @ weights
    rows = []
    for bucket in sorted(severity_labels.unique()):
        dates = severity_labels[severity_labels == bucket].index
        r_sub = returns.loc[returns.index.intersection(dates)]
        p_sub = port.loc[port.index.intersection(dates)]
        rows.append({
            "bucket": bucket,
            "n_observations": len(r_sub),
            "mean_portfolio_loss": float(-p_sub.mean()) if len(p_sub) > 0 else np.nan,
            "std_portfolio_loss": float(p_sub.std()) if len(p_sub) > 1 else np.nan,
            "max_portfolio_loss": float(-p_sub.min()) if len(p_sub) > 0 else np.nan,
            "mean_vol": float(r_sub.std().mean()) if len(r_sub) > 1 else np.nan,
        })
    df = pd.DataFrame(rows).set_index("bucket")
    return df


def get_severity_conditioning_vector(
    severity_label: int, n_buckets: int
) -> np.ndarray:
    """One-hot encode a single severity bucket label."""
    v = np.zeros(n_buckets, dtype=np.float32)
    v[severity_label] = 1.0
    return v


def encode_severity_batch(
    severity_labels: np.ndarray, n_buckets: int
) -> np.ndarray:
    """Encode array of severity labels to one-hot matrix, shape (N, n_buckets)."""
    N = len(severity_labels)
    out = np.zeros((N, n_buckets), dtype=np.float32)
    for i, lbl in enumerate(severity_labels):
        out[i, int(lbl)] = 1.0
    return out
