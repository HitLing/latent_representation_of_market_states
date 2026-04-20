"""Weighted VaR and ES estimation.

Transparent, explicit formulas applied to the weighted empirical loss
distribution.  No neural quantile predictor.  No supervised VaR model.

VaR_α : smallest loss l such that cumulative weight ≥ α
ES_α  : weighted average of losses exceeding VaR_α
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def weighted_var(
    losses: np.ndarray,
    weights: np.ndarray,
    confidence_level: float = 0.99,
) -> float:
    """Weighted Value-at-Risk.

    Algorithm
    ---------
    Sort (loss, weight) pairs ascending.
    VaR = l_k where k = argmin{j : Σ_{i≤j} w_i ≥ confidence_level}.
    """
    if len(losses) == 0:
        return float("nan")
    order = np.argsort(losses)
    sl = losses[order]
    sw = weights[order]
    cum_w = np.cumsum(sw)
    idx = np.searchsorted(cum_w, confidence_level, side="left")
    idx = min(idx, len(sl) - 1)
    return float(sl[idx])


def weighted_es(
    losses: np.ndarray,
    weights: np.ndarray,
    confidence_level: float = 0.99,
) -> float:
    """Weighted Expected Shortfall (Conditional VaR).

    ES = weighted average of losses strictly exceeding VaR.
    If no losses exceed VaR (can happen for coarse distributions),
    returns VaR itself.
    """
    if len(losses) == 0:
        return float("nan")
    var = weighted_var(losses, weights, confidence_level)
    tail_mask = losses > var
    if tail_mask.sum() == 0:
        return var
    tail_losses = losses[tail_mask]
    tail_weights = weights[tail_mask]
    total_tail_weight = tail_weights.sum()
    if total_tail_weight < 1e-12:
        return var
    return float(np.sum(tail_losses * tail_weights) / total_tail_weight)


def compute_risk_metrics(
    losses: np.ndarray,
    weights: np.ndarray,
    confidence_levels: list[float] = None,
) -> dict:
    """Compute VaR and ES for all specified confidence levels.

    Returns dict: {'VaR_0.95': ..., 'ES_0.95': ..., 'VaR_0.99': ..., ...}
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    result: dict = {}
    for level in confidence_levels:
        result[f"VaR_{level}"] = weighted_var(losses, weights, level)
        result[f"ES_{level}"] = weighted_es(losses, weights, level)

    result["mean_loss"] = float(np.average(losses, weights=weights))
    result["std_loss"] = float(
        np.sqrt(np.average((losses - result["mean_loss"]) ** 2, weights=weights))
    )
    result["max_loss"] = float(losses.max())
    return result


def quantile_loss(
    predicted_quantile: float,
    realized_loss: float,
    alpha: float,
) -> float:
    """Pinball (quantile) loss for VaR evaluation.

    QL(q, y, α) = (y − q) · α      if y > q
                  (q − y) · (1−α)  otherwise
    """
    if realized_loss > predicted_quantile:
        return (realized_loss - predicted_quantile) * alpha
    return (predicted_quantile - realized_loss) * (1.0 - alpha)


def compute_quantile_loss_series(
    var_series: pd.Series,
    realized_losses: pd.Series,
    alpha: float,
) -> pd.Series:
    """Compute daily quantile loss series (for model comparison)."""
    common = var_series.index.intersection(realized_losses.index)
    ql = pd.Series(
        [
            quantile_loss(var_series.loc[d], realized_losses.loc[d], alpha)
            for d in common
        ],
        index=common,
        name=f"quantile_loss_{alpha}",
    )
    return ql
