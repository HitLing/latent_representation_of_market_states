"""Portfolio-level feature computation."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_portfolio_returns(
    returns: pd.DataFrame, weights: pd.Series
) -> pd.Series:
    """Compute portfolio return series: Σ_i w_i · r_{i,t}."""
    w = weights.reindex(returns.columns, fill_value=0.0)
    if w.sum() == 0:
        raise ValueError("Portfolio weights sum to zero after reindex.")
    w = w / w.sum()
    port = returns @ w
    port.name = "portfolio_return"
    return port


def compute_portfolio_cumulative(portfolio_returns: pd.Series) -> pd.Series:
    """Cumulative return: ∏(1 + r_t) - 1."""
    cum = (1 + portfolio_returns).cumprod() - 1
    cum.name = "portfolio_cumulative"
    return cum


def compute_rolling_drawdown(
    portfolio_returns: pd.Series, window: int
) -> pd.Series:
    """Rolling maximum drawdown over *window* bars.

    DD_t = (max_{s in [t-w, t]} C_s - C_t) / (1 + max_{s in [t-w, t]} C_s)

    where C_t is the cumulative return at time t.
    """
    cum = (1 + portfolio_returns).cumprod()
    rolling_max = cum.rolling(window, min_periods=1).max()
    dd = (rolling_max - cum) / (1 + rolling_max)
    dd = dd.clip(lower=0)
    dd.name = "rolling_drawdown"
    return dd


def compute_portfolio_exposure_features(
    returns: pd.DataFrame, weights: pd.Series
) -> pd.DataFrame:
    """Static exposure features derived from portfolio weights.

    These are time-invariant for static weights but included for
    completeness and to support dynamic-weight extensions.
    """
    w = weights.reindex(returns.columns, fill_value=0.0)
    w = w / w.sum()
    w_vals = w.values

    herfindahl = float(np.sum(w_vals ** 2))
    effective_n = 1.0 / herfindahl if herfindahl > 0 else float(len(w_vals))
    top1 = float(w_vals.max())
    top3 = float(np.sort(w_vals)[::-1][:3].sum())

    n = len(returns)
    out = pd.DataFrame(
        {
            "effective_n": np.full(n, effective_n),
            "top1_weight": np.full(n, top1),
            "top3_weight_share": np.full(n, top3),
            "herfindahl": np.full(n, herfindahl),
        },
        index=returns.index,
    )
    return out
