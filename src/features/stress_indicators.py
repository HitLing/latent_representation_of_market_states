"""Partial stress indicators for the formal stress index.

Each indicator captures one dimension of market stress and is
rank-normalised to [0,1] using the empirical CDF.  Normalisation is
always fit on the training set and applied to out-of-sample data.

Indicators
----------
1. return_shock        — negative cross-sectional return shock
2. volatility_spike    — short-to-long volatility ratio
3. correlation_stress  — elevated average pairwise correlation
4. portfolio_loss_proxy — rolling portfolio drawdown
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Partial indicators (raw, un-normalised)
# ---------------------------------------------------------------------------

def compute_return_shock_indicator(
    returns: pd.DataFrame, cfg: dict
) -> pd.Series:
    """Standardised negative return shock (clipped at zero).

    Formula: max(0, -mean_return / rolling_std)
    Positive = stress (large negative cross-sectional return).
    """
    vcfg = cfg.get("volatility", {})
    short_w: int = vcfg.get("short_window", 5)

    mean_ret = returns.mean(axis=1)
    roll_std = mean_ret.rolling(short_w).std().replace(0, np.nan)
    shock = (-mean_ret / roll_std).clip(lower=0)
    shock.name = "return_shock_raw"
    return shock.fillna(0.0)


def compute_volatility_spike_indicator(
    returns: pd.DataFrame, cfg: dict
) -> pd.Series:
    """Short-to-long realised volatility ratio.

    ratio > 1 signals a volatility spike relative to the recent baseline.
    """
    vcfg = cfg.get("volatility", {})
    short_w: int = vcfg.get("short_window", 5)
    long_w: int = vcfg.get("long_window", 21)

    proxy = returns.mean(axis=1)
    vol_short = proxy.rolling(short_w).std()
    vol_long = proxy.rolling(long_w).std().replace(0, np.nan)
    ratio = (vol_short / vol_long).fillna(1.0)
    ratio.name = "vol_spike_raw"
    return ratio


def compute_correlation_stress_indicator(
    returns: pd.DataFrame, cfg: dict
) -> pd.Series:
    """Rolling mean absolute pairwise correlation.

    High values indicate correlation contagion — a hallmark of stress regimes.
    """
    ccfg = cfg.get("correlation", {})
    window: int = ccfg.get("rolling_window", 21)
    n_assets = returns.shape[1]

    if n_assets < 2:
        return pd.Series(0.0, index=returns.index, name="corr_stress_raw")

    arr = returns.to_numpy()
    pairs = [(i, j) for i in range(n_assets) for j in range(n_assets) if i < j]
    result = pd.Series(np.nan, index=returns.index, name="corr_stress_raw")

    for t in range(window - 1, len(returns)):
        chunk = arr[t - window + 1 : t + 1]
        vals = []
        for i, j in pairs:
            xi, xj = chunk[:, i], chunk[:, j]
            if xi.std() < 1e-10 or xj.std() < 1e-10:
                continue
            vals.append(abs(np.corrcoef(xi, xj)[0, 1]))
        result.iloc[t] = float(np.mean(vals)) if vals else 0.0

    return result.fillna(0.0)


def compute_portfolio_loss_proxy_indicator(
    returns: pd.DataFrame, weights: pd.Series, cfg: dict
) -> pd.Series:
    """Rolling portfolio drawdown — captures accumulated loss pressure."""
    vcfg = cfg.get("volatility", {})
    long_w: int = vcfg.get("long_window", 21)

    w = weights.reindex(returns.columns, fill_value=0.0)
    w = w / w.sum()
    port = returns @ w

    cum = (1 + port).cumprod()
    rolling_max = cum.rolling(long_w, min_periods=1).max()
    dd = ((rolling_max - cum) / rolling_max.replace(0, np.nan)).clip(lower=0)
    dd.name = "portfolio_loss_proxy_raw"
    return dd.fillna(0.0)


# ---------------------------------------------------------------------------
# Rank normalisation
# ---------------------------------------------------------------------------

def rank_normalize(
    series: pd.Series,
    fit_series: pd.Series | None = None,
) -> pd.Series:
    """Rank-normalise series to [0,1] using empirical CDF.

    If *fit_series* is provided, the CDF is estimated on *fit_series*
    (the training split) and applied to *series* via linear interpolation.
    This prevents look-ahead bias.
    """
    reference = fit_series if fit_series is not None else series
    ref_sorted = np.sort(reference.dropna().values)
    n = len(ref_sorted)
    if n == 0:
        return pd.Series(0.0, index=series.index, name=series.name)

    # CDF ranks: (0.5/n, 1.5/n, ..., (n-0.5)/n) — mid-point rule
    cdf_vals = (np.arange(n) + 0.5) / n

    def _map(val: float) -> float:
        if np.isnan(val):
            return np.nan
        return float(np.interp(val, ref_sorted, cdf_vals, left=0.0, right=1.0))

    mapped = series.map(_map)
    mapped.name = series.name
    return mapped


# ---------------------------------------------------------------------------
# Combined computation
# ---------------------------------------------------------------------------

def compute_all_indicators(
    returns: pd.DataFrame,
    weights: pd.Series,
    cfg: dict,
    fit_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute and rank-normalise all four stress indicators.

    Parameters
    ----------
    returns      : full (or out-of-sample) return DataFrame
    weights      : portfolio weights
    cfg          : feature config dict
    fit_returns  : if provided, CDF normalisation is fit on these returns (train set)

    Returns
    -------
    pd.DataFrame with columns [return_shock, volatility_spike,
                                correlation_stress, portfolio_loss_proxy],
    all values in [0, 1].
    """
    fit_r = fit_returns if fit_returns is not None else returns

    raw_indicators = {
        "return_shock": compute_return_shock_indicator(returns, cfg),
        "volatility_spike": compute_volatility_spike_indicator(returns, cfg),
        "correlation_stress": compute_correlation_stress_indicator(returns, cfg),
        "portfolio_loss_proxy": compute_portfolio_loss_proxy_indicator(returns, weights, cfg),
    }

    fit_indicators = {
        "return_shock": compute_return_shock_indicator(fit_r, cfg),
        "volatility_spike": compute_volatility_spike_indicator(fit_r, cfg),
        "correlation_stress": compute_correlation_stress_indicator(fit_r, cfg),
        "portfolio_loss_proxy": compute_portfolio_loss_proxy_indicator(fit_r, weights, cfg),
    }

    normalised: dict[str, pd.Series] = {}
    for name, series in raw_indicators.items():
        normalised[name] = rank_normalize(series, fit_series=fit_indicators[name])

    out = pd.DataFrame(normalised, index=returns.index)
    logger.debug(
        f"Computed indicators: means = "
        + ", ".join(f"{k}={v.mean():.3f}" for k, v in normalised.items())
    )
    return out
