"""Market state feature computation.

All features use a rolling basis and return a DataFrame with the same
DatetimeIndex as the input (NaN for the initial warm-up period).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_all_features(
    returns: pd.DataFrame,
    weights: pd.Series,
    cfg: dict,
) -> pd.DataFrame:
    """Compute all configured market features.

    Feature groups are activated via ``cfg['feature_groups']`` boolean flags.
    Returns a single DataFrame with all feature columns.
    """
    groups = cfg.get("feature_groups", {})
    frames: list[pd.DataFrame] = []

    if groups.get("returns", True):
        frames.append(compute_return_features(returns, cfg))

    if groups.get("volatility", True):
        frames.append(compute_volatility_features(returns, cfg))

    if groups.get("correlation", True):
        frames.append(compute_correlation_features(returns, cfg))

    if groups.get("dispersion", True):
        frames.append(compute_dispersion_features(returns, cfg))

    if groups.get("portfolio_proxy", True):
        frames.append(compute_portfolio_proxy_features(returns, weights, cfg))

    if not frames:
        raise ValueError("No feature groups enabled in config.")

    features = pd.concat(frames, axis=1)
    features = features.sort_index()

    n_nan = features.isnull().sum().sum()
    nan_pct = 100 * n_nan / features.size
    logger.info(
        f"Computed {features.shape[1]} features over {len(features)} dates "
        f"({nan_pct:.1f}% NaN, expected during warm-up)"
    )
    return features


def compute_return_features(returns: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return-based cross-sectional summary features."""
    vcfg = cfg.get("volatility", {})
    short_w = vcfg.get("short_window", 5)

    out = pd.DataFrame(index=returns.index)
    out["cs_mean_return"] = returns.mean(axis=1)
    out["cs_min_return"] = returns.min(axis=1)
    out["cs_max_return"] = returns.max(axis=1)
    out[f"rolling_mean_{short_w}d"] = out["cs_mean_return"].rolling(short_w).mean()
    return out


def compute_volatility_features(returns: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Rolling volatility features."""
    vcfg = cfg.get("volatility", {})
    short_w: int = vcfg.get("short_window", 5)
    long_w: int = vcfg.get("long_window", 21)

    # Portfolio-level proxy (simple mean across assets)
    proxy = returns.mean(axis=1)

    out = pd.DataFrame(index=returns.index)
    out[f"realized_vol_{short_w}d"] = proxy.rolling(short_w).std()
    out[f"realized_vol_{long_w}d"] = proxy.rolling(long_w).std()

    # Vol ratio — key stress signal
    short_vol = out[f"realized_vol_{short_w}d"]
    long_vol = out[f"realized_vol_{long_w}d"]
    out["vol_ratio"] = short_vol / long_vol.replace(0, np.nan)

    # Cross-sectional mean vol (per-asset rolling std averaged)
    cs_vol = returns.rolling(short_w).std().mean(axis=1)
    out["cs_mean_vol"] = cs_vol

    return out


def compute_correlation_features(returns: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Rolling pairwise correlation features."""
    ccfg = cfg.get("correlation", {})
    window: int = ccfg.get("rolling_window", 21)

    n_assets = returns.shape[1]
    if n_assets < 2:
        return pd.DataFrame(index=returns.index)

    dates = returns.index
    mean_abs_corr = pd.Series(np.nan, index=dates, name="mean_abs_corr")
    max_corr = pd.Series(np.nan, index=dates, name="max_pairwise_corr")
    min_corr = pd.Series(np.nan, index=dates, name="min_pairwise_corr")

    arr = returns.to_numpy()
    cols = list(range(n_assets))
    pairs = [(i, j) for i in cols for j in cols if i < j]

    for t in range(window - 1, len(dates)):
        window_data = arr[t - window + 1 : t + 1]
        # Compute pairwise correlations for this window
        corr_vals = []
        for i, j in pairs:
            x, y = window_data[:, i], window_data[:, j]
            sx, sy = x.std(), y.std()
            if sx < 1e-10 or sy < 1e-10:
                continue
            c = np.corrcoef(x, y)[0, 1]
            corr_vals.append(c)

        if corr_vals:
            cv = np.array(corr_vals)
            mean_abs_corr.iloc[t] = np.mean(np.abs(cv))
            max_corr.iloc[t] = np.max(cv)
            min_corr.iloc[t] = np.min(cv)

    out = pd.concat([mean_abs_corr, max_corr, min_corr], axis=1)
    return out


def compute_dispersion_features(returns: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Cross-sectional dispersion features."""
    out = pd.DataFrame(index=returns.index)
    out["cs_std"] = returns.std(axis=1)
    out["cs_range"] = returns.max(axis=1) - returns.min(axis=1)
    return out


def compute_portfolio_proxy_features(
    returns: pd.DataFrame, weights: pd.Series, cfg: dict
) -> pd.DataFrame:
    """Portfolio-level proxy features."""
    vcfg = cfg.get("volatility", {})
    long_w: int = vcfg.get("long_window", 21)

    # Align weights
    w = weights.reindex(returns.columns, fill_value=0.0)
    w = w / w.sum()

    port_ret = returns @ w

    out = pd.DataFrame(index=returns.index)
    out["portfolio_return"] = port_ret
    out["portfolio_rolling_vol"] = port_ret.rolling(long_w).std()

    # Rolling drawdown proxy: max loss over window
    cum = (1 + port_ret).cumprod()
    rolling_max = cum.rolling(long_w, min_periods=1).max()
    drawdown = (rolling_max - cum) / rolling_max.replace(0, np.nan)
    out["portfolio_rolling_dd"] = drawdown.clip(lower=0)

    return out
