"""Regime diagnostics and imbalance analysis."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.regimes.partition import RegimePartition

logger = logging.getLogger(__name__)


def compute_regime_statistics(
    partition: RegimePartition,
    returns: pd.DataFrame,
    portfolio_returns: pd.Series,
) -> dict:
    """Comprehensive regime statistics."""
    normal_ret = returns.loc[returns.index.intersection(partition.normal_dates)]
    stress_ret = returns.loc[returns.index.intersection(partition.stress_dates)]
    normal_port = portfolio_returns.loc[portfolio_returns.index.intersection(partition.normal_dates)]
    stress_port = portfolio_returns.loc[portfolio_returns.index.intersection(partition.stress_dates)]

    def _safe_mean(s: pd.Series) -> float:
        return float(s.mean()) if len(s) > 0 else float("nan")

    def _safe_std(s: pd.Series) -> float:
        return float(s.std()) if len(s) > 1 else float("nan")

    # Episode detection
    eps = compute_stress_episode_stats(partition.regime_labels)
    n_episodes = len(eps)
    avg_len = float(eps["length"].mean()) if len(eps) > 0 else 0.0

    normal_vol = _safe_std(normal_port)
    stress_vol = _safe_std(stress_port)
    vol_ratio = (stress_vol / normal_vol) if (normal_vol and normal_vol > 0) else float("nan")

    stats = {
        "n_total": partition.n_total,
        "n_normal": partition.n_normal,
        "n_stress": partition.n_stress,
        "stress_fraction": partition.stress_fraction,
        "normal_mean_return": _safe_mean(normal_port),
        "stress_mean_return": _safe_mean(stress_port),
        "normal_volatility": normal_vol,
        "stress_volatility": stress_vol,
        "vol_ratio_stress_over_normal": vol_ratio,
        "n_stress_episodes": n_episodes,
        "avg_stress_episode_length": avg_len,
    }

    logger.info(
        f"Regime stats: stress_vol/normal_vol = {vol_ratio:.2f}, "
        f"n_episodes = {n_episodes}, avg_len = {avg_len:.1f} days"
    )
    return stats


def compute_rolling_regime_share(
    regime_labels: pd.Series, window: int = 63
) -> pd.Series:
    """Rolling fraction of stress observations over *window* trading days."""
    share = regime_labels.astype(int).rolling(window, min_periods=1).mean()
    share.name = f"rolling_stress_share_{window}d"
    return share


def compute_stress_episode_stats(regime_labels: pd.Series) -> pd.DataFrame:
    """Identify contiguous stress episodes and compute per-episode statistics."""
    is_stress = regime_labels.astype(int)
    # Detect transitions
    transitions = is_stress.diff().fillna(is_stress)

    episodes = []
    in_stress = False
    start = None

    for date, val in is_stress.items():
        if val == 1 and not in_stress:
            start = date
            in_stress = True
        elif val == 0 and in_stress:
            episodes.append({"start": start, "end": date, "length": 0})
            in_stress = False
    if in_stress:
        episodes.append({"start": start, "end": regime_labels.index[-1], "length": 0})

    if not episodes:
        return pd.DataFrame(columns=["start", "end", "length"])

    df = pd.DataFrame(episodes)
    # Compute length in trading days
    for i, row in df.iterrows():
        mask = (regime_labels.index >= row["start"]) & (regime_labels.index <= row["end"])
        df.at[i, "length"] = int(regime_labels.loc[mask].sum())
    return df


def within_regime_variance_summary(
    returns: pd.DataFrame,
    partition: RegimePartition,
) -> pd.DataFrame:
    """Per-asset variance within each regime."""
    normal_r = returns.loc[returns.index.intersection(partition.normal_dates)]
    stress_r = returns.loc[returns.index.intersection(partition.stress_dates)]

    summary = pd.DataFrame({
        "normal_var": normal_r.var(),
        "stress_var": stress_r.var(),
    })
    summary["var_ratio"] = summary["stress_var"] / summary["normal_var"].replace(0, np.nan)
    return summary


def imbalance_report(
    partition: RegimePartition,
    returns: pd.DataFrame,
) -> str:
    """Human-readable imbalance report for logging."""
    var_summary = within_regime_variance_summary(returns, partition)
    avg_ratio = float(var_summary["var_ratio"].mean())

    lines = [
        "=== Regime Imbalance Report ===",
        f"  Total obs  : {partition.n_total}",
        f"  Normal obs : {partition.n_normal} ({partition.n_normal/partition.n_total:.1%})",
        f"  Stress obs : {partition.n_stress} ({partition.stress_fraction:.1%})",
        f"  Imbalance ratio (normal:stress) : {partition.n_normal/max(partition.n_stress,1):.1f}:1",
        f"  Mean stress/normal variance ratio : {avg_ratio:.2f}x",
        f"  Alpha (threshold quantile) : {partition.alpha:.3f}",
        f"  Score threshold            : {partition.threshold:.4f}",
    ]
    return "\n".join(lines)
