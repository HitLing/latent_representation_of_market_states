"""VaR and ES backtesting.

Implements formal backtesting procedures:
  - Kupiec (1995)      : unconditional coverage test
  - Christoffersen (1998) : conditional coverage test (coverage + independence)
  - Simple ES backtest : McNeil–Frey mean-exceedance residual approach

All backtests are run ONLY on the final OOS test set.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_var_exceedances(
    realized_losses: pd.Series,
    var_estimates: pd.Series,
) -> pd.Series:
    """Exceedance indicator: 1 if realised_loss > VaR, else 0."""
    common = realized_losses.index.intersection(var_estimates.index)
    exc = (realized_losses.loc[common] > var_estimates.loc[common]).astype(int)
    exc.name = "exceedance"
    return exc


def kupiec_test(
    exceedances: pd.Series,
    confidence_level: float,
    significance_level: float = 0.05,
) -> dict:
    """Kupiec (1995) unconditional coverage test.

    H₀: P(exceedance) = 1 − confidence_level
    LR_uc = −2 · [log L(p₀) − log L(p̂)] ~ χ²(1) under H₀

    where p₀ = 1 − confidence_level  (expected rate)
          p̂  = n₁ / T              (empirical rate)
    """
    T = len(exceedances)
    n1 = int(exceedances.sum())  # number of exceedances
    n0 = T - n1

    p0 = 1.0 - confidence_level  # expected exceedance rate
    p_hat = n1 / T if T > 0 else 0.0

    # Log-likelihood under H₀
    def _safe_log(x: float) -> float:
        return math.log(x) if x > 1e-12 else -1e9

    ll_null = n1 * _safe_log(p0) + n0 * _safe_log(1.0 - p0)
    ll_alt = (
        n1 * _safe_log(p_hat) + n0 * _safe_log(1.0 - p_hat)
        if 0 < p_hat < 1
        else -1e9
    )

    lr_uc = -2.0 * (ll_null - ll_alt)
    lr_uc = max(0.0, lr_uc)
    p_value = float(stats.chi2.sf(lr_uc, df=1))
    reject = p_value < significance_level

    return {
        "test_name": "Kupiec_UC",
        "n_obs": T,
        "n_exceedances": n1,
        "empirical_rate": p_hat,
        "expected_rate": p0,
        "LR_stat": lr_uc,
        "p_value": p_value,
        "reject_H0": reject,
        "significance_level": significance_level,
    }


def christoffersen_test(
    exceedances: pd.Series,
    confidence_level: float,
    significance_level: float = 0.05,
) -> dict:
    """Christoffersen (1998) conditional coverage test.

    Tests both unconditional coverage (LR_uc) and independence (LR_ind).
    LR_cc = LR_uc + LR_ind ~ χ²(2) under H₀

    Independence test uses the 2×2 transition matrix of the hit sequence.
    """
    hits = exceedances.values.astype(int)
    T = len(hits)

    # Transition counts
    n00 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 0)))
    n01 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 1)))
    n10 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 0)))
    n11 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 1)))

    n1 = int(hits.sum())
    n0 = T - n1
    p0 = 1.0 - confidence_level
    p_hat = n1 / T if T > 0 else 0.0

    def _safe_log(x: float) -> float:
        return math.log(x) if x > 1e-12 else -1e9

    # --- Unconditional LR ---
    ll_null = n1 * _safe_log(p0) + n0 * _safe_log(1.0 - p0)
    ll_uc = (
        n1 * _safe_log(p_hat) + n0 * _safe_log(1.0 - p_hat)
        if 0 < p_hat < 1 else -1e9
    )
    lr_uc = max(0.0, -2.0 * (ll_null - ll_uc))

    # --- Independence LR ---
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi_hat = (n01 + n11) / (n00 + n01 + n10 + n11) if T > 1 else 0.0

    ll_ind_alt = (
        (n00 * _safe_log(1.0 - pi01) + n01 * _safe_log(pi01 + 1e-12))
        + (n10 * _safe_log(1.0 - pi11 + 1e-12) + n11 * _safe_log(pi11 + 1e-12))
    )
    ll_ind_null = (
        (n00 + n10) * _safe_log(1.0 - pi_hat)
        + (n01 + n11) * _safe_log(pi_hat + 1e-12)
    )
    lr_ind = max(0.0, -2.0 * (ll_ind_null - ll_ind_alt))

    lr_cc = lr_uc + lr_ind

    p_val_uc = float(stats.chi2.sf(lr_uc, df=1))
    p_val_ind = float(stats.chi2.sf(lr_ind, df=1))
    p_val_cc = float(stats.chi2.sf(lr_cc, df=2))

    return {
        "test_name": "Christoffersen_CC",
        "LR_uc": lr_uc,
        "LR_ind": lr_ind,
        "LR_cc": lr_cc,
        "p_value_uc": p_val_uc,
        "p_value_ind": p_val_ind,
        "p_value_cc": p_val_cc,
        "reject_H0_coverage": p_val_uc < significance_level,
        "reject_H0_independence": p_val_ind < significance_level,
        "reject_H0_joint": p_val_cc < significance_level,
    }


def es_backtest_simple(
    realized_losses: pd.Series,
    es_estimates: pd.Series,
    var_estimates: pd.Series,
    confidence_level: float,
) -> dict:
    """Simple ES backtest (McNeil–Frey 2000 approach).

    On exceedance days: compute (realised_loss − ES_estimate).
    If ES is accurately estimated, mean residual ≈ 0.
    """
    exceedances = compute_var_exceedances(realized_losses, var_estimates)
    exc_dates = exceedances[exceedances == 1].index

    if len(exc_dates) == 0:
        return {
            "test_name": "ES_simple",
            "n_exceedances": 0,
            "mean_residual": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
        }

    rl = realized_losses.loc[exc_dates]
    es = es_estimates.reindex(exc_dates)
    residuals = rl.values - es.values

    mean_res = float(residuals.mean())
    std_res = float(residuals.std())
    n = len(residuals)
    t_stat = mean_res / (std_res / math.sqrt(n)) if std_res > 0 else 0.0
    p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))) if n > 1 else 1.0

    return {
        "test_name": "ES_simple",
        "n_exceedances": n,
        "mean_residual": mean_res,
        "t_stat": t_stat,
        "p_value": p_val,
        "reject_zero_residual": p_val < 0.05,
    }


def run_full_backtest(
    realized_losses: pd.Series,
    var_estimates: pd.Series,
    es_estimates: pd.Series,
    confidence_level: float,
    significance_level: float = 0.05,
) -> dict:
    """Run all backtests and return combined results."""
    exceedances = compute_var_exceedances(realized_losses, var_estimates)
    kupiec = kupiec_test(exceedances, confidence_level, significance_level)
    christoffersen = christoffersen_test(exceedances, confidence_level, significance_level)
    es_bt = es_backtest_simple(realized_losses, es_estimates, var_estimates, confidence_level)

    logger.info(
        f"Backtest (CL={confidence_level:.0%}): "
        f"exc_rate={kupiec['empirical_rate']:.3%} "
        f"(expected {kupiec['expected_rate']:.3%}), "
        f"Kupiec p={kupiec['p_value']:.3f}, "
        f"CC p={christoffersen['p_value_cc']:.3f}"
    )

    return {
        "confidence_level": confidence_level,
        "exceedances": exceedances,
        "kupiec": kupiec,
        "christoffersen": christoffersen,
        "es_backtest": es_bt,
    }
