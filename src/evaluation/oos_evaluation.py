"""Final OOS risk evaluation.

This module is ONLY applied to the held-out test set.
It is never used to tune parameters or make modelling decisions.

Evaluation metrics
------------------
- VaR exceedance rate
- Kupiec unconditional coverage test
- Christoffersen conditional coverage test
- Simple ES backtest
- Quantile / tail loss metrics
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.risk.backtesting import run_full_backtest, compute_var_exceedances
from src.risk.weighted_risk_metrics import (
    compute_quantile_loss_series,
    compute_risk_metrics,
)

logger = logging.getLogger(__name__)


def run_oos_evaluation(
    realized_losses: pd.Series,
    var_estimates: dict[str, pd.Series],
    es_estimates: dict[str, pd.Series],
    confidence_levels: list[float] | None = None,
    significance_level: float = 0.05,
) -> dict:
    """Run all OOS backtests for each confidence level.

    Parameters
    ----------
    realized_losses : pd.Series, DatetimeIndex, positive = loss
    var_estimates   : {method_name: pd.Series of rolling VaR estimates}
    es_estimates    : {method_name: pd.Series of rolling ES estimates}

    Returns nested dict: {method_name: {confidence_level: backtest_results}}
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    results: dict = {}
    for method in var_estimates:
        results[method] = {}
        for cl in confidence_levels:
            key = f"CL_{cl}"
            var_key = f"VaR_{cl}"
            es_key = f"ES_{cl}"

            # Get the series for this CL
            var_ser = var_estimates[method].get(var_key) if isinstance(
                var_estimates[method], dict
            ) else var_estimates[method]

            es_ser = es_estimates[method].get(es_key) if isinstance(
                es_estimates[method], dict
            ) else es_estimates[method]

            if var_ser is None or es_ser is None:
                logger.warning(f"Missing VaR/ES series for {method} at CL={cl}")
                continue

            bt = run_full_backtest(
                realized_losses, var_ser, es_ser, cl, significance_level
            )

            # Quantile loss
            ql = compute_quantile_loss_series(var_ser, realized_losses, cl)
            bt["mean_quantile_loss"] = float(ql.mean())

            results[method][key] = bt

    return results


def format_oos_report(results: dict) -> str:
    """Format OOS evaluation results as a readable string."""
    lines = ["=== OOS Risk Evaluation Report ==="]
    for method, cl_results in results.items():
        lines.append(f"\n  Method: {method}")
        for cl_key, bt in cl_results.items():
            kupiec = bt.get("kupiec", {})
            cc = bt.get("christoffersen", {})
            lines.append(
                f"    [{cl_key}] "
                f"exc_rate={kupiec.get('empirical_rate', float('nan')):.3%} "
                f"(exp={kupiec.get('expected_rate', float('nan')):.3%}), "
                f"Kupiec p={kupiec.get('p_value', float('nan')):.3f} "
                f"[{'FAIL' if kupiec.get('reject_H0') else 'pass'}], "
                f"CC p={cc.get('p_value_cc', float('nan')):.3f} "
                f"[{'FAIL' if cc.get('reject_H0_joint') else 'pass'}], "
                f"mean_QL={bt.get('mean_quantile_loss', float('nan')):.5f}"
            )
    return "\n".join(lines)


def compute_rolling_var_estimates(
    assembly_fn: callable,
    returns: pd.DataFrame,
    weights: pd.Series,
    rolling_splits: list,
    confidence_levels: list[float] = None,
) -> pd.DataFrame:
    """Compute rolling VaR/ES estimates for backtesting.

    assembly_fn : callable(train_returns) → ScenarioAssembly
    rolling_splits : list of TemporalSplit

    Returns DataFrame with date index and VaR/ES columns.
    """
    from src.risk.portfolio_loss import compute_scenario_losses
    from src.risk.weighted_risk_metrics import compute_risk_metrics

    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    rows = []
    for sp in rolling_splits:
        train_ret = returns.iloc[sp.train_idx]
        test_ret = returns.iloc[sp.test_idx]

        try:
            assembly = assembly_fn(train_ret)
            losses, w = compute_scenario_losses(assembly, weights)
            metrics = compute_risk_metrics(losses, w, confidence_levels)
        except Exception as e:
            logger.warning(f"Rolling eval failed: {e}")
            metrics = {f"VaR_{cl}": float("nan") for cl in confidence_levels}
            metrics.update({f"ES_{cl}": float("nan") for cl in confidence_levels})

        for test_date in test_ret.index:
            row = {"date": test_date, **metrics}
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("date")
    return df
