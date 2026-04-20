"""Restricted calibration protocol.

Calibrated parameters and their hierarchy
------------------------------------------
alpha : main structural parameter — stress threshold
m     : tail enrichment parameter — synthetic mass ratio
M     : stability/compression parameter — prototype count
b     : secondary technical parameter — severity buckets

Rules
-----
- Parameters are searched over a grid on the INNER VALIDATION split only.
- The final OOS test set is NEVER touched during calibration.
- The objective metric is validated quantile loss at a chosen confidence level.
"""
from __future__ import annotations

import logging
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calibrate_parameters(
    train_fn: Callable[[dict], float],
    param_grid: dict[str, list],
    fixed_params: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Grid search calibration over allowed parameter combinations.

    Parameters
    ----------
    train_fn   : callable(params_dict) → validation_metric (lower = better)
    param_grid : {param_name: [value1, value2, ...]}
    fixed_params : parameters held fixed during calibration

    Returns
    -------
    best_params : dict of best parameter values
    results     : DataFrame with all grid results
    """
    fixed = fixed_params or {}
    param_names = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]

    rows = []
    best_metric = float("inf")
    best_params: dict = {}

    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        params.update(fixed)
        try:
            metric = train_fn(params)
        except Exception as e:
            logger.warning(f"Calibration trial failed for {params}: {e}")
            metric = float("inf")

        rows.append({**params, "metric": metric})

        if metric < best_metric:
            best_metric = metric
            best_params = dict(params)

    results = pd.DataFrame(rows)
    logger.info(
        f"Calibration complete: best_params={best_params}, "
        f"best_metric={best_metric:.6f}"
    )
    return best_params, results


def make_inner_val_split(
    train_data: pd.DataFrame | np.ndarray,
    inner_val_ratio: float = 0.20,
) -> tuple:
    """Create an inner validation split from training data (temporal).

    Returns (inner_train, inner_val) — both temporally ordered.
    """
    n = len(train_data)
    split = int(np.floor(n * (1.0 - inner_val_ratio)))

    if isinstance(train_data, pd.DataFrame):
        return train_data.iloc[:split], train_data.iloc[split:]
    elif isinstance(train_data, pd.Series):
        return train_data.iloc[:split], train_data.iloc[split:]
    else:
        return train_data[:split], train_data[split:]


def compute_calibration_metric(
    val_losses: np.ndarray,
    val_var: float,
    val_es: float,
    alpha: float = 0.99,
) -> float:
    """Compute the calibration objective metric on inner validation.

    Default: quantile loss at confidence level alpha.
    """
    ql_sum = sum(
        (loss - val_var) * alpha if loss > val_var else (val_var - loss) * (1.0 - alpha)
        for loss in val_losses
    )
    return float(ql_sum / max(len(val_losses), 1))


def log_calibration_summary(best_params: dict, results: pd.DataFrame) -> None:
    """Log a readable summary of calibration results."""
    lines = [
        "=== Calibration Summary ===",
        f"  Best parameters: {best_params}",
        f"  Best metric    : {results['metric'].min():.6f}",
        f"  Grid size      : {len(results)} combinations",
    ]
    for param in [k for k in best_params.keys() if k != "metric"]:
        subset = results.groupby(param)["metric"].mean()
        lines.append(f"  {param} sensitivity (mean metric): {subset.to_dict()}")
    logger.info("\n".join(lines))
