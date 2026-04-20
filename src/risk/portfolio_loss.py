"""Transparent portfolio loss computation.

Loss for scenario i = −1 · Σ_j w_j · r_{ij}
(positive loss = negative portfolio return)

No neural prediction.  No black box.  Explicit linear formula.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.assembly.scenario_assembly import ScenarioAssembly

logger = logging.getLogger(__name__)


def compute_scenario_losses(
    assembly: ScenarioAssembly,
    weights: pd.Series,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute portfolio loss for each scenario in the assembly.

    Returns
    -------
    losses  : np.ndarray shape (n_scenarios,), positive = loss
    weights : np.ndarray shape (n_scenarios,), scenario probability weights
    """
    scenarios = assembly.get_all_scenarios()
    scenario_weights = assembly.get_all_weights()

    # Identify asset return columns
    asset_cols = feature_columns or [c for c in scenarios.columns if c in weights.index]
    if not asset_cols:
        raise ValueError(
            "No matching columns between scenario data and portfolio weights."
        )

    w = weights.reindex(asset_cols, fill_value=0.0)
    w = w / w.sum()

    scenario_returns = scenarios[asset_cols].to_numpy()
    losses = -(scenario_returns @ w.values)  # loss = -return

    logger.info(
        f"Computed {len(losses)} scenario losses: "
        f"mean={losses.mean():.4f}, max={losses.max():.4f}"
    )
    return losses, scenario_weights


def compute_portfolio_loss_series(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Compute realised portfolio loss time series.

    Returns pd.Series (same DatetimeIndex), positive = loss days.
    """
    w = weights.reindex(returns.columns, fill_value=0.0)
    w = w / w.sum()
    port_return = returns @ w
    losses = -port_return
    losses.name = "portfolio_loss"
    return losses


def build_loss_distribution(
    losses: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build weighted empirical loss CDF.

    Returns
    -------
    sorted_losses        : np.ndarray, ascending
    cumulative_weights   : np.ndarray, cumulative probability
    """
    order = np.argsort(losses)
    sorted_losses = losses[order]
    sorted_weights = weights[order]
    cumulative_weights = np.cumsum(sorted_weights)
    return sorted_losses, cumulative_weights
