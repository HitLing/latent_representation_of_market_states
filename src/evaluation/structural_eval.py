"""Structural evaluation — independent plausibility and robustness checks.

This evaluation stage is independent of OOS backtesting.
It assesses the methodology from the inside:
  - Synthetic scenario plausibility
  - Dependence structure preservation
  - Bootstrap stability of VaR/ES
  - Sensitivity to alpha, m, M
  - Rolling-window robustness
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from src.generator.diagnostics import compute_generation_diagnostics
from src.tail.dependence import compare_dependence_structures
from src.normal_model.stability import (
    bootstrap_var_es_stability,
    compare_M_values_stability,
)
from src.regimes.stress_index import StressIndex

logger = logging.getLogger(__name__)


def run_plausibility_checks(
    historical_stress: pd.DataFrame,
    synthetic_stress: pd.DataFrame,
    normal_returns: pd.DataFrame,
) -> dict:
    """Plausibility checks for synthetic stress scenarios.

    Checks
    ------
    1. Marginal return statistics (mean, std, skew, kurtosis).
    2. Correlation structure preservation (Frobenius distance).
    3. Tail coverage (fraction of synthetic in the historical tail).
    """
    diag = compute_generation_diagnostics(historical_stress, synthetic_stress)
    dep = compare_dependence_structures(normal_returns, historical_stress, synthetic_stress)

    return {
        "generation_diagnostics": diag,
        "dependence_comparison": {
            "frob_stress_vs_normal": dep.get("frobenius_distance_stress_vs_normal"),
            "frob_synthetic_vs_stress": dep.get("frobenius_distance_synthetic_vs_stress"),
        },
        "plausibility_summary": {
            "correlation_preserved": (
                dep.get("frobenius_distance_synthetic_vs_stress", float("inf")) < 2.0
            ),
            "tail_coverage_ok": diag.get("tail_coverage", 0) > 0.05,
        },
    }


def run_sensitivity_alpha(
    returns: pd.DataFrame,
    weights: pd.Series,
    feature_cfg: dict,
    eval_fn: Callable[[float], dict],
    alpha_values: list[float],
) -> pd.DataFrame:
    """Sensitivity of risk metrics to alpha (stress threshold).

    eval_fn : callable(alpha) → {'VaR_0.99': ..., 'ES_0.99': ...}
    Returns DataFrame with alpha as index.
    """
    rows = []
    for alpha in alpha_values:
        try:
            metrics = eval_fn(alpha)
        except Exception as e:
            logger.warning(f"Alpha sensitivity failed for alpha={alpha}: {e}")
            metrics = {}
        metrics["alpha"] = alpha
        rows.append(metrics)
    df = pd.DataFrame(rows).set_index("alpha")
    logger.info(f"Alpha sensitivity: {len(df)} values tested")
    return df


def run_sensitivity_m(
    eval_fn: Callable[[float], dict],
    m_values: list[float],
) -> pd.DataFrame:
    """Sensitivity of risk metrics to m (synthetic mass ratio)."""
    rows = []
    for m in m_values:
        try:
            metrics = eval_fn(m)
        except Exception as e:
            logger.warning(f"m sensitivity failed for m={m}: {e}")
            metrics = {}
        metrics["m"] = m
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("m")


def run_sensitivity_M(
    embeddings: np.ndarray,
    loss_fn: Callable[[np.ndarray], tuple[float, float]],
    M_values: list[int],
    n_bootstrap: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Sensitivity of VaR/ES to prototype count M."""
    return compare_M_values_stability(
        embeddings, M_values, loss_fn, n_bootstrap=n_bootstrap, seed=seed
    )


def run_bootstrap_stability(
    embeddings: np.ndarray,
    prototype_model,
    loss_fn: Callable[[np.ndarray], tuple[float, float]],
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict:
    """Bootstrap stability of VaR/ES under prototype weight resampling."""
    return bootstrap_var_es_stability(
        embeddings, prototype_model, loss_fn, n_bootstrap=n_bootstrap, seed=seed
    )


def summarize_structural_eval(
    plausibility: dict,
    alpha_sensitivity: pd.DataFrame | None,
    m_sensitivity: pd.DataFrame | None,
    M_sensitivity: pd.DataFrame | None,
    bootstrap_stability: dict | None,
) -> dict:
    """Compile a structured summary of all structural evaluation results."""
    summary: dict = {"plausibility": plausibility.get("plausibility_summary", {})}

    if alpha_sensitivity is not None and len(alpha_sensitivity) > 0:
        for col in alpha_sensitivity.columns:
            if "VaR" in col or "ES" in col:
                summary[f"alpha_sensitivity_{col}_range"] = float(
                    alpha_sensitivity[col].max() - alpha_sensitivity[col].min()
                )

    if m_sensitivity is not None and len(m_sensitivity) > 0:
        for col in m_sensitivity.columns:
            if "VaR" in col or "ES" in col:
                summary[f"m_sensitivity_{col}_range"] = float(
                    m_sensitivity[col].max() - m_sensitivity[col].min()
                )

    if bootstrap_stability is not None:
        summary["bootstrap_var_cv"] = bootstrap_stability.get("var_cv")
        summary["bootstrap_es_cv"] = bootstrap_stability.get("es_cv")
        summary["bootstrap_stable"] = (
            (bootstrap_stability.get("var_cv", 1.0) or 1.0) < 0.10
        )

    return summary
