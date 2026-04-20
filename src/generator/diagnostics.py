"""CVAE generator diagnostics — plausibility of generated scenarios."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_marginal_stats(
    returns: pd.DataFrame, label: str = ""
) -> pd.DataFrame:
    """Per-asset marginal statistics."""
    prefix = f"{label}_" if label else ""
    rows = {}
    for col in returns.columns:
        s = returns[col].dropna()
        rows[col] = {
            f"{prefix}mean": float(s.mean()),
            f"{prefix}std": float(s.std()),
            f"{prefix}skew": float(stats.skew(s)),
            f"{prefix}kurt": float(stats.kurtosis(s)),
            f"{prefix}p01": float(s.quantile(0.01)),
            f"{prefix}p05": float(s.quantile(0.05)),
        }
    return pd.DataFrame(rows).T


def compare_marginals(
    historical_stress: pd.DataFrame,
    synthetic_stress: pd.DataFrame,
) -> pd.DataFrame:
    """Relative differences between historical and synthetic marginal stats."""
    hist_cols = [c for c in historical_stress.columns]
    syn_cols = [c for c in synthetic_stress.columns if c != "severity_bucket"]
    common = [c for c in hist_cols if c in syn_cols]

    hist_stats = compute_marginal_stats(historical_stress[common], "hist")
    syn_stats = compute_marginal_stats(synthetic_stress[common], "syn")

    combined = hist_stats.join(syn_stats)

    for metric in ["mean", "std", "skew", "kurt"]:
        h_col = f"hist_{metric}"
        s_col = f"syn_{metric}"
        if h_col in combined.columns and s_col in combined.columns:
            denom = combined[h_col].abs().replace(0, np.nan)
            combined[f"rel_diff_{metric}"] = (combined[s_col] - combined[h_col]) / denom

    return combined


def compute_generation_diagnostics(
    historical_stress: pd.DataFrame,
    synthetic_stress: pd.DataFrame,
    severity_labels_hist: pd.Series | None = None,
) -> dict:
    """Full diagnostics report for generated scenarios."""
    asset_cols_hist = list(historical_stress.columns)
    asset_cols_syn = [c for c in synthetic_stress.columns if c != "severity_bucket"]
    common = [c for c in asset_cols_hist if c in asset_cols_syn]

    marginal_comp = compare_marginals(
        historical_stress[common], synthetic_stress[common]
    )

    # Correlation distance
    hist_corr = historical_stress[common].corr().values
    syn_corr = synthetic_stress[common].corr().values if len(synthetic_stress) > 1 else hist_corr
    frob_dist = float(np.linalg.norm(syn_corr - hist_corr, "fro"))

    # Tail coverage: fraction of synthetic in bottom 10% of historical
    tail_threshold = historical_stress[common].mean(axis=1).quantile(0.10)
    syn_mean = synthetic_stress[common].mean(axis=1)
    tail_coverage = float((syn_mean <= tail_threshold).mean())

    # Severity distribution
    sev_dist: dict = {}
    if "severity_bucket" in synthetic_stress.columns:
        vc = synthetic_stress["severity_bucket"].value_counts(normalize=True)
        sev_dist = {int(k): float(v) for k, v in vc.items()}

    return {
        "marginal_comparison": marginal_comp,
        "correlation_frobenius_distance": frob_dist,
        "tail_coverage": tail_coverage,
        "severity_distribution": sev_dist,
        "n_historical": len(historical_stress),
        "n_synthetic": len(synthetic_stress),
    }
