"""Explicit weight computation for scenario assembly.

Weight structure
----------------
The final scenario set S_final = S_normal_proto ∪ S_stress_hist ∪ S_stress_syn
carries a proper discrete probability distribution.

1. Normal prototype weights:
   w_j^normal = (n_assigned_j / N_normal) · (N_normal / N_total)

2. Historical stress weights:
   w_i^hist   = (1 / N_stress) · (N_stress / N_total)

3. Synthetic stress weights (mandatory constraint):
   Σ w_syn = m · Σ w_hist_stress

All weights sum to 1.0 globally after renormalisation.
Synthetic scenarios ENRICH the tail — they do NOT overwrite it.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_normal_prototype_weights(
    prototype_assignments: np.ndarray,
    M: int,
    normal_fraction: float,
) -> np.ndarray:
    """Weights for M normal prototypes.

    w_j = (n_assigned_j / N_normal) · normal_fraction

    Returns ndarray shape (M,), sums to ``normal_fraction``.
    """
    N_normal = len(prototype_assignments)
    cluster_sizes = np.bincount(prototype_assignments, minlength=M).astype(float)
    local_weights = cluster_sizes / (N_normal + 1e-12)
    weights = local_weights * normal_fraction
    assert abs(weights.sum() - normal_fraction) < 1e-5, (
        f"Normal weights sum to {weights.sum():.6f} ≠ {normal_fraction:.6f}"
    )
    return weights


def compute_historical_stress_weights(
    n_stress_obs: int,
    stress_fraction: float,
) -> np.ndarray:
    """Equal weights for historical stress observations.

    Each observation gets weight = stress_fraction / n_stress_obs.
    Returns ndarray shape (n_stress_obs,), sums to ``stress_fraction``.
    """
    if n_stress_obs == 0:
        return np.array([], dtype=float)
    weights = np.full(n_stress_obs, stress_fraction / n_stress_obs)
    return weights


def compute_synthetic_stress_weights(
    n_synthetic: int,
    historical_stress_total_weight: float,
    m: float,
) -> np.ndarray:
    """Equal weights for synthetic stress scenarios.

    MANDATORY CONSTRAINT: Σ w_syn = m · Σ w_hist_stress

    This guarantees synthetic scenarios enrich — not replace — the tail.
    """
    if n_synthetic == 0:
        return np.array([], dtype=float)
    total_syn_mass = m * historical_stress_total_weight
    weights = np.full(n_synthetic, total_syn_mass / n_synthetic)
    return weights


def renormalize_to_distribution(
    normal_weights: np.ndarray,
    hist_stress_weights: np.ndarray,
    syn_stress_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Renormalise all weights to sum to 1.0 globally.

    Preserves relative mass between groups.
    """
    total = (
        normal_weights.sum()
        + hist_stress_weights.sum()
        + syn_stress_weights.sum()
    )
    if total < 1e-12:
        raise ValueError("Total weight is zero — cannot normalise.")

    w_n = normal_weights / total
    w_h = hist_stress_weights / total
    w_s = syn_stress_weights / total

    assert abs(w_n.sum() + w_h.sum() + w_s.sum() - 1.0) < 1e-5, (
        "Renormalised weights do not sum to 1."
    )
    return w_n, w_h, w_s


def validate_weights(
    normal_weights: np.ndarray,
    hist_weights: np.ndarray,
    syn_weights: np.ndarray,
    m: float,
) -> dict:
    """Validate weight structure for methodological correctness."""
    checks: dict[str, dict] = {}

    total = normal_weights.sum() + hist_weights.sum() + syn_weights.sum()
    checks["weights_sum_to_one"] = {
        "passed": abs(total - 1.0) < 1e-4,
        "value": total,
    }

    checks["all_non_negative"] = {
        "passed": bool(
            (normal_weights >= 0).all()
            and (hist_weights >= 0).all()
            and (syn_weights >= 0).all()
        ),
        "value": None,
    }

    checks["no_nan_inf"] = {
        "passed": bool(
            np.isfinite(normal_weights).all()
            and np.isfinite(hist_weights).all()
            and (np.isfinite(syn_weights).all() if len(syn_weights) > 0 else True)
        ),
        "value": None,
    }

    # Synthetic mass constraint
    hist_total = float(hist_weights.sum())
    syn_total = float(syn_weights.sum())
    expected_syn = m * hist_total
    syn_mass_ok = (
        abs(syn_total - expected_syn) < 1e-4
        if len(syn_weights) > 0
        else True
    )
    checks["synthetic_mass_constraint"] = {
        "passed": syn_mass_ok,
        "syn_mass": syn_total,
        "expected": expected_syn,
    }

    for name, result in checks.items():
        status = "PASS" if result["passed"] else "FAIL"
        logger.log(
            logging.INFO if result["passed"] else logging.WARNING,
            f"Weight check [{status}] {name}"
        )
    return checks
