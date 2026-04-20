"""CVAE inference — generate synthetic stress scenarios.

Output is always stress scenarios ONLY (severity-conditioned).
Scenarios are inverse-scaled back to return space after generation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from src.generator.cvae import CVAE
from src.tail.severity import get_severity_conditioning_vector

logger = logging.getLogger(__name__)


def generate_stress_scenarios(
    model: CVAE,
    n_scenarios_per_bucket: dict[int, int],
    n_severity_buckets: int,
    scaler_params: dict,
    asset_names: list[str] | None = None,
    device: str = "cpu",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic stress scenarios conditioned on severity bucket.

    Parameters
    ----------
    n_scenarios_per_bucket : {bucket_id: n_to_generate}
    scaler_params          : dict with 'mean' and 'std' for inverse scaling
    asset_names            : column names for output DataFrame
    """
    torch.manual_seed(seed)
    model.eval()

    all_scenarios = []
    all_bucket_labels = []

    for bucket, n_gen in sorted(n_scenarios_per_bucket.items()):
        if n_gen <= 0:
            continue
        cond_vec = get_severity_conditioning_vector(bucket, n_severity_buckets)
        cond_tensor = torch.tensor(cond_vec, dtype=torch.float32)

        generated = model.generate(cond_tensor, n_samples=n_gen, device=device)
        generated_np = generated.cpu().numpy()
        all_scenarios.append(generated_np)
        all_bucket_labels.extend([bucket] * n_gen)

    if not all_scenarios:
        logger.warning("No scenarios generated (all bucket counts are 0).")
        n_assets = model.input_dim
        cols = asset_names or [f"asset_{i}" for i in range(n_assets)]
        return pd.DataFrame(columns=cols)

    X_scaled = np.concatenate(all_scenarios, axis=0)
    X_original = inverse_scale(X_scaled, scaler_params)

    n_assets = X_original.shape[1]
    cols = asset_names or [f"asset_{i}" for i in range(n_assets)]

    df = pd.DataFrame(X_original, columns=cols)
    df["severity_bucket"] = all_bucket_labels

    logger.info(
        f"Generated {len(df)} synthetic stress scenarios across "
        f"{len(n_scenarios_per_bucket)} severity buckets"
    )
    return df


def compute_generation_budget(
    D_stress_anchor: pd.DataFrame,
    m: float,
    severity_labels: pd.Series,
    n_severity_buckets: int,
) -> dict[int, int]:
    """Compute how many synthetic scenarios to generate per severity bucket.

    Total budget = round(m * |D_stress_anchor|).
    Distributed proportionally to severity bucket counts in anchor set.
    """
    anchor_dates = D_stress_anchor.index
    sev_in_anchor = severity_labels.loc[severity_labels.index.intersection(anchor_dates)]

    total_budget = max(1, round(m * len(D_stress_anchor)))

    bucket_counts = {b: int((sev_in_anchor == b).sum()) for b in range(n_severity_buckets)}
    total_anchor = sum(bucket_counts.values())

    if total_anchor == 0:
        # Fallback: uniform distribution
        per_bucket = total_budget // n_severity_buckets
        budget = {b: per_bucket for b in range(n_severity_buckets)}
    else:
        budget = {}
        allocated = 0
        buckets = sorted(bucket_counts.keys())
        for i, b in enumerate(buckets):
            if i == len(buckets) - 1:
                budget[b] = total_budget - allocated
            else:
                n_b = round(total_budget * bucket_counts[b] / total_anchor)
                budget[b] = max(0, n_b)
                allocated += budget[b]

    logger.info(f"Generation budget (m={m}): total={total_budget}, per bucket={budget}")
    return budget


def inverse_scale(X_scaled: np.ndarray, scaler_params: dict) -> np.ndarray:
    """Reverse standardisation: X = X_scaled * std + mean."""
    mean = scaler_params["mean"]
    std = scaler_params["std"]
    return X_scaled * std + mean


def validate_generated_scenarios(
    synthetic: pd.DataFrame,
    historical_stress: pd.DataFrame,
    tolerance_std_factor: float = 3.0,
) -> dict:
    """Sanity check: flag scenarios with returns far outside historical range."""
    asset_cols = [c for c in synthetic.columns if c != "severity_bucket"]
    hist_cols = [c for c in historical_stress.columns if c in asset_cols]

    hist = historical_stress[hist_cols]
    syn = synthetic[asset_cols].reindex(columns=hist_cols, fill_value=0.0)

    hist_std = hist.std()
    hist_mean = hist.mean()

    outlier_mask = (syn - hist_mean).abs() > tolerance_std_factor * hist_std
    n_flagged = int(outlier_mask.any(axis=1).sum())

    return {
        "n_scenarios": len(syn),
        "n_flagged_outlier": n_flagged,
        "flagged_fraction": n_flagged / max(len(syn), 1),
        "tolerance_std_factor": tolerance_std_factor,
    }
