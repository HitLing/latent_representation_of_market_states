"""Data preprocessing — cleaning, normalisation, and missing-value handling."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def preprocess_returns(returns: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Full preprocessing pipeline.

    Steps
    -----
    1. Forward-fill missing values (up to ``max_consecutive_fill`` bars).
    2. Drop columns whose overall missing fraction exceeds ``max_missing_frac``.
    3. Winsorise extreme returns if ``winsorize_returns=True``.
    4. Log summary statistics.

    Returns cleaned DataFrame with same DatetimeIndex.
    """
    prep = cfg.get("preprocessing", {})
    max_missing = prep.get("max_missing_frac", 0.10)
    max_consec = prep.get("max_consecutive_fill", 5)
    do_wins = prep.get("winsorize_returns", True)
    wins_q = prep.get("winsorize_quantile", 0.001)

    df = returns.copy()

    # 1. Limited forward fill
    df = forward_fill_limited(df, max_consecutive=max_consec)

    # 2. Drop high-missing columns
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > max_missing].index.tolist()
    if drop_cols:
        logger.warning(f"Dropping {len(drop_cols)} columns (too many NaN): {drop_cols}")
        df = df.drop(columns=drop_cols)

    # 3. Fill any remaining NaN with 0 (after policy above)
    remaining_na = df.isnull().sum().sum()
    if remaining_na > 0:
        logger.warning(f"Filling {remaining_na} remaining NaN with 0.0")
        df = df.fillna(0.0)

    # 4. Winsorise
    if do_wins:
        df = winsorize_returns(df, quantile=wins_q)

    logger.info(
        f"Preprocessed returns: {df.shape}, "
        f"mean={df.mean().mean():.5f}, std={df.std().mean():.5f}"
    )
    return df


def forward_fill_limited(df: pd.DataFrame, max_consecutive: int = 5) -> pd.DataFrame:
    """Forward fill, but reset runs longer than *max_consecutive* back to NaN."""
    filled = df.ffill()

    # Re-identify positions that were NaN in original and count consecutive run length
    was_nan = df.isnull()
    for col in df.columns:
        nan_col = was_nan[col]
        if not nan_col.any():
            continue
        # Count length of each consecutive NaN run
        run_len = nan_col.astype(int).groupby((nan_col != nan_col.shift()).cumsum()).cumsum()
        # Positions where run exceeds limit
        bad_mask = nan_col & (run_len > max_consecutive)
        filled.loc[bad_mask, col] = np.nan

    return filled


def winsorize_returns(returns: pd.DataFrame, quantile: float = 0.001) -> pd.DataFrame:
    """Winsorise each asset return series at (quantile, 1-quantile)."""
    df = returns.copy()
    for col in df.columns:
        lo = df[col].quantile(quantile)
        hi = df[col].quantile(1.0 - quantile)
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price series."""
    return np.log(prices / prices.shift(1))


def standardize_features(
    features: np.ndarray,
    fit_on: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Z-score standardise features.

    Fits on ``fit_on`` if provided (the training set), otherwise on ``features``.

    Returns
    -------
    standardized : np.ndarray, same shape as features
    scaler_params : dict with keys 'mean' and 'std'
    """
    source = fit_on if fit_on is not None else features
    mean = np.nanmean(source, axis=0)
    std = np.nanstd(source, axis=0)
    std = np.where(std < 1e-10, 1.0, std)  # avoid divide-by-zero
    scaled = (features - mean) / std
    return scaled, {"mean": mean, "std": std}


def apply_scaler(features: np.ndarray, scaler_params: dict) -> np.ndarray:
    """Apply precomputed z-score scaler to new data."""
    mean = scaler_params["mean"]
    std = scaler_params["std"]
    return (features - mean) / std
