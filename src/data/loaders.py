"""Data loaders — real market data and synthetic demo data generator.

Synthetic data is generated when real data is absent or ``use_synthetic=True``.
The generator produces realistic multivariate return series with configurable
stress episodes so that the full methodology pipeline can be exercised
without proprietary market data.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_market_data(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load market returns and portfolio weights.

    Returns
    -------
    returns : pd.DataFrame
        DatetimeIndex, columns = asset names, values = daily simple returns.
    weights : pd.Series
        index = asset names, values = portfolio weights (sum to 1).

    If ``cfg['use_synthetic']`` is True (or the raw file does not exist),
    the synthetic generator is called instead.
    """
    use_syn = cfg.get("use_synthetic", True)
    raw_path = Path(cfg.get("raw_data_path", "data/raw/market_data.parquet"))

    if use_syn or not raw_path.exists():
        logger.info("Generating synthetic market data (use_synthetic=True or file missing)")
        returns, weights = generate_synthetic_data(cfg)
    else:
        logger.info(f"Loading real market data from {raw_path}")
        returns = _load_raw_file(raw_path)
        weights = load_portfolio_weights(
            cfg.get("portfolio", {}).get("weights_path"),
            returns.columns.tolist(),
        )

    validate_returns(returns)
    logger.info(
        f"Loaded data: {len(returns)} days, {returns.shape[1]} assets, "
        f"{returns.index[0].date()} – {returns.index[-1].date()}"
    )
    return returns, weights


def generate_synthetic_data(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic market data with stress episodes.

    The data-generating process:
    1. A Markov chain determines the regime sequence (normal/stress).
    2. In each regime, correlated returns are drawn via Cholesky decomposition.
    3. Stress episodes feature higher volatility, higher correlation, and a
       negative drift (crisis asymmetry).
    4. A business-day DatetimeIndex starting at 2015-01-02 is attached.
    5. The resulting parquet is saved to ``data/raw/synthetic_market_data.parquet``.

    Parameters are read from ``cfg['synthetic']``.
    """
    syn = cfg.get("synthetic", {})
    n_assets: int = syn.get("n_assets", 10)
    n_days: int = syn.get("n_days", 2000)
    seed: int = syn.get("seed", 42)
    normal_vol: float = syn.get("normal_vol", 0.010)
    stress_vol: float = syn.get("stress_vol", 0.040)
    stress_prob: float = syn.get("stress_prob", 0.08)
    stress_dur_mean: float = syn.get("stress_duration_mean", 10)
    corr_normal: float = syn.get("correlation_normal", 0.30)
    corr_stress: float = syn.get("correlation_stress", 0.70)

    rng = np.random.default_rng(seed)

    asset_names = syn.get("asset_names") or [f"Asset_{i:02d}" for i in range(n_assets)]

    # ------------------------------------------------------------------
    # Build correlation matrices
    # ------------------------------------------------------------------
    def _corr_matrix(rho: float, n: int) -> np.ndarray:
        C = np.full((n, n), rho)
        np.fill_diagonal(C, 1.0)
        return C

    C_normal = _corr_matrix(corr_normal, n_assets)
    C_stress = _corr_matrix(corr_stress, n_assets)
    L_normal = np.linalg.cholesky(C_normal)
    L_stress = np.linalg.cholesky(C_stress)

    # ------------------------------------------------------------------
    # Generate regime sequence using geometric sojourn-time model
    # ------------------------------------------------------------------
    regimes = np.zeros(n_days, dtype=int)  # 0 = normal, 1 = stress
    i = 0
    while i < n_days:
        if rng.random() < stress_prob and regimes[max(0, i - 1)] == 0:
            duration = max(1, int(rng.exponential(stress_dur_mean)))
            end = min(i + duration, n_days)
            regimes[i:end] = 1
            i = end
        else:
            regimes[i] = 0
            i += 1

    n_stress = regimes.sum()
    n_normal = n_days - n_stress
    logger.info(
        f"Synthetic regime split: {n_normal} normal days ({n_normal/n_days:.1%}), "
        f"{n_stress} stress days ({n_stress/n_days:.1%})"
    )

    # ------------------------------------------------------------------
    # Generate correlated returns per regime
    # ------------------------------------------------------------------
    returns_arr = np.zeros((n_days, n_assets))
    for t in range(n_days):
        z = rng.standard_normal(n_assets)
        if regimes[t] == 0:
            r = normal_vol * (L_normal @ z)
            # Small positive drift in normal regime
            r += 0.0003
        else:
            r = stress_vol * (L_stress @ z)
            # Negative drift / crisis asymmetry
            r -= 0.005

        returns_arr[t] = r

    # ------------------------------------------------------------------
    # Build DataFrame
    # ------------------------------------------------------------------
    bdays = pd.bdate_range(start="2015-01-02", periods=n_days)
    returns = pd.DataFrame(returns_arr, index=bdays, columns=asset_names)
    returns.index.name = "date"

    # Save synthetic data
    save_path = Path("data/raw/synthetic_market_data.parquet")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(save_path)
    logger.info(f"Saved synthetic data -> {save_path}")

    # Equal portfolio weights
    weights = pd.Series(
        np.full(n_assets, 1.0 / n_assets), index=asset_names, name="weight"
    )

    return returns, weights


def load_portfolio_weights(
    path: str | Path | None, asset_names: list[str]
) -> pd.Series:
    """Load portfolio weights from file, or return equal weights if path is None."""
    if path is None:
        n = len(asset_names)
        return pd.Series(np.full(n, 1.0 / n), index=asset_names, name="weight")

    path = Path(path)
    if path.suffix == ".parquet":
        w = pd.read_parquet(path).squeeze()
    else:
        w = pd.read_csv(path, index_col=0).squeeze()

    w = w.reindex(asset_names).fillna(0.0)
    total = w.sum()
    if total == 0:
        raise ValueError("Portfolio weights sum to zero.")
    return w / total


def validate_returns(returns: pd.DataFrame) -> None:
    """Validate return DataFrame for common data-quality issues."""
    n, k = returns.shape

    # Missing values
    missing_frac = returns.isnull().mean()
    bad_cols = missing_frac[missing_frac > 0.1].index.tolist()
    if bad_cols:
        logger.warning(f"Columns with >10% missing values: {bad_cols}")

    # Zero-variance assets
    zero_var = returns.std()[returns.std() < 1e-10].index.tolist()
    if zero_var:
        logger.warning(f"Zero-variance assets detected: {zero_var}")

    # Extreme outliers (> 50% single-day return)
    extreme = (returns.abs() > 0.50).any(axis=1).sum()
    if extreme > 0:
        logger.warning(f"{extreme} observations with |return| > 50%")


def _load_raw_file(path: Path) -> pd.DataFrame:
    """Load raw market data file (parquet or csv)."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".txt"):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df
