"""Tests for stress index computation."""
import numpy as np
import pandas as pd
import pytest

from src.features.stress_indicators import (
    compute_return_shock_indicator,
    compute_volatility_spike_indicator,
    compute_correlation_stress_indicator,
    compute_portfolio_loss_proxy_indicator,
    rank_normalize,
    compute_all_indicators,
)
from src.regimes.stress_index import StressIndex


@pytest.fixture
def sample_returns() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n, k = 300, 5
    dates = pd.bdate_range("2020-01-01", periods=n)
    ret = pd.DataFrame(rng.normal(0, 0.01, (n, k)), index=dates,
                       columns=[f"A{i}" for i in range(k)])
    w = pd.Series(np.full(k, 1.0 / k), index=ret.columns)
    return ret, w


_FEAT_CFG = {
    "volatility": {"short_window": 5, "long_window": 21},
    "correlation": {"rolling_window": 21},
}


def test_return_shock_non_negative(sample_returns):
    ret, w = sample_returns
    s = compute_return_shock_indicator(ret, _FEAT_CFG)
    assert (s >= 0).all(), "Return shock indicator must be non-negative"


def test_vol_spike_positive(sample_returns):
    ret, w = sample_returns
    s = compute_volatility_spike_indicator(ret, _FEAT_CFG)
    assert s.dropna().shape[0] > 0


def test_corr_stress_range(sample_returns):
    ret, w = sample_returns
    s = compute_correlation_stress_indicator(ret, _FEAT_CFG)
    valid = s.dropna()
    assert (valid >= 0).all()
    assert (valid <= 1).all()


def test_portfolio_loss_proxy_non_negative(sample_returns):
    ret, w = sample_returns
    s = compute_portfolio_loss_proxy_indicator(ret, w, _FEAT_CFG)
    assert (s >= 0).all()


def test_rank_normalize_range(sample_returns):
    ret, w = sample_returns
    raw = compute_return_shock_indicator(ret, _FEAT_CFG)
    normalised = rank_normalize(raw)
    valid = normalised.dropna()
    assert (valid >= 0).all()
    assert (valid <= 1).all()


def test_rank_normalize_fit_on_train(sample_returns):
    ret, w = sample_returns
    raw = compute_return_shock_indicator(ret, _FEAT_CFG)
    train_raw = raw.iloc[:200]
    full_norm = rank_normalize(raw, fit_series=train_raw)
    assert len(full_norm) == len(raw)


def test_compute_all_indicators_shape(sample_returns):
    ret, w = sample_returns
    out = compute_all_indicators(ret, w, _FEAT_CFG)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {
        "return_shock", "volatility_spike", "correlation_stress", "portfolio_loss_proxy"
    }
    assert len(out) == len(ret)


def test_all_indicators_in_unit_interval(sample_returns):
    ret, w = sample_returns
    out = compute_all_indicators(ret, w, _FEAT_CFG)
    for col in out.columns:
        valid = out[col].dropna()
        assert (valid >= -1e-9).all(), f"{col} has negative values"
        assert (valid <= 1 + 1e-9).all(), f"{col} exceeds 1"


def test_stress_index_fit_transform(sample_returns):
    ret, w = sample_returns
    cfg = {"regimes": {"alpha": 0.10}}
    si = StressIndex(cfg)
    score, labels = si.fit_transform(ret, w, _FEAT_CFG)
    assert len(score) == len(ret)
    assert labels.dtype == bool
    assert si.threshold_ is not None


def test_stress_index_threshold_matches_alpha(sample_returns):
    ret, w = sample_returns
    alpha = 0.10
    cfg = {"regimes": {"alpha": alpha}}
    si = StressIndex(cfg)
    score, labels = si.fit_transform(ret, w, _FEAT_CFG)
    empirical_fraction = labels.mean()
    # Should be close to alpha (within ±3% due to rounding)
    assert abs(empirical_fraction - alpha) < 0.05


def test_stress_index_sensitivity(sample_returns):
    ret, w = sample_returns
    cfg = {"regimes": {"alpha": 0.10}}
    si = StressIndex(cfg)
    si.fit(ret, w, _FEAT_CFG)
    sens = si.sensitivity_analysis(ret, w, _FEAT_CFG, [0.05, 0.10, 0.15])
    assert "n_stress" in sens.columns
    assert len(sens) == 3
    # More stress obs with higher alpha
    assert sens.loc[0.15, "n_stress"] >= sens.loc[0.05, "n_stress"]


def test_stress_index_serialization(sample_returns):
    ret, w = sample_returns
    cfg = {"regimes": {"alpha": 0.10}}
    si = StressIndex(cfg)
    si.fit(ret, w, _FEAT_CFG)
    state = si.to_dict()
    si2 = StressIndex.from_dict(state, cfg)
    assert si2.threshold_ == pytest.approx(si.threshold_, rel=1e-6)
