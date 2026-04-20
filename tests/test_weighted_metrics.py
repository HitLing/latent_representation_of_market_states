"""Tests for weighted VaR/ES and backtesting."""
import numpy as np
import pandas as pd
import pytest

from src.risk.weighted_risk_metrics import (
    weighted_var, weighted_es, compute_risk_metrics, quantile_loss,
)
from src.risk.backtesting import (
    compute_var_exceedances, kupiec_test, christoffersen_test,
    run_full_backtest,
)


def _uniform_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def test_weighted_var_uniform():
    """VaR with uniform weights should approximate standard quantile."""
    rng = np.random.default_rng(0)
    losses = rng.normal(0, 1, 10000)
    w = _uniform_weights(len(losses))
    var = weighted_var(losses, w, 0.99)
    expected = np.quantile(losses, 0.99)
    assert abs(var - expected) < 0.05


def test_weighted_var_monotone():
    """Higher confidence level → higher VaR."""
    rng = np.random.default_rng(1)
    losses = rng.normal(0, 1, 1000)
    w = _uniform_weights(len(losses))
    var_95 = weighted_var(losses, w, 0.95)
    var_99 = weighted_var(losses, w, 0.99)
    assert var_99 >= var_95


def test_weighted_es_geq_var():
    """ES ≥ VaR by definition."""
    rng = np.random.default_rng(2)
    losses = rng.normal(0, 1, 1000)
    w = _uniform_weights(len(losses))
    for cl in [0.95, 0.99]:
        var = weighted_var(losses, w, cl)
        es = weighted_es(losses, w, cl)
        assert es >= var - 1e-8, f"ES < VaR at CL={cl}"


def test_weighted_es_uniform():
    """ES with uniform weights should be close to conditional expectation."""
    rng = np.random.default_rng(3)
    losses = rng.normal(0, 1, 10000)
    w = _uniform_weights(len(losses))
    cl = 0.99
    var = weighted_var(losses, w, cl)
    es = weighted_es(losses, w, cl)
    expected_es = losses[losses > var].mean()
    assert abs(es - expected_es) < 0.1


def test_compute_risk_metrics_keys():
    losses = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12])
    w = _uniform_weights(len(losses))
    metrics = compute_risk_metrics(losses, w, [0.95, 0.99])
    assert "VaR_0.95" in metrics
    assert "ES_0.95" in metrics
    assert "VaR_0.99" in metrics
    assert "ES_0.99" in metrics
    assert "mean_loss" in metrics


def test_quantile_loss_sign():
    """QL is always non-negative."""
    assert quantile_loss(0.02, 0.03, 0.99) >= 0
    assert quantile_loss(0.02, 0.01, 0.99) >= 0


def test_kupiec_correct_model():
    """A correctly calibrated VaR should NOT be rejected."""
    rng = np.random.default_rng(10)
    n = 500
    # Generate hits at approximately 1% rate
    hits = (rng.uniform(0, 1, n) < 0.01).astype(int)
    exc = pd.Series(hits)
    result = kupiec_test(exc, confidence_level=0.99, significance_level=0.05)
    assert not result["reject_H0"], "Correctly calibrated VaR should not be rejected"


def test_kupiec_wrong_model():
    """A badly over-estimated exceedance rate should be rejected."""
    # 10% exceedance rate when expecting 1%
    exc = pd.Series([1 if i % 10 == 0 else 0 for i in range(200)])
    result = kupiec_test(exc, confidence_level=0.99, significance_level=0.05)
    assert result["reject_H0"], "10% exceedance at 99% CL should be rejected"


def test_christoffersen_returns_dict():
    exc = pd.Series([0, 0, 1, 0, 0, 0, 0, 1, 0, 0] * 20)
    result = christoffersen_test(exc, confidence_level=0.99)
    assert "LR_uc" in result
    assert "LR_ind" in result
    assert "LR_cc" in result
    assert "p_value_cc" in result


def test_var_exceedances_correct():
    dates = pd.bdate_range("2020-01-01", periods=5)
    losses = pd.Series([0.01, 0.03, 0.02, 0.04, 0.01], index=dates)
    var = pd.Series([0.02, 0.02, 0.02, 0.02, 0.02], index=dates)
    exc = compute_var_exceedances(losses, var)
    assert list(exc.values) == [0, 1, 0, 1, 0]


def test_run_full_backtest_structure():
    rng = np.random.default_rng(20)
    dates = pd.bdate_range("2020-01-01", periods=250)
    losses = pd.Series(rng.normal(0.002, 0.01, 250), index=dates)
    var = pd.Series(0.025, index=dates)
    es = pd.Series(0.035, index=dates)
    result = run_full_backtest(losses, var, es, 0.99)
    assert "kupiec" in result
    assert "christoffersen" in result
    assert "es_backtest" in result
