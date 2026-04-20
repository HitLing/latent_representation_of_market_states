"""Tests for distribution partition."""
import numpy as np
import pandas as pd
import pytest

from src.regimes.stress_index import StressIndex
from src.regimes.partition import (
    create_partition, validate_partition_quality, RegimePartition,
)

_FEAT_CFG = {
    "volatility": {"short_window": 5, "long_window": 21},
    "correlation": {"rolling_window": 21},
}


@pytest.fixture
def partition_fixture():
    rng = np.random.default_rng(1)
    n, k = 500, 4
    dates = pd.bdate_range("2018-01-01", periods=n)
    ret = pd.DataFrame(rng.normal(0, 0.01, (n, k)), index=dates,
                       columns=[f"A{i}" for i in range(k)])
    w = pd.Series(np.full(k, 1.0 / k), index=ret.columns)
    cfg = {"regimes": {"alpha": 0.10}}
    si = StressIndex(cfg)
    score, labels = si.fit_transform(ret, w, _FEAT_CFG)
    part = create_partition(score, labels, si.alpha, si.threshold_)
    return part, ret


def test_partition_sizes_consistent(partition_fixture):
    part, _ = partition_fixture
    assert part.n_normal + part.n_stress == part.n_total


def test_partition_no_overlap(partition_fixture):
    part, _ = partition_fixture
    overlap = set(part.normal_dates) & set(part.stress_dates)
    assert len(overlap) == 0, "Normal and stress dates must not overlap"


def test_partition_cover_all_dates(partition_fixture):
    part, _ = partition_fixture
    all_dates = set(part.normal_dates) | set(part.stress_dates)
    assert len(all_dates) == part.n_total


def test_partition_stress_fraction(partition_fixture):
    part, _ = partition_fixture
    assert abs(part.stress_fraction - part.n_stress / part.n_total) < 1e-9


def test_filter_data_dataframe(partition_fixture):
    part, ret = partition_fixture
    normal_ret = part.filter_data(ret, "normal")
    stress_ret = part.filter_data(ret, "stress")
    assert len(normal_ret) == part.n_normal
    assert len(stress_ret) == part.n_stress


def test_filter_data_ndarray(partition_fixture):
    part, ret = partition_fixture
    arr = ret.to_numpy()
    # arr may be longer than partition if warm-up dropped some rows
    arr_trimmed = arr[:part.n_total]
    normal_arr = part.filter_data(arr_trimmed, "normal")
    assert len(normal_arr) == part.n_normal


def test_partition_to_dataframe(partition_fixture):
    part, _ = partition_fixture
    df = part.to_dataframe()
    assert "stress_score" in df.columns
    assert "is_stress" in df.columns
    assert len(df) == part.n_total


def test_validate_partition_quality(partition_fixture):
    part, _ = partition_fixture
    checks = validate_partition_quality(part)
    assert "enough_stress_obs" in checks
    assert "enough_normal_obs" in checks
    # With 500 obs, both should pass
    assert checks["enough_normal_obs"]["passed"]
    assert checks["enough_stress_obs"]["passed"]
