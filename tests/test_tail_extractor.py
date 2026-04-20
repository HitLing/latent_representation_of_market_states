"""Tests for tail extractor and severity buckets."""
import numpy as np
import pandas as pd
import pytest

from src.regimes.stress_index import StressIndex
from src.regimes.partition import create_partition
from src.tail.extractor import extract_tail, validate_tail_extract_no_overlap
from src.tail.severity import assign_severity_buckets, encode_severity_batch

_FEAT_CFG = {
    "volatility": {"short_window": 5, "long_window": 21},
    "correlation": {"rolling_window": 21},
}


@pytest.fixture
def stress_partition():
    rng = np.random.default_rng(7)
    n, k = 600, 5
    dates = pd.bdate_range("2017-01-01", periods=n)
    ret = pd.DataFrame(rng.normal(0, 0.015, (n, k)), index=dates,
                       columns=[f"S{i}" for i in range(k)])
    w = pd.Series(np.full(k, 0.2), index=ret.columns)
    cfg = {"regimes": {"alpha": 0.10}}
    si = StressIndex(cfg)
    score, labels = si.fit_transform(ret, w, _FEAT_CFG)
    part = create_partition(score, labels, si.alpha, si.threshold_)
    return part, ret, w, score


def test_tail_extract_no_overlap(stress_partition):
    part, ret, w, score = stress_partition
    cfg = {
        "regimes": {"alpha": 0.10},
        "tail": {"train_gen_fraction": 0.60, "anchor_fraction": 0.25,
                 "struct_eval_fraction": 0.15, "n_severity_buckets": 3},
    }
    extract = extract_tail(part, ret, w, cfg)
    # No date overlap
    a = set(extract.D_stress_train_gen.index)
    b = set(extract.D_stress_anchor.index)
    c = set(extract.D_stress_struct_eval.index)
    assert len(a & b) == 0
    assert len(a & c) == 0
    assert len(b & c) == 0


def test_tail_extract_sizes(stress_partition):
    part, ret, w, score = stress_partition
    cfg = {
        "regimes": {"alpha": 0.10},
        "tail": {"train_gen_fraction": 0.60, "anchor_fraction": 0.25,
                 "struct_eval_fraction": 0.15, "n_severity_buckets": 3},
    }
    extract = extract_tail(part, ret, w, cfg)
    total = len(extract.D_stress_train_gen) + len(extract.D_stress_anchor) + len(extract.D_stress_struct_eval)
    assert total == part.n_stress


def test_tail_extract_severity_labels_valid(stress_partition):
    part, ret, w, score = stress_partition
    cfg = {
        "regimes": {"alpha": 0.10},
        "tail": {"train_gen_fraction": 0.60, "anchor_fraction": 0.25,
                 "struct_eval_fraction": 0.15, "n_severity_buckets": 3},
    }
    extract = extract_tail(part, ret, w, cfg)
    buckets = extract.severity_labels.unique()
    assert all(0 <= b < 3 for b in buckets)


def test_severity_encoding_shape():
    labels = np.array([0, 1, 2, 1, 0])
    encoded = encode_severity_batch(labels, n_buckets=3)
    assert encoded.shape == (5, 3)
    assert (encoded.sum(axis=1) == 1).all()


def test_severity_encoding_one_hot():
    labels = np.array([0, 2])
    encoded = encode_severity_batch(labels, n_buckets=3)
    assert encoded[0, 0] == 1.0 and encoded[0, 1] == 0.0
    assert encoded[1, 2] == 1.0 and encoded[1, 0] == 0.0


def test_validate_no_overlap_raises():
    dates_a = pd.bdate_range("2020-01-01", periods=5)
    dates_b = pd.bdate_range("2020-01-08", periods=5)
    dates_overlap = pd.bdate_range("2020-01-14", periods=5)
    df_a = pd.DataFrame(index=dates_a)
    df_b = pd.DataFrame(index=dates_b)
    df_c = pd.DataFrame(index=dates_a)  # overlap with a
    with pytest.raises(ValueError, match="Circular contamination"):
        validate_tail_extract_no_overlap(df_a, df_b, df_c)
