"""Tests for scenario assembly and weight computation."""
import numpy as np
import pandas as pd
import pytest

from src.assembly.weighting import (
    compute_normal_prototype_weights,
    compute_historical_stress_weights,
    compute_synthetic_stress_weights,
    renormalize_to_distribution,
    validate_weights,
)
from src.assembly.scenario_assembly import (
    ScenarioAssembly, assemble_baseline_hs,
)
from src.normal_model.prototypes import fit_prototypes


# ---------------------------------------------------------------------------
# Weighting tests
# ---------------------------------------------------------------------------

def test_normal_prototype_weights_sum():
    assignments = np.array([0, 0, 1, 2, 1, 0])
    w = compute_normal_prototype_weights(assignments, M=3, normal_fraction=0.7)
    assert abs(w.sum() - 0.7) < 1e-5


def test_historical_stress_weights_uniform():
    w = compute_historical_stress_weights(n_stress_obs=10, stress_fraction=0.3)
    assert len(w) == 10
    assert abs(w.sum() - 0.3) < 1e-5
    assert np.allclose(w, w[0])  # uniform


def test_synthetic_weights_constraint():
    w_hist = np.full(10, 0.03)   # sum = 0.3
    m = 0.5
    w_syn = compute_synthetic_stress_weights(
        n_synthetic=5,
        historical_stress_total_weight=w_hist.sum(),
        m=m,
    )
    # sum(w_syn) should equal m * sum(w_hist)
    assert abs(w_syn.sum() - m * w_hist.sum()) < 1e-5


def test_renormalize_sums_to_one():
    w_n = np.array([0.5, 0.3])
    w_h = np.array([0.1, 0.05])
    w_s = np.array([0.03])
    w_n2, w_h2, w_s2 = renormalize_to_distribution(w_n, w_h, w_s)
    total = w_n2.sum() + w_h2.sum() + w_s2.sum()
    assert abs(total - 1.0) < 1e-5


def test_validate_weights_pass():
    w_n = np.array([0.5, 0.2])
    w_h = np.array([0.1, 0.1])
    w_s = np.array([0.05, 0.05])
    checks = validate_weights(w_n, w_h, w_s, m=0.5)
    assert checks["weights_sum_to_one"]["passed"]
    assert checks["all_non_negative"]["passed"]


# ---------------------------------------------------------------------------
# ScenarioAssembly tests
# ---------------------------------------------------------------------------

def _make_assembly(n_normal=20, n_hist=8, n_syn=4, n_assets=5):
    cols = [f"A{i}" for i in range(n_assets)]
    rng = np.random.default_rng(0)
    normal = pd.DataFrame(rng.normal(0, 0.01, (n_normal, n_assets)), columns=cols)
    hist = pd.DataFrame(rng.normal(-0.02, 0.03, (n_hist, n_assets)), columns=cols)
    syn = pd.DataFrame(rng.normal(-0.025, 0.04, (n_syn, n_assets)), columns=cols)

    w_n = np.full(n_normal, 0.5 / n_normal)
    w_h = np.full(n_hist, 0.3 / n_hist)
    m = 0.5
    w_s = np.full(n_syn, m * w_h.sum() / n_syn)
    # Renorm
    total = w_n.sum() + w_h.sum() + w_s.sum()
    return ScenarioAssembly(
        normal_scenarios=normal,
        hist_stress_scenarios=hist,
        syn_stress_scenarios=syn,
        normal_weights=w_n / total,
        hist_stress_weights=w_h / total,
        syn_stress_weights=w_s / total,
        m=m,
        n_total_scenarios=n_normal + n_hist + n_syn,
        normal_fraction=w_n.sum() / total,
        stress_fraction=(w_h.sum() + w_s.sum()) / total,
        synthetic_mass=w_s.sum() / total,
        historical_stress_mass=w_h.sum() / total,
    )


def test_assembly_weights_sum_to_one():
    asm = _make_assembly()
    assert abs(asm.get_all_weights().sum() - 1.0) < 1e-5


def test_assembly_n_total_scenarios():
    asm = _make_assembly(n_normal=20, n_hist=8, n_syn=4)
    assert asm.n_total_scenarios == 32


def test_assembly_source_labels():
    asm = _make_assembly(n_normal=20, n_hist=8, n_syn=4)
    labels = asm.get_source_labels()
    assert labels.count("normal_proto") == 20
    assert labels.count("stress_hist") == 8
    assert labels.count("stress_syn") == 4


def test_assembly_validate_passes():
    asm = _make_assembly()
    asm.validate()  # should not raise


def test_assembly_to_dataframe():
    asm = _make_assembly()
    df = asm.to_dataframe()
    assert "weight" in df.columns
    assert "source" in df.columns
    assert abs(df["weight"].sum() - 1.0) < 1e-5


def test_baseline_hs_assembly():
    rng = np.random.default_rng(5)
    ret = pd.DataFrame(rng.normal(0, 0.01, (100, 4)),
                       columns=["A", "B", "C", "D"])
    asm = assemble_baseline_hs(ret)
    assert abs(asm.get_all_weights().sum() - 1.0) < 1e-5
    assert len(asm.hist_stress_scenarios) == 0
    assert len(asm.syn_stress_scenarios) == 0


def test_assembly_save_load(tmp_path):
    asm = _make_assembly()
    path = tmp_path / "assembly.parquet"
    asm.save(path)
    loaded = ScenarioAssembly.load(path)
    assert abs(loaded.get_all_weights().sum() - 1.0) < 1e-5
    assert loaded.n_total_scenarios == asm.n_total_scenarios
