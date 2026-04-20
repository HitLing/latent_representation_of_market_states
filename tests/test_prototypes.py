"""Tests for normal-state prototype model."""
import numpy as np
import pytest
import tempfile
from pathlib import Path

from src.normal_model.prototypes import (
    fit_prototypes, assign_to_prototypes,
    compute_prototype_reconstruction_quality,
    select_M_by_reconstruction,
    save_prototypes, load_prototypes,
)


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, (200, 8)).astype(np.float32)


def test_fit_prototypes_shapes(embeddings):
    proto = fit_prototypes(embeddings, M=10, seed=0)
    assert proto.centers.shape == (10, 8)
    assert proto.weights.shape == (10,)
    assert proto.assignments.shape == (200,)
    assert proto.cluster_sizes.shape == (10,)


def test_prototype_weights_sum_to_one(embeddings):
    proto = fit_prototypes(embeddings, M=15, seed=0)
    assert abs(proto.weights.sum() - 1.0) < 1e-5


def test_prototype_weights_non_negative(embeddings):
    proto = fit_prototypes(embeddings, M=10, seed=0)
    assert (proto.weights >= 0).all()


def test_assignments_valid_range(embeddings):
    M = 10
    proto = fit_prototypes(embeddings, M=M, seed=0)
    assert proto.assignments.min() >= 0
    assert proto.assignments.max() < M


def test_assign_to_prototypes(embeddings):
    proto = fit_prototypes(embeddings, M=10, seed=0)
    new_emb = np.random.default_rng(99).normal(0, 1, (50, 8)).astype(np.float32)
    assignments = assign_to_prototypes(new_emb, proto)
    assert assignments.shape == (50,)
    assert assignments.min() >= 0
    assert assignments.max() < 10


def test_reconstruction_quality_keys(embeddings):
    proto = fit_prototypes(embeddings, M=10, seed=0)
    quality = compute_prototype_reconstruction_quality(embeddings, proto)
    assert "mean_reconstruction_error" in quality
    assert "explained_variance_ratio" in quality
    assert quality["explained_variance_ratio"] >= 0
    assert quality["explained_variance_ratio"] <= 1.0 + 1e-6


def test_reconstruction_quality_improves_with_M(embeddings):
    quality_5 = compute_prototype_reconstruction_quality(
        embeddings, fit_prototypes(embeddings, M=5, seed=0)
    )
    quality_20 = compute_prototype_reconstruction_quality(
        embeddings, fit_prototypes(embeddings, M=20, seed=0)
    )
    assert quality_20["mean_reconstruction_error"] <= quality_5["mean_reconstruction_error"] + 1e-6


def test_select_M_by_reconstruction_shape(embeddings):
    df = select_M_by_reconstruction(embeddings, M_values=[5, 10, 15], seed=0)
    assert len(df) == 3
    assert "mean_reconstruction_error" in df.columns


def test_save_load_prototypes(embeddings):
    proto = fit_prototypes(embeddings, M=10, seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prototypes.npz"
        save_prototypes(proto, path)
        loaded = load_prototypes(path)
    assert np.allclose(loaded.centers, proto.centers)
    assert np.allclose(loaded.weights, proto.weights)
    assert loaded.M == proto.M


def test_random_method(embeddings):
    proto = fit_prototypes(embeddings, M=10, method="random", seed=0)
    assert proto.centers.shape == (10, 8)
    assert abs(proto.weights.sum() - 1.0) < 1e-5
