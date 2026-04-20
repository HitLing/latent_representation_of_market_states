"""CLI: Train VAE and build normal-state prototypes.

Usage
-----
python -m scripts.train_normal_model --config configs/experiment/full_method.yaml
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.utils.logging_utils import setup_logging
from src.utils.seeds import set_seed
from src.utils.config import load_experiment_config
from src.utils.io import (
    load_dataframe, load_array, load_pickle,
    save_array, save_pickle, save_json, ensure_dir,
)
from src.data.preprocessing import standardize_features
from src.data.splits import make_temporal_split
from src.regimes.stress_index import StressIndex
from src.regimes.partition import create_partition
from src.normal_model.vae import VAE, train_vae, encode_data, save_vae
from src.normal_model.prototypes import fit_prototypes, save_prototypes
from src.normal_model.stability import select_M_by_reconstruction

logger = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output-dir", "-o", default=None)
@click.option("--log-level", default="INFO")
def main(config: str, output_dir: str | None, log_level: str) -> None:
    """Train VAE on D_normal and build prototype model."""
    cfg = load_experiment_config(config)
    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "experiment")
    out_dir = Path(output_dir or exp_cfg.get("output_dir", f"outputs/experiments/{exp_name}"))

    setup_logging(log_level, log_dir=out_dir / "logs", experiment_name=exp_name)
    set_seed(exp_cfg.get("seed", 42))

    data_dir = out_dir / "data"
    model_dir = out_dir / "models"
    ensure_dir(model_dir)

    # ------------------------------------------------------------------
    # Load prepared data
    # ------------------------------------------------------------------
    returns = load_dataframe(data_dir / "returns_clean.parquet")
    weights = load_dataframe(data_dir / "portfolio_weights.parquet").squeeze()
    import json
    with open(data_dir / "split_info.json") as f:
        split_info = json.load(f)

    train_returns = returns.iloc[split_info["train_idx"]]

    # ------------------------------------------------------------------
    # Compute stress index and partition on train
    # ------------------------------------------------------------------
    feat_cfg = cfg.get("features", cfg)
    regime_cfg = cfg.get("regimes", {})

    si = StressIndex(cfg)
    si.alpha = regime_cfg.get("alpha", 0.10)
    stress_score, regime_labels = si.fit_transform(train_returns, weights, feat_cfg)

    partition = create_partition(stress_score, regime_labels, si.alpha, si.threshold_)

    # Save partition
    partition.to_dataframe().to_parquet(model_dir / "partition_train.parquet")
    save_json(si.to_dict(), model_dir / "stress_index_state.json")

    # ------------------------------------------------------------------
    # Get D_normal for VAE training
    # ------------------------------------------------------------------
    normal_returns = partition.filter_data(train_returns, "normal")
    logger.info(f"D_normal training size: {len(normal_returns)}")

    # Standardise
    X_normal = normal_returns.to_numpy(dtype=np.float32)
    X_scaled, scaler_params = standardize_features(X_normal)
    save_pickle(scaler_params, model_dir / "normal_scaler.pkl")

    # Inner val split for VAE
    n_inner = len(X_scaled)
    n_val = int(n_inner * 0.15)
    X_train_vae = X_scaled[:-n_val] if n_val > 0 else X_scaled
    X_val_vae = X_scaled[-n_val:] if n_val > 0 else None

    # ------------------------------------------------------------------
    # Train VAE
    # ------------------------------------------------------------------
    vae_cfg = cfg.get("model_vae", cfg.get("vae", {}))
    input_dim = X_scaled.shape[1]
    latent_dim = int(vae_cfg.get("latent_dim", 8))
    hidden_dims = list(vae_cfg.get("hidden_dims", [64, 32]))
    dropout = float(vae_cfg.get("dropout", 0.1))

    vae = VAE(input_dim=input_dim, hidden_dims=hidden_dims,
              latent_dim=latent_dim, dropout=dropout)
    history = train_vae(vae, X_train_vae, vae_cfg, val_data=X_val_vae)
    save_vae(vae, model_dir / "vae.pt")
    save_json(history, model_dir / "vae_training_history.json")

    # ------------------------------------------------------------------
    # Encode and build prototypes
    # ------------------------------------------------------------------
    embeddings = encode_data(vae, X_scaled)
    save_array(embeddings, model_dir / "normal_embeddings.npy")

    M = int(vae_cfg.get("M", 20))
    method = vae_cfg.get("prototype_method", "kmeans")
    seed = int(exp_cfg.get("seed", 42))

    # Optional: assess M selection
    M_values = cfg.get("calibration", {}).get("parameters", {}).get("M", {}).get("search_values", [M])
    if len(M_values) > 1:
        m_selection = select_M_by_reconstruction(embeddings, M_values, seed=seed)
        m_selection.to_parquet(model_dir / "M_selection.parquet")
        logger.info(f"M selection results:\n{m_selection}")

    proto = fit_prototypes(embeddings, M=M, method=method, seed=seed)
    save_prototypes(proto, model_dir / "prototypes.npz")
    save_array(proto.weights, model_dir / "prototype_weights.npy")
    save_array(proto.assignments, model_dir / "prototype_assignments.npy")

    save_json({
        "M": M,
        "inertia": proto.inertia,
        "cluster_sizes": proto.cluster_sizes.tolist(),
        "weights": proto.weights.tolist(),
    }, model_dir / "prototype_summary.json")

    logger.info(f"Normal model training complete. Models saved to {model_dir}")


if __name__ == "__main__":
    main()
