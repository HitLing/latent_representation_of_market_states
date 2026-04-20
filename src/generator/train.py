"""CVAE training loop."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.generator.cvae import CVAE, cvae_loss
from src.tail.severity import encode_severity_batch

logger = logging.getLogger(__name__)


def prepare_stress_training_data(
    stress_returns: pd.DataFrame,
    severity_labels: pd.Series,
    n_severity_buckets: int,
    scaler_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Prepare stress data for CVAE training.

    Steps
    -----
    1. Align stress_returns and severity_labels by date.
    2. Standardise returns (fit on training data, or apply existing scaler).
    3. One-hot encode severity labels.

    Returns
    -------
    X_scaled : np.ndarray, shape (N, n_assets)
    C_onehot : np.ndarray, shape (N, n_severity_buckets)
    scaler_params : dict with 'mean' and 'std'
    """
    # Align
    common = stress_returns.index.intersection(severity_labels.index)
    X = stress_returns.loc[common].to_numpy(dtype=np.float32)
    sev = severity_labels.loc[common].to_numpy(dtype=int)

    # Standardise
    if scaler_params is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-10, 1.0, std)
        scaler_params = {"mean": mean, "std": std}

    X_scaled = (X - scaler_params["mean"]) / scaler_params["std"]

    # One-hot condition
    C = encode_severity_batch(sev, n_severity_buckets)

    logger.info(
        f"Stress training data: N={len(X_scaled)}, "
        f"input_dim={X_scaled.shape[1]}, condition_dim={C.shape[1]}"
    )
    return X_scaled.astype(np.float32), C, scaler_params


def train_cvae(
    model: CVAE,
    X_train: np.ndarray,
    C_train: np.ndarray,
    cfg: dict,
    device: str = "cpu",
    X_val: np.ndarray | None = None,
    C_val: np.ndarray | None = None,
    checkpoint_dir: Path | None = None,
) -> dict:
    """Train CVAE on stress scenarios with severity conditioning.

    Parameters
    ----------
    X_train : shape (N, input_dim) — standardised stress returns
    C_train : shape (N, condition_dim) — one-hot severity conditions
    cfg     : model config dict
    """
    batch_size: int = cfg.get("batch_size", 32)
    lr: float = cfg.get("learning_rate", 1e-3)
    n_epochs: int = cfg.get("n_epochs", 150)
    kl_weight: float = cfg.get("kl_weight", 1.0)
    patience: int = cfg.get("early_stopping_patience", 20)

    model = model.to(device)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    Ct = torch.tensor(C_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, Ct), batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and C_val is not None:
        Xv = torch.tensor(X_val, dtype=torch.float32)
        Cv = torch.tensor(C_val, dtype=torch.float32)
        val_loader = DataLoader(TensorDataset(Xv, Cv), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "best_epoch": [0]}

    best_val_loss = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_total = 0.0
        for xb, cb in loader:
            xb, cb = xb.to(device), cb.to(device)
            optimizer.zero_grad()
            recon, mu, log_var = model(xb, cb)
            loss, _ = cvae_loss(recon, xb, mu, log_var, kl_weight)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(xb)
        train_loss = train_total / len(Xt)
        history["train_loss"].append(train_loss)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for xb, cb in val_loader:
                    xb, cb = xb.to(device), cb.to(device)
                    recon, mu, log_var = model(xb, cb)
                    loss, _ = cvae_loss(recon, xb, mu, log_var, kl_weight)
                    val_total += loss.item() * len(xb)
            val_loss = val_total / len(Xv)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                history["best_epoch"][0] = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"CVAE early stopping at epoch {epoch}")
                    break

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"CVAE epoch {epoch:4d}/{n_epochs}: "
                f"train={train_loss:.5f}"
                + (f", val={val_loss:.5f}" if not np.isnan(val_loss) else "")
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best CVAE weights (epoch {history['best_epoch'][0]})")

    # Save checkpoint
    if checkpoint_dir is not None:
        save_cvae_checkpoint(model, history, {}, checkpoint_dir / "cvae_best.pt")

    return history


def save_cvae_checkpoint(
    model: CVAE,
    history: dict,
    scaler_params: dict,
    path: Path | str,
) -> None:
    """Save CVAE model, training history, and data scaler."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "condition_dim": model.condition_dim,
            "latent_dim": model.latent_dim,
            "history": history,
            "scaler_params": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in scaler_params.items()
            },
        },
        path,
    )
    logger.info(f"Saved CVAE checkpoint -> {path}")


def load_cvae_checkpoint(
    path: Path | str, cfg: dict
) -> tuple[CVAE, dict, dict]:
    """Load CVAE model, history, and scaler from checkpoint.

    Returns (model, history, scaler_params).
    """
    ckpt = torch.load(Path(path), map_location="cpu")
    hidden_dims: list[int] = cfg.get("hidden_dims", [64, 64])
    dropout: float = cfg.get("dropout", 0.1)
    model = CVAE(
        input_dim=ckpt["input_dim"],
        condition_dim=ckpt["condition_dim"],
        hidden_dims=hidden_dims,
        latent_dim=ckpt["latent_dim"],
        dropout=dropout,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scaler = ckpt.get("scaler_params", {})
    for k, v in scaler.items():
        if isinstance(v, list):
            scaler[k] = np.array(v, dtype=np.float32)

    logger.info(f"Loaded CVAE checkpoint from {path}")
    return model, ckpt.get("history", {}), scaler
