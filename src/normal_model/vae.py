"""VAE for normal-state latent representation.

Purpose
-------
Learn a compact latent representation of normal market-state windows.
The latent space is used for prototype construction (clustering),
NOT for generating new normal scenarios — that distinction is critical.

Architecture
------------
Encoder : MLP → (μ, log σ²)
Reparam : z = μ + ε · exp(0.5 · log σ²),  ε ~ N(0,I)
Decoder : MLP → reconstruction
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class VAEEncoder(nn.Module):
    """MLP encoder: input_dim → hidden_dims → (μ, log σ²)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        blocks = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            blocks.append(_mlp_block(in_d, out_d, dropout))
        self.backbone = nn.Sequential(*blocks)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    """MLP decoder: latent_dim → hidden_dims (reversed) → input_dim."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = [latent_dim] + list(reversed(hidden_dims))
        blocks = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            blocks.append(_mlp_block(in_d, out_d, dropout))
        self.backbone = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(dims[-1], output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.backbone(z)
        return self.output_layer(h)


class VAE(nn.Module):
    """Full VAE: encoder + reparameterisation + decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = VAEDecoder(latent_dim, hidden_dims, input_dim, dropout)

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """z = μ + ε · exp(0.5 · log σ²),  ε ~ N(0,I)."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at eval time

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, μ, log σ²)."""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def get_latent(
        self, x: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        """Return latent representation (μ if deterministic, sample otherwise)."""
        mu, log_var = self.encoder(x)
        if deterministic:
            return mu
        return self.reparameterize(mu, log_var)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def vae_loss(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """VAE loss = MSE reconstruction + kl_weight · KL divergence.

    KL = −0.5 · Σ(1 + log σ² − μ² − σ²)
    """
    recon_loss = nn.functional.mse_loss(reconstruction, original, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total = recon_loss + kl_weight * kl_loss
    return total, {
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": total.item(),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_vae(
    model: VAE,
    data: np.ndarray,
    cfg: dict,
    device: str = "cpu",
    val_data: np.ndarray | None = None,
) -> dict:
    """Train VAE on normal-state data.

    Parameters
    ----------
    data : np.ndarray, shape (N, input_dim), pre-standardised
    cfg  : model config dict (batch_size, learning_rate, n_epochs, …)
    """
    batch_size: int = cfg.get("batch_size", 64)
    lr: float = cfg.get("learning_rate", 1e-3)
    n_epochs: int = cfg.get("n_epochs", 100)
    kl_weight: float = cfg.get("kl_weight", 1.0)
    patience: int = cfg.get("early_stopping_patience", 15)

    model = model.to(device)

    X = torch.tensor(data, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_data is not None:
        Xv = torch.tensor(val_data, dtype=torch.float32)
        val_loader = DataLoader(TensorDataset(Xv), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list] = {"train_loss": [], "val_loss": [], "best_epoch": [0]}
    best_val_loss = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        # --- train ---
        model.train()
        train_total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, log_var = model(batch)
            loss, _ = vae_loss(recon, batch, mu, log_var, kl_weight)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(batch)
        train_loss = train_total / len(X)
        history["train_loss"].append(train_loss)

        # --- validate ---
        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    recon, mu, log_var = model(batch)
                    loss, _ = vae_loss(recon, batch, mu, log_var, kl_weight)
                    val_total += loss.item() * len(batch)
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
                    logger.info(f"VAE early stopping at epoch {epoch}")
                    break

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"VAE epoch {epoch:4d}/{n_epochs}: "
                f"train={train_loss:.5f}"
                + (f", val={val_loss:.5f}" if not np.isnan(val_loss) else "")
            )

    # Restore best model if validation was used
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best VAE weights (epoch {history['best_epoch'][0]})")

    return history


def encode_data(
    model: VAE,
    data: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Encode data to latent representations (deterministic: returns μ)."""
    model.eval()
    model = model.to(device)
    X = torch.tensor(data, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    latents = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            latents.append(mu.cpu().numpy())
    return np.concatenate(latents, axis=0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_vae(model: VAE, path: Path | str) -> None:
    """Save VAE state dict and architecture params."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
        },
        path,
    )
    logger.info(f"Saved VAE -> {path}")


def load_vae(path: Path | str, cfg: dict) -> VAE:
    """Load VAE from saved checkpoint."""
    checkpoint = torch.load(Path(path), map_location="cpu")
    input_dim = checkpoint["input_dim"]
    latent_dim = checkpoint["latent_dim"]
    hidden_dims: list[int] = cfg.get("hidden_dims", [64, 32])
    dropout: float = cfg.get("dropout", 0.1)
    model = VAE(input_dim, hidden_dims, latent_dim, dropout)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    logger.info(f"Loaded VAE from {path}")
    return model
