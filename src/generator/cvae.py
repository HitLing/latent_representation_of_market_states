"""Conditional VAE for synthetic stress scenario generation.

Methodology
-----------
The CVAE generates synthetic stress scenarios conditioned on severity bucket.
This is controlled enrichment of rare tail observations under limited
historical stress data.  It is NOT reconstruction of the true crisis
distribution, and it is NOT unconditional generation.

Architecture
------------
Encoder : [x ‖ c] → (μ, log σ²)
Reparam : z = μ + ε · exp(0.5 · log σ²)
Decoder : [z ‖ c] → x̂
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class CVAEEncoder(nn.Module):
    """Encoder: [x ‖ c] → (μ, log σ²)."""

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = input_dim + condition_dim
        dims = [in_dim] + list(hidden_dims)
        blocks = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            blocks.append(_mlp_block(d_in, d_out, dropout))
        self.backbone = nn.Sequential(*blocks)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xc = torch.cat([x, c], dim=1)
        h = self.backbone(xc)
        return self.fc_mu(h), self.fc_logvar(h)


class CVAEDecoder(nn.Module):
    """Decoder: [z ‖ c] → x̂."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = latent_dim + condition_dim
        dims = [in_dim] + list(reversed(hidden_dims))
        blocks = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            blocks.append(_mlp_block(d_in, d_out, dropout))
        self.backbone = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(dims[-1], output_dim)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        zc = torch.cat([z, c], dim=1)
        h = self.backbone(zc)
        return self.output_layer(h)


class CVAE(nn.Module):
    """Full Conditional VAE."""

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.encoder = CVAEEncoder(input_dim, condition_dim, hidden_dims, latent_dim, dropout)
        self.decoder = CVAEDecoder(latent_dim, condition_dim, hidden_dims, input_dim, dropout)

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, μ, log σ²)."""
        mu, log_var = self.encoder(x, c)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z, c)
        return recon, mu, log_var

    def generate(
        self,
        condition: torch.Tensor,
        n_samples: int = 1,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Generate *n_samples* scenarios for a given condition vector.

        Samples z ~ N(0, I), decodes with condition.
        Condition shape: (condition_dim,) or (1, condition_dim).
        Returns tensor shape (n_samples, input_dim).
        """
        self.eval()
        self.to(device)
        with torch.no_grad():
            if condition.dim() == 1:
                condition = condition.unsqueeze(0)
            c_rep = condition.expand(n_samples, -1).to(device)
            z = torch.randn(n_samples, self.latent_dim, device=device)
            return self.decoder(z, c_rep)


def cvae_loss(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """CVAE loss = MSE reconstruction + kl_weight · KL(N(μ,σ²) ‖ N(0,1))."""
    recon_loss = nn.functional.mse_loss(reconstruction, original, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total = recon_loss + kl_weight * kl_loss
    return total, {
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": total.item(),
    }
