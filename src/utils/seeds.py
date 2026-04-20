"""Reproducibility utilities — set all RNG seeds uniformly."""
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global random seed set to {seed}")


def get_rng(seed: int) -> np.random.Generator:
    """Return a seeded NumPy Generator for local use (avoids polluting global state)."""
    return np.random.default_rng(seed)
