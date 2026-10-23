"""Minimal training and calibration helpers for the two GRUs.

The historical experiment trained each test year on the previous three calendar
years, selected the epoch count on the final six months inside that window, then
reinitialized and refit on the full three years. Test-year labels were never used
for fitting, epoch selection or calibration.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

try:
    from .models import gaussian_nll, rank_correlation_loss
except ImportError:  # Allows direct execution with src on PYTHONPATH.
    from models import gaussian_nll, rank_correlation_loss


def seed_everything(seed: int = 17) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ranking_epoch(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    gradient_clip: float = 3.0,
) -> float:
    model.train()
    losses: list[float] = []
    for features, target_rank in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = rank_correlation_loss(model(features), target_rank)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def train_overnight_epoch(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    gradient_clip: float = 3.0,
) -> float:
    model.train()
    losses: list[float] = []
    for features, target_percent in batches:
        optimizer.zero_grad(set_to_none=True)
        mean, variance = model(features)
        loss = gaussian_nll(mean, variance, target_percent)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def calibrate_overnight_distribution(
    predicted_mean: np.ndarray,
    predicted_variance: np.ndarray,
    realized_return: np.ndarray,
) -> dict[str, float]:
    """Fit the frozen affine mean and variance scaling on inner validation only."""

    mean = np.asarray(predicted_mean, dtype=float)
    variance = np.asarray(predicted_variance, dtype=float)
    target = np.asarray(realized_return, dtype=float)
    slope = float(
        np.clip(
            np.mean((mean - mean.mean()) * (target - target.mean()))
            / max(float(np.var(mean)), 1e-12),
            0.0,
            2.0,
        )
    )
    intercept = float(target.mean() - slope * mean.mean())
    variance_scale = float(
        np.clip(np.mean((target - intercept - slope * mean) ** 2 / variance), 0.25, 4.0)
    )
    return {
        "intercept_percent": intercept,
        "slope": slope,
        "variance_scale": variance_scale,
    }


