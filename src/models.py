"""The two independent GRUs used by the opening-repricing experiment.

Inputs are already causal, clipped and accompanied by missing-value masks.
The two networks do not share parameters.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FiveDayRankingGRU(nn.Module):
    """Ranks the t+1-open to t+6-open return across stocks."""

    def __init__(self, input_size: int = 20, hidden_size: int = 32) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.rnn(x)
        return self.head(sequence[:, -1]).squeeze(-1)


class OvernightGapGRU(nn.Module):
    """Predicts the next-open arithmetic return and its conditional variance."""

    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 32,
        variance_floor: float = 0.01,
    ) -> None:
        super().__init__()
        self.variance_floor = variance_floor
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, _ = self.rnn(x)
        raw = self.head(sequence[:, -1])
        mean = raw[:, 0]
        variance = F.softplus(raw[:, 1]) + self.variance_floor
        return mean, variance


def rank_correlation_loss(score: torch.Tensor, target_rank: torch.Tensor) -> torch.Tensor:
    """Original five-day ranking loss; target ranks are scaled to [-1, 1]."""

    centered_score = score - score.mean()
    centered_target = target_rank - target_rank.mean()
    denominator = (
        centered_score.square().mean().sqrt()
        * centered_target.square().mean().sqrt()
    ).clamp_min(1e-6)
    correlation = (centered_score * centered_target).mean() / denominator
    return 1.0 - correlation + 0.01 * centered_score.square().mean()


def gaussian_nll(
    mean: torch.Tensor,
    variance: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL used for the overnight return in percentage-point units."""

    return 0.5 * (variance.log() + (target - mean).square() / variance).mean()


def build_models() -> tuple[FiveDayRankingGRU, OvernightGapGRU]:
    """Build two independent models with the frozen research architecture."""

    return FiveDayRankingGRU(), OvernightGapGRU()


