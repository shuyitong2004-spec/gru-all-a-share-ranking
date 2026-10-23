"""Opening-surprise calibration, risk-return scoring and capped allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class Calibration:
    mean_return: np.ndarray
    mean_discount: np.ndarray
    beta: np.ndarray
    residual_variance: np.ndarray
    cell_variance: np.ndarray


def rank_deciles(score: np.ndarray) -> np.ndarray:
    percentile = rankdata(np.asarray(score, dtype=float)) / len(score)
    return np.minimum((percentile * 10).astype(int), 9)


def opening_discount(
    predicted_gap: np.ndarray,
    actual_gap: np.ndarray,
    clip: float = 0.05,
) -> np.ndarray:
    """Positive means the observed open is below the model-implied open."""

    discount = (1.0 + np.asarray(predicted_gap)) / (1.0 + np.asarray(actual_gap)) - 1.0
    return np.clip(discount, -clip, clip)


def surprise_bins(
    predicted_gap: np.ndarray,
    predicted_variance: np.ndarray,
    actual_gap: np.ndarray,
    boundaries: tuple[float, float, float] = (0.5, 1.0, 2.0),
) -> np.ndarray:
    sigma = np.maximum(np.sqrt(np.asarray(predicted_variance)), 0.002)
    standardized = (np.asarray(predicted_gap) - np.asarray(actual_gap)) / sigma
    return np.digitize(np.abs(standardized), boundaries)


def fit_calibration(
    decile: np.ndarray,
    discount: np.ndarray,
    realized_five_day_return: np.ndarray,
    risk_bin: np.ndarray,
    slope_prior: float = 1000.0,
    variance_prior: float = 1000.0,
    variance_floor: float = 1e-6,
) -> Calibration:
    """Fit only on fully matured observations from the prior three years."""

    b = np.asarray(decile, dtype=int)
    x = np.asarray(discount, dtype=float)
    y = np.asarray(realized_five_day_return, dtype=float)
    v = np.asarray(risk_bin, dtype=int)
    valid = (b >= 0) & (b < 10) & np.isfinite(x) & np.isfinite(y)
    b, x, y, v = b[valid], x[valid], y[valid], v[valid]

    count = np.bincount(b, minlength=10).astype(float)
    if np.any(count == 0):
        raise ValueError("Every rank decile needs at least one calibration observation")
    mean_return = np.bincount(b, weights=y, minlength=10) / count
    mean_discount = np.bincount(b, weights=x, minlength=10) / count
    centered_x = x - mean_discount[b]
    centered_y = y - mean_return[b]
    sum_xx = np.bincount(b, weights=centered_x**2, minlength=10)
    sum_xy = np.bincount(b, weights=centered_x * centered_y, minlength=10)
    prior_scale = max(float(np.mean(centered_x**2)), 1e-10)
    beta = sum_xy / (sum_xx + slope_prior * prior_scale)
    residual = centered_y - beta[b] * centered_x

    base_variance = np.maximum(
        np.bincount(b, weights=residual**2, minlength=10) / count,
        variance_floor,
    )
    cell = b * 4 + v
    cell_count = np.bincount(cell, minlength=40)
    cell_ss = np.bincount(cell, weights=residual**2, minlength=40)
    cell_variance = (cell_ss + variance_prior * np.repeat(base_variance, 4)) / (
        cell_count + variance_prior
    )
    return Calibration(
        mean_return=mean_return,
        mean_discount=mean_discount,
        beta=beta,
        residual_variance=base_variance,
        cell_variance=np.maximum(cell_variance.reshape(10, 4), variance_floor),
    )


def risk_return_score(
    decile: np.ndarray,
    discount: np.ndarray,
    risk_bin: np.ndarray,
    calibration: Calibration,
    round_trip_cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = np.asarray(decile, dtype=int)
    v = np.asarray(risk_bin, dtype=int)
    expected_return = calibration.mean_return[b] + calibration.beta[b] * (
        np.asarray(discount) - calibration.mean_discount[b]
    )
    variance = calibration.cell_variance[b, v]
    score = (expected_return - round_trip_cost) / variance
    return score, expected_return, variance


def capped_proportional_weights(values: np.ndarray, cap: float = 0.01) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    values[~np.isfinite(values)] = 0.0
    weights = np.zeros_like(values)
    active = values > 0
    remaining = min(1.0, cap * int(active.sum()))
    while active.any() and remaining > 1e-14:
        ids = np.flatnonzero(active)
        proposal = remaining * values[ids] / values[ids].sum()
        over = proposal > cap
        if not over.any():
            weights[ids] = proposal
            break
        fixed = ids[over]
        weights[fixed] = cap
        remaining -= cap * len(fixed)
        active[fixed] = False
    return weights


def select_and_allocate(
    original_rank_score: np.ndarray,
    repricing_score: np.ndarray,
    selection_fraction: float = 0.10,
    cap: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Select positive repricing scores, using original rank as a deterministic tie break."""

    rank_score = np.asarray(original_rank_score, dtype=float)
    score = np.asarray(repricing_score, dtype=float)
    k = int(np.ceil(len(score) * selection_fraction))
    order = np.lexsort((np.arange(len(score)), -rank_score, -score))[:k]
    selected = order[score[order] > 0]
    return selected, capped_proportional_weights(score[selected], cap)


