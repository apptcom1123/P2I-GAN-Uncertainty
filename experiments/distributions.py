"""Analytic utilities for equally weighted per-pixel Gaussian mixtures."""

from __future__ import annotations

from typing import Iterable, Mapping, Tuple

import numpy as np
from scipy.special import logsumexp, ndtr


EPS = 1e-6


def mixture_components(event: Mapping[str, np.ndarray], scale: float = 1.0):
    """Return encoded-space means/stds [S,T,H,W], optionally variance-scaled."""
    means = np.asarray(event["member_mean"], dtype=np.float64)
    logvars = np.asarray(event["member_logvar"], dtype=np.float64)
    stds = np.exp(0.5 * logvars) * 255.0
    if scale != 1.0:
        center = np.asarray(event["mean"], dtype=np.float64)
        means = center[None] + float(scale) * (means - center[None])
        stds = float(scale) * stds
    return means, np.maximum(stds, EPS)


def selected_components(event, selector: np.ndarray, scale: float = 1.0):
    means, stds = mixture_components(event, scale)
    truth = np.asarray(event["target"], dtype=np.float64)[selector]
    means, stds = means[:, selector], stds[:, selector]
    valid = np.isfinite(truth) & np.all(np.isfinite(means), axis=0) & np.all(np.isfinite(stds), axis=0)
    return truth[valid], means[:, valid], stds[:, valid]


def deterministic_subsample(truth, means, stds, maximum: int, seed: int):
    if maximum > 0 and truth.size > maximum:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(truth.size, size=maximum, replace=False)
        return truth[chosen], means[:, chosen], stds[:, chosen]
    return truth, means, stds


def mixture_pit(truth: np.ndarray, means: np.ndarray, stds: np.ndarray,
                chunk_size: int = 65536) -> np.ndarray:
    output = np.empty(truth.size, dtype=np.float64)
    for start in range(0, truth.size, chunk_size):
        stop = min(start + chunk_size, truth.size)
        z = (truth[None, start:stop] - means[:, start:stop]) / stds[:, start:stop]
        output[start:stop] = ndtr(z).mean(axis=0)
    return np.clip(output, 0.0, 1.0)


def mixture_nll(truth: np.ndarray, means: np.ndarray, stds: np.ndarray,
                chunk_size: int = 65536) -> float:
    total, count = 0.0, 0
    log_members = np.log(means.shape[0])
    log_two_pi = np.log(2.0 * np.pi)
    for start in range(0, truth.size, chunk_size):
        stop = min(start + chunk_size, truth.size)
        sigma = stds[:, start:stop]
        z = (truth[None, start:stop] - means[:, start:stop]) / sigma
        log_pdf = -0.5 * (z * z + log_two_pi) - np.log(sigma)
        values = -(logsumexp(log_pdf, axis=0) - log_members)
        total += float(values.sum())
        count += values.size
    return total / max(count, 1)


def _normal_absolute_expectation(delta: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = delta / np.maximum(sigma, EPS)
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return 2.0 * sigma * phi + delta * (2.0 * ndtr(z) - 1.0)


def mixture_crps(truth: np.ndarray, means: np.ndarray, stds: np.ndarray) -> float:
    """Exact univariate Gaussian-mixture CRPS, averaged over selected pixels."""
    first = _normal_absolute_expectation(means - truth[None], stds).mean(axis=0)
    delta = means[:, None, :] - means[None, :, :]
    pair_sigma = np.sqrt(stds[:, None, :] ** 2 + stds[None, :, :] ** 2)
    second = 0.5 * _normal_absolute_expectation(delta, pair_sigma).mean(axis=(0, 1))
    return float(np.mean(first - second))


def mixture_ppf(probability: float, means: np.ndarray, stds: np.ndarray,
                iterations: int = 28) -> np.ndarray:
    low = np.min(means - 9.0 * stds, axis=0)
    high = np.max(means + 9.0 * stds, axis=0)
    for _ in range(iterations):
        midpoint = 0.5 * (low + high)
        cdf = ndtr((midpoint[None] - means) / stds).mean(axis=0)
        below = cdf < probability
        low = np.where(below, midpoint, low)
        high = np.where(below, high, midpoint)
    return 0.5 * (low + high)


def mixture_intervals(truth: np.ndarray, means: np.ndarray, stds: np.ndarray,
                      levels: Iterable[float], convert) -> dict:
    result = {}
    for level in levels:
        tail = 1.0 - float(level)
        lower = mixture_ppf(tail / 2.0, means, stds)
        upper = mixture_ppf(1.0 - tail / 2.0, means, stds)
        covered = (truth >= lower) & (truth <= upper)
        y_value, low_value, high_value = convert(truth), convert(lower), convert(upper)
        width = high_value - low_value
        winkler = width.copy()
        winkler += (2.0 / tail) * (low_value - y_value) * (y_value < low_value)
        winkler += (2.0 / tail) * (y_value - high_value) * (y_value > high_value)
        result[f"{level:.2f}"] = {
            "coverage": float(np.mean(covered)),
            "width": float(np.mean(width)),
            "winkler": float(np.mean(winkler)),
        }
    return result
