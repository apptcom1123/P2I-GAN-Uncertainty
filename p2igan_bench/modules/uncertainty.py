"""Uncertainty utilities for Bayesian spatio-temporal interpolation."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


def heteroscedastic_gaussian_nll(
    prediction: torch.Tensor,
    target: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gaussian NLL with predicted per-pixel log-variance."""
    variance = torch.exp(logvar).clamp_min(eps)
    loss = 0.5 * (((target - prediction) ** 2) / variance + torch.log(variance))
    if beta > 0.0:
        loss = (variance.detach() ** beta) * loss
    return loss.mean()


def summarize_uncertainty(raw_samples: torch.Tensor, *, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """Split raw Bayesian samples into mean / aleatoric / epistemic summaries.

    raw_samples shape: [S, T, C, H, W, 2]
    """
    prediction_samples = raw_samples[..., 0]
    logvar_samples = raw_samples[..., 1]

    mean = prediction_samples.mean(dim=0)
    epistemic_var = prediction_samples.var(dim=0, unbiased=False)
    aleatoric_var = torch.exp(logvar_samples).mean(dim=0)
    predictive_var = (epistemic_var + aleatoric_var).clamp_min(eps)

    return {
        "mean": mean,
        "aleatoric_var": aleatoric_var.clamp_min(eps),
        "epistemic_var": epistemic_var.clamp_min(0.0),
        "predictive_var": predictive_var,
        "aleatoric_std": torch.sqrt(aleatoric_var.clamp_min(eps)),
        "epistemic_std": torch.sqrt(epistemic_var.clamp_min(0.0)),
        "predictive_std": torch.sqrt(predictive_var),
    }


def kl_weight_equal(num_batches: int, **_: int) -> float:
    return 1.0 / max(1, num_batches)


def kl_weight_epoch(num_batches: int, current_epoch: int, epochs_at_start: int = 0, **_: int) -> float:
    return (1.0 / max(1, num_batches)) * (1.0 / (2.0 ** (current_epoch + epochs_at_start)))


def kl_weight_blundell(num_batches: int, batch_idx: int, **_: int) -> float:
    try:
        return 2 ** (num_batches - batch_idx - 1) / (2 ** num_batches - 1)
    except OverflowError:
        return 0.0


def match_kl_weight(name: str):
    normalized = str(name).lower()
    if normalized == "equal":
        return kl_weight_equal
    if normalized == "epoch":
        return kl_weight_epoch
    if normalized == "blundell":
        return kl_weight_blundell
    raise ValueError(f"Unknown KL weighting scheme: {name}")


def _power_spectrum_2d(field: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    spectrum = torch.fft.rfft2(field)
    return (spectrum.abs() ** 2).clamp_min(eps)


def _standardize_noise(field: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = field - field.mean()
    scale = centered.std(unbiased=False).clamp_min(eps)
    return centered / scale


def generate_correlated_noise(
    reference: torch.Tensor,
    *,
    num_members: int,
    temporal_mode: str = "per_frame",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Generate FFT-shaped correlated noise from interpolation inputs.

    reference shape: [T, C, H, W]
    output shape: [E, T, C, H, W]
    """
    if reference.ndim != 4:
        raise ValueError(f"reference must have shape [T, C, H, W], got {tuple(reference.shape)}")

    t, c, h, w = reference.shape
    ref = reference.float()

    if c > 1:
        ref_scalar = ref.mean(dim=1)
    else:
        ref_scalar = ref[:, 0]

    if temporal_mode == "shared":
        power = _power_spectrum_2d(ref_scalar.mean(dim=0), eps=eps).expand(t, -1, -1)
    else:
        power = torch.stack([_power_spectrum_2d(ref_scalar[i], eps=eps) for i in range(t)], dim=0)

    amplitude = torch.sqrt(power)
    noise = []
    for _ in range(num_members):
        member_frames = []
        for frame_idx in range(t):
            spectral_shape = amplitude[frame_idx].shape
            real = torch.randn(spectral_shape, device=ref.device, dtype=ref.dtype)
            imag = torch.randn(spectral_shape, device=ref.device, dtype=ref.dtype)
            spectrum = torch.complex(real, imag) * amplitude[frame_idx]
            field = torch.fft.irfft2(spectrum, s=(h, w)).real
            member_frames.append(_standardize_noise(field, eps=eps))
        member = torch.stack(member_frames, dim=0).unsqueeze(1).repeat(1, c, 1, 1)
        noise.append(member)
    return torch.stack(noise, dim=0)


def generate_ensemble_from_uncertainty(
    mean: torch.Tensor,
    predictive_var: torch.Tensor,
    reference: torch.Tensor,
    *,
    num_members: int,
    temporal_mode: str = "per_frame",
    clip_min: float | None = 0.0,
    clip_max: float | None = 1.0,
) -> torch.Tensor:
    """Generate correlated ensemble members from predictive mean/variance."""
    correlated_noise = generate_correlated_noise(
        reference,
        num_members=num_members,
        temporal_mode=temporal_mode,
    )
    ensemble = mean.unsqueeze(0) + correlated_noise * torch.sqrt(predictive_var.clamp_min(1e-6)).unsqueeze(0)
    if clip_min is not None or clip_max is not None:
        min_value = -math.inf if clip_min is None else clip_min
        max_value = math.inf if clip_max is None else clip_max
        ensemble = ensemble.clamp(min=min_value, max=max_value)
    return ensemble


def enable_mc_dropout(module: nn.Module) -> None:
    """Keep dropout stochastic during inference-time Monte Carlo sampling."""
    for child in module.modules():
        if isinstance(child, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            child.train()
