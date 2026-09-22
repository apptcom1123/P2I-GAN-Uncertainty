"""Uncertainty utilities for Bayesian spatio-temporal interpolation."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def member_gaussian_nll(
    member_mean: torch.Tensor,
    target: torch.Tensor,
    member_logvar: torch.Tensor,
    *,
    beta: float = 0.0,
) -> torch.Tensor:
    """Mean Gaussian NLL over explicit model members ``[S,B,...]``."""
    member_mean = member_mean.float()
    member_logvar = member_logvar.float()
    target = target.float().unsqueeze(0)
    variance = member_logvar.exp().clamp_min(1e-6)
    loss = 0.5 * ((target - member_mean).square() / variance + member_logvar)
    if beta > 0.0:
        loss = variance.detach().pow(beta) * loss
    return loss.mean()


def gaussian_mixture_nll(
    member_mean: torch.Tensor,
    target: torch.Tensor,
    member_logvar: torch.Tensor,
) -> torch.Tensor:
    """Proper NLL for the equally weighted Gaussian mixture produced by M4."""
    member_mean = member_mean.float()
    member_logvar = member_logvar.float()
    target = target.float().unsqueeze(0)
    log_prob = -0.5 * (
        (target - member_mean).square() * torch.exp(-member_logvar)
        + member_logvar
        + math.log(2.0 * math.pi)
    )
    return -(torch.logsumexp(log_prob, dim=0) - math.log(member_mean.shape[0])).mean()


def ensemble_crps_loss(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable empirical CRPS for ensemble samples ``[S,B,...]``."""
    samples, target = samples.float(), target.float()
    accuracy = (samples - target.unsqueeze(0)).abs().mean()
    pairwise = (samples[:, None] - samples[None, :]).abs().mean()
    return accuracy - 0.5 * pairwise


def multiscale_energy_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    pool_sizes: tuple[int, ...] = (1, 4, 8),
) -> torch.Tensor:
    """Multivariate proper score preserving spatial dependence at several scales."""
    samples, target = samples.float(), target.float()
    s, b, t, c, h, w = samples.shape
    total = samples.new_zeros(())
    valid_scales = 0
    for pool in pool_sizes:
        if pool > min(h, w):
            continue
        if pool > 1:
            xs = F.avg_pool2d(samples.reshape(s * b * t, c, h, w), pool, pool)
            ys = F.avg_pool2d(target.reshape(b * t, c, h, w), pool, pool)
            xs = xs.reshape(s, b, -1)
            ys = ys.reshape(b, -1)
        else:
            xs, ys = samples.reshape(s, b, -1), target.reshape(b, -1)
        norm = math.sqrt(xs.shape[-1])
        accuracy = torch.linalg.vector_norm(xs - ys.unsqueeze(0), dim=-1).mean() / norm
        spread = torch.linalg.vector_norm(xs[:, None] - xs[None, :], dim=-1).mean() / norm
        total = total + accuracy - 0.5 * spread
        valid_scales += 1
    return total / max(1, valid_scales)


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
