"""Loss functions used by the P2I uncertainty trainers."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "gan_loss",
    "pm_loss",
    "compute_forward_difference",
    "temporal_difference_kl_loss",
    "AdversarialLoss",
]


def pm_loss(pred: torch.Tensor, target: torch.Tensor, omega: float = 10.0) -> torch.Tensor:
    """Pixel-wise sorted marginal matching loss."""

    pred = pred.float()
    target = target.float()

    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)

    k = max(1, math.ceil((1.0 - 0.99) * pred_flat.size(1)))

    pred_tail = torch.topk(pred_flat, k=k, dim=1, largest=True, sorted=True).values
    target_tail = torch.topk(target_flat, k=k, dim=1, largest=True, sorted=True).values

    l_pm = torch.mean((pred_tail - target_tail) ** 2)

    return omega * l_pm


def _softmax_temperature(tensor: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature-scaled softmax over flattened spatial dims."""
    size = tensor.size()
    tensor = tensor.view(size[0], size[1], -1)
    softmax = F.softmax(tensor / temperature, dim=-1)
    return softmax.view(size)


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL divergence between temporal difference distributions."""
    p = p.view(p.size(0), p.size(1), -1)
    q = q.view(q.size(0), q.size(1), -1)
    return F.kl_div(
        p.clamp_min(1e-8).log(), q, reduction="batchmean"
    ).clamp_min(0.0)


def compute_forward_difference(series: torch.Tensor) -> torch.Tensor:
    """Forward temporal difference."""
    return series[:, 1:] - series[:, :-1]


def temporal_difference_kl_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """KL matching between predicted and target temporal-change maps.

    Inputs use ``[B,T,...]``. The spatial/channel values of every temporal
    difference are converted to a probability distribution before computing
    ``KL(target || prediction)``.
    """
    prediction_probability = _softmax_temperature(
        compute_forward_difference(prediction), temperature
    )
    target_probability = _softmax_temperature(
        compute_forward_difference(target), temperature
    )
    return _kl_divergence(prediction_probability, target_probability)


class AdversarialLoss(nn.Module):
    """Multi-mode adversarial loss."""

    def __init__(self, loss_type: str = "nsgan", target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.loss_type = loss_type
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

        if loss_type == "nsgan":
            self.criterion = nn.BCELoss()
        elif loss_type == "lsgan":
            self.criterion = nn.MSELoss()
        elif loss_type == "hinge":
            self.criterion = nn.ReLU()
        else:
            raise ValueError(f"Unsupported GAN loss type: {loss_type}")

    def forward(self, outputs: torch.Tensor, is_real: bool, is_disc: bool | None = None) -> torch.Tensor:
        if self.loss_type == "hinge":
            if is_disc is None:
                raise ValueError("`is_disc` must be set when using hinge loss.")

            if is_disc:
                if is_real:
                    return self.criterion(1 - outputs).mean()
                else:
                    return self.criterion(1 + outputs).mean()
            else:
                # generator loss: -D(G(z))
                return (-outputs).mean()

        # NSGAN / LSGAN
        labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
        return self.criterion(outputs, labels)


_GAN_LOSS_CACHE: Dict[Tuple[str, float, float], AdversarialLoss] = {}


def gan_loss(
    logits: torch.Tensor,
    target_is_real: bool,
    *,
    loss_type: str = "nsgan",
    is_disc: bool = False,
    target_real_label: float = 1.0,
    target_fake_label: float = 0.0,
) -> torch.Tensor:
    """GAN loss helper mirroring :class:`AdversarialLoss`."""

    key = (loss_type, target_real_label, target_fake_label)
    loss_fn = _GAN_LOSS_CACHE.get(key)
    if loss_fn is None:
        loss_fn = AdversarialLoss(loss_type, target_real_label, target_fake_label)
        _GAN_LOSS_CACHE[key] = loss_fn

    if loss_fn.real_label.device != logits.device:
        loss_fn = loss_fn.to(logits.device)
        _GAN_LOSS_CACHE[key] = loss_fn  # cache moved module

    return loss_fn(logits, target_is_real, is_disc=is_disc)
