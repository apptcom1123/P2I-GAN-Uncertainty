"""Uncertainty utilities for Bayesian spatio-temporal interpolation."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def member_gaussian_nll(
    member_mean: torch.Tensor,
    target: torch.Tensor,
    member_logvar: torch.Tensor,
    *,
    beta: float = 0.0,
) -> torch.Tensor:
    r"""Mean Gaussian NLL over explicit model members ``[S,B,...]``.

    Let :math:`s\in\{1,\ldots,S\}` index members and :math:`i` index all
    batch/spatio-temporal elements. With member mean :math:`\mu_{s,i}` and
    predicted log-variance :math:`\ell_{s,i}`,

    .. math::

        \tilde\sigma_{s,i}^2
        = \max\!\left(\exp(\ell_{s,i}), 10^{-6}\right),

    .. math::

        L_{s,i} = \frac{1}{2}\left[
            \frac{(y_i-\mu_{s,i})^2}{\tilde\sigma_{s,i}^2}
            + \ell_{s,i}
        \right].

    The returned scalar is

    .. math::

        \mathcal L = \frac{1}{SN}\sum_{s=1}^{S}\sum_{i=1}^{N}
        \begin{cases}
        \operatorname{stopgrad}\!\left[(\tilde\sigma_{s,i}^2)^\beta\right]
            L_{s,i}, & \beta>0,\\
        L_{s,i}, & \beta=0.
        \end{cases}

    The constant :math:`\tfrac12\log(2\pi)` is omitted.
    """
    member_mean = member_mean.float()
    member_logvar = member_logvar.float()
    target = target.float().unsqueeze(0)
    variance = member_logvar.exp().clamp_min(1e-6)
    loss = 0.5 * ((target - member_mean).square() / variance + member_logvar)
    if beta > 0.0:
        loss = variance.detach().pow(beta) * loss
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


def enable_mc_dropout(module: nn.Module) -> None:
    """Keep dropout stochastic during inference-time Monte Carlo sampling."""
    for child in module.modules():
        if isinstance(child, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            child.train()
