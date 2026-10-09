"""Minimal mean-field Bayesian layers used by the P2I BNN decoder."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("posterior_init_scale must be positive")
    return math.log(math.expm1(value))


class BayesianConv2d(nn.Module):
    """Conv2d with a factorized Gaussian posterior and zero-mean Gaussian prior.

    Outside :func:`sampled_bayesian_weights`, the posterior mean is used. Inside
    that context, one reparameterized weight sample is cached for the complete
    forward pass so every use of a layer represents the same sampled network.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        prior_scale: float = 0.1,
        posterior_init_scale: float = 0.005,
    ) -> None:
        super().__init__()
        if prior_scale <= 0.0:
            raise ValueError("prior_scale must be positive")
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(kernel_size), int(kernel_size))
        self.stride = (int(stride), int(stride))
        self.padding = (int(padding), int(padding))
        self.groups = int(groups)

        weight_shape = (
            self.out_channels,
            self.in_channels // self.groups,
            *self.kernel_size,
        )
        self.weight_mu = nn.Parameter(torch.empty(weight_shape))
        initial_rho = _inverse_softplus(float(posterior_init_scale))
        self.weight_rho = nn.Parameter(torch.full(weight_shape, initial_rho))
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(self.out_channels))
            self.bias_rho = nn.Parameter(torch.full((self.out_channels,), initial_rho))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)
        self.register_buffer("prior_scale", torch.tensor(float(prior_scale)))
        self._sampled_weight = None
        self._sampled_bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight_mu, a=0.0, mode="fan_in", nonlinearity="relu")
        if self.bias_mu is not None:
            fan_in = (self.in_channels // self.groups) * math.prod(self.kernel_size)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)

    def set_posterior_mean(self, weight: torch.Tensor,
                           bias: torch.Tensor | None = None) -> None:
        """Initialize the posterior mean from an equivalent deterministic conv."""
        if weight.shape != self.weight_mu.shape:
            raise ValueError(
                f"weight shape mismatch: expected {tuple(self.weight_mu.shape)}, "
                f"got {tuple(weight.shape)}"
            )
        with torch.no_grad():
            self.weight_mu.copy_(weight)
            if self.bias_mu is not None:
                if bias is None or bias.shape != self.bias_mu.shape:
                    raise ValueError("a matching deterministic bias is required")
                self.bias_mu.copy_(bias)

    @staticmethod
    def _posterior_scale(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho).clamp_min(1e-8)

    def sample_parameters(self, generator: torch.Generator | None = None) -> None:
        weight_scale = self._posterior_scale(self.weight_rho)
        weight_noise = torch.randn(
            self.weight_mu.shape,
            device=self.weight_mu.device,
            dtype=self.weight_mu.dtype,
            generator=generator,
        )
        self._sampled_weight = self.weight_mu + weight_scale * weight_noise
        if self.bias_mu is not None:
            bias_scale = self._posterior_scale(self.bias_rho)
            bias_noise = torch.randn(
                self.bias_mu.shape,
                device=self.bias_mu.device,
                dtype=self.bias_mu.dtype,
                generator=generator,
            )
            self._sampled_bias = self.bias_mu + bias_scale * bias_noise

    def clear_sample(self) -> None:
        self._sampled_weight = None
        self._sampled_bias = None

    def kl_divergence(self) -> torch.Tensor:
        prior_scale = self.prior_scale.to(dtype=self.weight_mu.dtype)

        def normal_kl(mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            return 0.5 * (
                (scale.square() + mean.square()) / prior_scale.square()
                - 1.0
                + 2.0 * (prior_scale.log() - scale.log())
            ).sum()

        kl = normal_kl(self.weight_mu, self._posterior_scale(self.weight_rho))
        if self.bias_mu is not None:
            kl = kl + normal_kl(self.bias_mu, self._posterior_scale(self.bias_rho))
        return kl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_mu if self._sampled_weight is None else self._sampled_weight
        bias = self.bias_mu if self._sampled_bias is None else self._sampled_bias
        return F.conv2d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )


@contextmanager
def sampled_bayesian_weights(
    *modules: nn.Module,
    generator: torch.Generator | None = None,
) -> Iterator[None]:
    """Sample and cache every BayesianConv2d under ``modules`` for one pass."""
    layers = [
        layer
        for module in modules
        for layer in module.modules()
        if isinstance(layer, BayesianConv2d)
    ]
    for layer in layers:
        layer.sample_parameters(generator=generator)
    try:
        yield
    finally:
        for layer in layers:
            layer.clear_sample()


def bayesian_kl_divergence(module: nn.Module) -> torch.Tensor:
    """Sum analytic ``KL(q(weight)||p(weight))`` over Bayesian convolutions."""
    layers = [layer for layer in module.modules() if isinstance(layer, BayesianConv2d)]
    if not layers:
        return next(module.parameters()).new_zeros(())
    return torch.stack([layer.kl_divergence() for layer in layers]).sum()


__all__ = ["BayesianConv2d", "bayesian_kl_divergence", "sampled_bayesian_weights"]
