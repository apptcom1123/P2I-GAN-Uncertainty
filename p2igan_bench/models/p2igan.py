"""Spatio-temporal discriminator shared by the uncertainty models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from p2igan_bench.modules.layer import BaseNetwork, C2, C3


class P2IDiscriminator(BaseNetwork):
    """Fuse frame-stacked 2-D and temporal 3-D discriminator features."""

    def __init__(
        self,
        in_channels: int = 16,
        feature_channels: int = 1,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_channels = feature_channels

        self.d2d = nn.Sequential(
            C2(in_channels, 64, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            C2(64, 128, k=3, s=2, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            C2(128, 256, k=3, s=2, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            C2(256, 256, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            C2(256, 1, k=3, s=1, p=1),
        )
        self.d3d = nn.Sequential(
            C3(
                feature_channels,
                32,
                kt=3,
                ks=3,
                st=(1, 2, 2),
                pt=(1, 1, 1),
            ),
            nn.LeakyReLU(0.2, inplace=True),
            C3(32, 64, kt=3, ks=3, st=(1, 2, 2), pt=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            C3(64, 128, kt=3, ks=3, st=(1, 2, 2), pt=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            C3(128, 128, kt=3, ks=3, st=(2, 1, 1), pt=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv3d(128, 1, kernel_size=1)),
        )

        # A bounded learned mixture avoids one branch dominating purely by scale.
        self.alpha2d = nn.Parameter(torch.tensor(0.0))
        if init_weights:
            self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=0.2,
                    nonlinearity="leaky_relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "P2IDiscriminator expects [batch, time, channels, height, width], "
                f"but received shape {tuple(x.shape)}"
            )

        batch, time, channels, height, width = x.shape
        if channels != self.feature_channels:
            raise ValueError(
                f"Expected {self.feature_channels} feature channels, got {channels}"
            )
        if time * channels != self.in_channels:
            raise ValueError(
                f"Expected time*channels={self.in_channels}, got {time * channels}"
            )

        out2d = self.d2d(x.reshape(batch, time * channels, height, width))
        out3d = self.d3d(x.permute(0, 2, 1, 3, 4)).mean(dim=2)
        if out3d.shape[-2:] != out2d.shape[-2:]:
            out3d = F.interpolate(
                out3d,
                size=out2d.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        weight2d = torch.sigmoid(self.alpha2d)
        fused = weight2d * out2d + (1.0 - weight2d) * out3d
        return fused.reshape(batch, -1)


__all__ = ["P2IDiscriminator"]
