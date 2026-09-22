"""Four P2I-GAN uncertainty variants with one explicit output contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from p2igan_bench.modules.layer import (
    BaseNetwork, BasicConv_do, BasicConv_do_eval, DownsampleDuplicateChannels,
    InputBlock, ResBlock_do, ResBlock_do_eval, UPPos,
)


@dataclass
class UncertaintyOutput:
    """Canonical output. Member tensors are ``[S,B,T,C,H,W]``."""
    member_mean: torch.Tensor
    member_logvar: torch.Tensor
    mean: torch.Tensor
    aleatoric_var: torch.Tensor
    secondary_var: torch.Tensor
    predictive_var: torch.Tensor
    secondary_kind: str
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)


class EBlock(nn.Module):
    def __init__(self, channels: int, num_res: int, block: type[nn.Module],
                 residual_scale: float, norm_groups: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(*[
            block(channels, residual_scale=residual_scale, norm_groups=norm_groups)
            for _ in range(num_res)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class P2IEncoder(nn.Module):
    def __init__(self, *, length: int, base: int, dropout: float, inference: bool) -> None:
        super().__init__()
        conv = BasicConv_do_eval if inference else BasicConv_do
        self.input = InputBlock(depth=2, k=4, rho=2.0, tau=0.05, chunk=16384)
        self.conv_in = conv(length, base, kernel_size=3, relu=False, stride=1, groups=4)
        self.downsample = DownsampleDuplicateChannels(length=length)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, masked: torch.Tensor, mask: torch.Tensor):
        b, t, c, h, w = masked.shape
        x = self.input(masked.reshape(b, t * c, h, w), mask.reshape(b, t * c, h, w)).float()
        encoded = self.conv_in(x)
        residual = x.repeat_interleave(4, dim=1)
        x0 = self.dropout(encoded + residual if encoded.shape[1] == residual.shape[1] else encoded)
        x0 = x0.reshape(b, x0.shape[1], h, w)
        x2 = self.downsample(x0)
        x4 = self.downsample(x2)
        return x0, x2, x4, self.downsample(x4)


class P2IDecoder(nn.Module):
    """A complete decoder tail; ensemble heads never share this module."""
    def __init__(self, *, base: int, length: int, height: int, width: int,
                 num_res: int, dropout: float, residual_scale: float,
                 norm_groups: int, inference: bool) -> None:
        super().__init__()
        block = ResBlock_do_eval if inference else ResBlock_do
        conv = BasicConv_do_eval if inference else BasicConv_do
        self.blocks = nn.ModuleList([
            EBlock(base * m, num_res, block, residual_scale, norm_groups)
            for m in (1, 2, 4, 8)
        ])
        self.ups = nn.ModuleList([
            UPPos(in_ch=base * 2, out_ch=base, H=height, W=width, T=length),
            UPPos(in_ch=base * 4, out_ch=base * 2, H=height // 2, W=width // 2, T=length),
            UPPos(in_ch=base * 8, out_ch=base * 4, H=height // 4, W=width // 4, T=length),
        ])
        self.dropout = nn.Dropout2d(dropout)
        groups = min(norm_groups, base)
        while base % groups:
            groups -= 1
        self.output_norm = nn.GroupNorm(groups, base)
        self.output = conv(base, length, kernel_size=1, relu=False, stride=1, groups=4)

    def forward(self, features) -> torch.Tensor:
        x0, x2, x4, x8 = features
        h = self.ups[2](self.dropout(self.blocks[3](x8)))
        h = self.ups[1](self.dropout(self.blocks[2](x4 + h)))
        h = self.ups[0](self.dropout(self.blocks[1](x2 + h)))
        h = self.blocks[0](x0 + h)
        return self.output(self.output_norm(self.dropout(h)))


class CoModUp(nn.Module):
    """Bounded global-style and spatial-condition modulation."""
    def __init__(self, in_ch: int, out_ch: int, cond_ch: int, style_dim: int, scale: float) -> None:
        super().__init__()
        groups = min(8, in_ch)
        while in_ch % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, in_ch)
        self.global_affine = nn.Linear(style_dim, 2 * in_ch)
        self.spatial_affine = nn.Conv2d(cond_ch, 2 * in_ch, 1)
        self.proj = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.scale = scale
        nn.init.zeros_(self.global_affine.weight)
        nn.init.zeros_(self.global_affine.bias)
        nn.init.zeros_(self.spatial_affine.weight)
        nn.init.zeros_(self.spatial_affine.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = self.norm(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        ga = self.global_affine(style).unsqueeze(-1).unsqueeze(-1)
        sa = self.spatial_affine(cond)
        if sa.shape[-2:] != x.shape[-2:]:
            sa = F.interpolate(sa, x.shape[-2:], mode="bilinear", align_corners=False)
        gg, bg = ga.chunk(2, dim=1)
        gs, bs = sa.chunk(2, dim=1)
        gain = 1.0 + self.scale * (torch.tanh(gg) + torch.tanh(gs))
        bias = self.scale * (torch.tanh(bg) + torch.tanh(bs))
        return F.relu(self.proj(gain * x + bias), inplace=True)


class CoModDecoder(nn.Module):
    def __init__(self, *, base: int, length: int, num_res: int, style_dim: int,
                 scale: float, residual_scale: float, norm_groups: int,
                 inference: bool) -> None:
        super().__init__()
        block = ResBlock_do_eval if inference else ResBlock_do
        conv = BasicConv_do_eval if inference else BasicConv_do
        self.deep = EBlock(base * 8, num_res, block, residual_scale, norm_groups)
        self.middle = nn.ModuleList([
            EBlock(base * 4, num_res, block, residual_scale, norm_groups),
            EBlock(base * 2, num_res, block, residual_scale, norm_groups),
        ])
        self.ups = nn.ModuleList([
            CoModUp(base * 8, base * 4, base * 4, style_dim, scale),
            CoModUp(base * 4, base * 2, base * 2, style_dim, scale),
            CoModUp(base * 2, base, base, style_dim, scale),
        ])
        self.final = EBlock(base, num_res, block, residual_scale, norm_groups)
        groups = min(norm_groups, base)
        while base % groups:
            groups -= 1
        self.output_norm = nn.GroupNorm(groups, base)
        self.output = conv(base, length, kernel_size=1, relu=False, stride=1, groups=4)

    def forward(self, features, styles) -> torch.Tensor:
        x0, x2, x4, x8 = features
        h = self.ups[0](self.deep(x8), x4, styles[0])
        h = self.ups[1](self.middle[0](h + x4), x2, styles[1])
        h = self.ups[2](self.middle[1](h + x2), x0, styles[2])
        return self.output(self.output_norm(self.final(h + x0)))


class _P2IUncertaintyBase(BaseNetwork):
    secondary_kind = "none"

    def __init__(self, config, *, inference: bool = False, init_weights: bool = True) -> None:
        super().__init__()
        data = config.get("data_loader") or config["data"]["train"]
        model, loss = config.get("model", {}), config.get("loss", {})
        self.length = int(data.get("sample_length", 16))
        self.height, self.width = int(data["h"]), int(data["w"])
        self.base = int(model.get("base_channels", 64))
        self.num_res = int(model.get("num_res", 4))
        self.residual_scale = float(model.get("residual_scale", 0.1))
        self.residual_norm_groups = int(model.get("residual_norm_groups", 8))
        self.dropout_rate = float(model.get("mc_dropout_rate", 0.0))
        self.preserve_observations = bool(model.get("preserve_observations", True))
        self.output_activation = str(model.get("output_activation", "sigmoid")).lower()
        self.logvar_min = float(loss.get("logvar_clamp_min", -10.0))
        self.logvar_max = float(loss.get("logvar_clamp_max", 8.0))
        self.inference, self._init_requested = inference, init_weights

    def _mean(self, raw: torch.Tensor, masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raw = raw.reshape(masked.shape)
        if self.output_activation == "sigmoid":
            raw = torch.sigmoid(raw)
        elif self.output_activation == "tanh":
            raw = torch.tanh(raw)
        elif self.output_activation != "identity":
            raise ValueError(f"Unsupported output activation: {self.output_activation}")
        return mask * masked + (1.0 - mask) * raw if self.preserve_observations else raw

    def _logvar(self, raw: torch.Tensor, shape) -> torch.Tensor:
        return raw.reshape(shape).clamp(self.logvar_min, self.logvar_max)

    def _pack(self, means: torch.Tensor, logvars: torch.Tensor, *, kind: str,
              aux: Optional[Dict[str, torch.Tensor]] = None) -> UncertaintyOutput:
        mean = means.mean(dim=0)
        aleatoric = logvars.exp().mean(dim=0)
        secondary = means.var(dim=0, unbiased=False)
        return UncertaintyOutput(means, logvars, mean, aleatoric, secondary,
                                 (aleatoric + secondary).clamp_min(1e-6), kind, aux or {})


class P2IJointGenerator(_P2IUncertaintyBase):
    """M1: deterministic dual head; sigma enters a joint discriminator during training."""
    secondary_kind = "none"

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.encoder = P2IEncoder(length=self.length, base=self.base, dropout=0.0, inference=self.inference)
        tail = dict(base=self.base, length=self.length, height=self.height, width=self.width,
                    num_res=self.num_res, dropout=0.0, residual_scale=self.residual_scale,
                    norm_groups=self.residual_norm_groups, inference=self.inference)
        self.mean_decoder, self.logvar_decoder = P2IDecoder(**tail), P2IDecoder(**tail)
        if self._init_requested:
            self.init_weights()

    def forward(self, masked, mask, **_) -> UncertaintyOutput:
        f = self.encoder(masked, mask)
        mean = self._mean(self.mean_decoder(f), masked, mask)
        logvar = self._logvar(self.logvar_decoder(f), mean.shape)
        return self._pack(mean.unsqueeze(0), logvar.unsqueeze(0), kind=self.secondary_kind)


class P2IDecoderEnsembleGenerator(_P2IUncertaintyBase):
    """M2: shared encoder and independently initialized complete decoder tails."""
    secondary_kind = "between_decoder"

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        heads = int(config.get("model", {}).get("num_heads", 4))
        self.encoder = P2IEncoder(length=self.length, base=self.base, dropout=0.0, inference=self.inference)
        tail = dict(base=self.base, length=self.length, height=self.height, width=self.width,
                    num_res=self.num_res, dropout=0.0, residual_scale=self.residual_scale,
                    norm_groups=self.residual_norm_groups, inference=self.inference)
        self.mean_decoders = nn.ModuleList([P2IDecoder(**tail) for _ in range(heads)])
        self.logvar_decoders = nn.ModuleList([P2IDecoder(**tail) for _ in range(heads)])
        if self._init_requested:
            self.init_weights()

    def forward(self, masked, mask, **_) -> UncertaintyOutput:
        f = self.encoder(masked, mask)
        means = torch.stack([self._mean(d(f), masked, mask) for d in self.mean_decoders])
        logvars = torch.stack([self._logvar(d(f), means[0].shape) for d in self.logvar_decoders])
        return self._pack(means, logvars, kind=self.secondary_kind)


class P2IMCDropoutGenerator(_P2IUncertaintyBase):
    """M3: one dual-head network sampled repeatedly with MC dropout."""
    secondary_kind = "mc_dropout"

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.encoder = P2IEncoder(length=self.length, base=self.base, dropout=self.dropout_rate, inference=self.inference)
        tail = dict(base=self.base, length=self.length, height=self.height, width=self.width,
                    num_res=self.num_res, dropout=self.dropout_rate, residual_scale=self.residual_scale,
                    norm_groups=self.residual_norm_groups, inference=self.inference)
        self.mean_decoder, self.logvar_decoder = P2IDecoder(**tail), P2IDecoder(**tail)
        if self._init_requested:
            self.init_weights()

    def forward(self, masked, mask, *, num_members: int = 1, **_) -> UncertaintyOutput:
        means, logvars = [], []
        for _ in range(max(1, num_members)):
            f = self.encoder(masked, mask)
            mean = self._mean(self.mean_decoder(f), masked, mask)
            means.append(mean)
            logvars.append(self._logvar(self.logvar_decoder(f), mean.shape))
        return self._pack(torch.stack(means), torch.stack(logvars), kind=self.secondary_kind)


class P2ILatentCoModGenerator(_P2IUncertaintyBase):
    """M4: latent z drives conditional/stochastic co-modulation of the mean decoder."""
    secondary_kind = "latent_style"

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        model = config.get("model", {})
        self.latent_dim = int(model.get("latent_dim", 64))
        self.style_dim = int(model.get("style_dim", 128))
        scale = float(model.get("modulation_scale", 0.10))
        self.encoder = P2IEncoder(length=self.length, base=self.base, dropout=0.0, inference=self.inference)
        self.mapping = nn.Sequential(
            nn.Linear(self.base * 8 + self.latent_dim + 1, self.style_dim * 2), nn.LeakyReLU(0.2),
            nn.Linear(self.style_dim * 2, self.style_dim * 3),
        )
        self.mean_decoder = CoModDecoder(base=self.base, length=self.length, num_res=self.num_res,
                                         style_dim=self.style_dim, scale=scale,
                                         residual_scale=self.residual_scale,
                                         norm_groups=self.residual_norm_groups,
                                         inference=self.inference)
        self.logvar_decoder = P2IDecoder(base=self.base, length=self.length, height=self.height, width=self.width,
                                         num_res=self.num_res, dropout=0.0,
                                         residual_scale=self.residual_scale,
                                         norm_groups=self.residual_norm_groups,
                                         inference=self.inference)
        if self._init_requested:
            self.init_weights()
            for up in self.mean_decoder.ups:
                nn.init.zeros_(up.global_affine.weight)
                nn.init.zeros_(up.global_affine.bias)
                nn.init.zeros_(up.spatial_affine.weight)
                nn.init.zeros_(up.spatial_affine.bias)

    def forward(self, masked, mask, *, num_members: int = 1, z: Optional[torch.Tensor] = None, **_) -> UncertaintyOutput:
        f = self.encoder(masked, mask)
        b = masked.shape[0]
        logvar = self._logvar(self.logvar_decoder(f), masked.shape)
        pooled = F.adaptive_avg_pool2d(f[-1], 1).flatten(1)
        density = mask.float().mean(dim=(1, 2, 3, 4)).unsqueeze(1)
        means, zs = [], []
        for index in range(max(1, num_members)):
            zi = z[index] if z is not None and z.ndim == 3 else z
            if zi is None:
                zi = torch.randn(b, self.latent_dim, device=masked.device, dtype=masked.dtype)
            styles = self.mapping(torch.cat([pooled, zi, density], dim=1)).chunk(3, dim=1)
            means.append(self._mean(self.mean_decoder(f, styles), masked, mask))
            zs.append(zi)
        means_t = torch.stack(means)
        return self._pack(means_t, logvar.unsqueeze(0).expand_as(means_t), kind=self.secondary_kind,
                          aux={"z": torch.stack(zs)})
