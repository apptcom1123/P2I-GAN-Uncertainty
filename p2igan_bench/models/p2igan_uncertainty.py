"""Four P2I-GAN uncertainty variants with one explicit output contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from p2igan_bench.modules.bayesian import (
    BayesianConv2d, bayesian_kl_divergence, sampled_bayesian_weights,
)
from p2igan_bench.modules.deconv_pytorch import DOConv2d
from p2igan_bench.modules.layer import (
    BaseNetwork, BasicConv_do, DownsampleDuplicateChannels, InputBlock,
    ResBlock_do, UPPos,
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
    def __init__(self, channels: int, num_res: int, residual_scale: float,
                 norm_groups: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(*[
            ResBlock_do(channels, residual_scale=residual_scale, norm_groups=norm_groups)
            for _ in range(num_res)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class P2IEncoder(nn.Module):
    def __init__(self, *, length: int, base: int, dropout: float) -> None:
        super().__init__()
        self.input = InputBlock(depth=2, k=4, rho=2.0, tau=0.05, chunk=16384)
        self.conv_in = BasicConv_do(length, base, kernel_size=3, relu=False, stride=1, groups=4)
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
                 norm_groups: int, output_relu: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            EBlock(base * m, num_res, residual_scale, norm_groups)
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
        self.output = BasicConv_do(
            base, length, kernel_size=1, relu=False, stride=1, groups=4
        )
        self.output_positive = bool(output_relu)

    def forward(self, features) -> torch.Tensor:
        x0, x2, x4, x8 = features
        h = self.ups[2](self.dropout(self.blocks[3](x8)))
        h = self.ups[1](self.dropout(self.blocks[2](x4 + h)))
        h = self.ups[0](self.dropout(self.blocks[1](x2 + h)))
        h = self.blocks[0](x0 + h)
        output = self.output(self.output_norm(self.dropout(h)))
        # A terminal ReLU can permanently kill a precipitation output channel:
        # negative logits become exactly zero and receive no mean-loss gradient.
        # Softplus preserves non-negativity while retaining a useful gradient.
        return F.softplus(output, beta=5.0) if self.output_positive else output


class BayesianResBlock(nn.Module):
    """Pre-normalized residual block with Bayesian convolution kernels."""

    def __init__(self, channels: int, residual_scale: float, norm_groups: int,
                 prior_scale: float, posterior_init_scale: float) -> None:
        super().__init__()
        groups = min(norm_groups, channels)
        while channels % groups:
            groups -= 1
        bayes = dict(prior_scale=prior_scale, posterior_init_scale=posterior_init_scale)
        self.norm = nn.GroupNorm(groups, channels)
        self.conv1 = BayesianConv2d(channels, channels, 3, padding=1, bias=False, **bayes)
        self.conv2 = BayesianConv2d(channels, channels, 3, padding=1, bias=False, **bayes)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.conv1(self.norm(x)), inplace=True)
        return x + self.residual_scale * self.conv2(residual)


class BayesianEBlock(nn.Module):
    def __init__(self, channels: int, num_res: int, residual_scale: float,
                 norm_groups: int, prior_scale: float,
                 posterior_init_scale: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(*[
            BayesianResBlock(
                channels, residual_scale, norm_groups,
                prior_scale, posterior_init_scale,
            )
            for _ in range(num_res)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BayesianUPPos(nn.Module):
    """UPPos whose channel projection is a Bayesian convolution."""

    def __init__(self, in_ch: int, out_ch: int, height: int, width: int,
                 prior_scale: float, posterior_init_scale: float) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, 1, height, width))
        self.proj = BayesianConv2d(
            in_ch, out_ch, 1, bias=True,
            prior_scale=prior_scale,
            posterior_init_scale=posterior_init_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        pos = 2.0 * torch.sigmoid(self.pos) - 1.0
        x = x + x * pos
        return F.relu(self.proj(x), inplace=True)


class BayesianP2IDecoder(nn.Module):
    """P2I decoder with mean-field Bayesian convolutions and unchanged geometry."""

    def __init__(self, *, base: int, length: int, height: int, width: int,
                 num_res: int, residual_scale: float, norm_groups: int,
                 prior_scale: float, posterior_init_scale: float,
                 output_relu: bool) -> None:
        super().__init__()
        block = dict(
            num_res=num_res,
            residual_scale=residual_scale,
            norm_groups=norm_groups,
            prior_scale=prior_scale,
            posterior_init_scale=posterior_init_scale,
        )
        self.blocks = nn.ModuleList([
            BayesianEBlock(channels=base * multiplier, **block)
            for multiplier in (1, 2, 4, 8)
        ])
        self.ups = nn.ModuleList([
            BayesianUPPos(base * 2, base, height, width, prior_scale, posterior_init_scale),
            BayesianUPPos(base * 4, base * 2, height // 2, width // 2,
                          prior_scale, posterior_init_scale),
            BayesianUPPos(base * 8, base * 4, height // 4, width // 4,
                          prior_scale, posterior_init_scale),
        ])
        groups = min(norm_groups, base)
        while base % groups:
            groups -= 1
        self.output_norm = nn.GroupNorm(groups, base)
        self.output = BayesianConv2d(
            base, length, 1, groups=4, bias=False,
            prior_scale=prior_scale,
            posterior_init_scale=posterior_init_scale,
        )
        self.output_positive = bool(output_relu)

    def forward(self, features) -> torch.Tensor:
        x0, x2, x4, x8 = features
        h = self.ups[2](self.blocks[3](x8))
        h = self.ups[1](self.blocks[2](x4 + h))
        h = self.ups[0](self.blocks[1](x2 + h))
        h = self.output(self.output_norm(self.blocks[0](x0 + h)))
        # Keep the probabilistic mean non-negative without the zero-gradient
        # region of ReLU. This is especially important for randomly initialized
        # Bayesian output kernels, whose logits can initially be mostly negative.
        return F.softplus(h, beta=5.0) if self.output_positive else h

    @staticmethod
    def _effective_doconv_weight(layer: DOConv2d) -> torch.Tensor:
        kernel_height, kernel_width = layer.kernel_size
        output_shape = (
            layer.out_channels,
            layer.in_channels // layer.groups,
            kernel_height,
            kernel_width,
        )
        if kernel_height * kernel_width == 1:
            return layer.W.reshape(output_shape)
        basis = layer.D + layer.D_diag
        coefficients = layer.W.reshape(
            layer.out_channels // layer.groups,
            layer.in_channels,
            layer.D_mul,
        )
        return torch.einsum("ims,ois->oim", basis, coefficients).reshape(output_shape)

    @classmethod
    def _copy_doconv(cls, target: BayesianConv2d, source: DOConv2d) -> None:
        target.set_posterior_mean(cls._effective_doconv_weight(source), source.bias)

    def initialize_from_deterministic(self, source: P2IDecoder) -> None:
        """Copy a trained DOConv decoder into the Bayesian posterior means."""
        for target_block, source_block in zip(self.blocks, source.blocks):
            for target_residual, source_residual in zip(
                target_block.layers, source_block.layers
            ):
                target_residual.norm.load_state_dict(source_residual.norm.state_dict())
                self._copy_doconv(
                    target_residual.conv1, source_residual.main[0].main[0]
                )
                self._copy_doconv(
                    target_residual.conv2, source_residual.main[1].main[0]
                )
        for target_up, source_up in zip(self.ups, source.ups):
            target_up.pos.data.copy_(source_up.pos.data)
            target_up.proj.set_posterior_mean(
                source_up.proj.weight,
                source_up.proj.bias,
            )
        self.output_norm.load_state_dict(source.output_norm.state_dict())
        self._copy_doconv(self.output, source.output.main[0])


class CoModUp(nn.Module):
    """Upsample using one condition-routed mixture of gain/bias experts."""

    def __init__(self, in_ch: int, out_ch: int, height: int, width: int,
                 num_experts: int, comod_scale: float) -> None:
        super().__init__()
        groups = min(8, in_ch)
        while in_ch % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, in_ch)
        self.num_experts = int(num_experts)
        if self.num_experts < 1:
            raise ValueError("num_experts must be positive")

        pos = torch.randn(self.num_experts, 2 * in_ch, height, width) * 0.01
        pos -= pos.mean(dim=0, keepdim=True)
        self.pos = nn.Parameter(pos)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.comod_scale = float(comod_scale)

    def _mix_experts(self, weights: torch.Tensor) -> torch.Tensor:
        if weights.ndim != 2 or weights.shape[1] != self.num_experts:
            raise ValueError(
                "expert weights must have shape "
                f"[B, {self.num_experts}], got {tuple(weights.shape)}"
            )
        return torch.einsum("be,echw->bchw", weights, self.pos)

    def forward(self, x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        x = self.norm(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        modulation = self._mix_experts(expert_weights)
        if modulation.shape[-2:] != x.shape[-2:]:
            modulation = F.interpolate(
                modulation, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        gain_delta, bias = modulation.chunk(2, dim=1)
        gain = 1.0 + self.comod_scale * gain_delta
        bias = self.comod_scale * bias
        return self.proj(gain * x + bias)


class CoModDecoder(nn.Module):
    def __init__(self, *, base: int, length: int, height: int, width: int,
                 num_res: int, num_experts: int, comod_scale: float,
                 dropout: float, residual_scale: float, norm_groups: int) -> None:
        super().__init__()
        self.deep = EBlock(base * 8, num_res, residual_scale, norm_groups)
        self.middle = nn.ModuleList([
            EBlock(base * 4, num_res, residual_scale, norm_groups),
            EBlock(base * 2, num_res, residual_scale, norm_groups),
        ])
        self.ups = nn.ModuleList([
            CoModUp(base * 8, base * 4, height // 4, width // 4, num_experts,
                    comod_scale),
            CoModUp(base * 4, base * 2, height // 2, width // 2, num_experts,
                    comod_scale),
            CoModUp(base * 2, base, height, width, num_experts,
                    comod_scale),
        ])
        self.dropout = nn.Dropout2d(dropout)
        self.final = EBlock(base, num_res, residual_scale, norm_groups)
        groups = min(norm_groups, base)
        while base % groups:
            groups -= 1
        self.output_norm = nn.GroupNorm(groups, base)
        self.output = BasicConv_do(base, length, kernel_size=1, relu=True, stride=1, groups=4)

    def forward(self, features, condition_weights) -> torch.Tensor:
        x0, x2, x4, x8 = features
        h = self.ups[0](self.dropout(self.deep(x8)), condition_weights[:, 0])
        h = self.ups[1](self.dropout(self.middle[0](h + x4)), condition_weights[:, 1])
        h = self.ups[2](self.dropout(self.middle[1](h + x2)), condition_weights[:, 2])
        return self.output(self.output_norm(self.dropout(self.final(h + x0))))


class _P2IUncertaintyBase(BaseNetwork):
    secondary_kind = "none"

    def __init__(self, config, *, init_weights: bool = True) -> None:
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
        self.logvar_min = float(loss.get("logvar_clamp_min", -10.0))
        self.logvar_max = float(loss.get("logvar_clamp_max", 8.0))
        self._init_requested = init_weights

    @staticmethod
    def _mean(raw: torch.Tensor, shape) -> torch.Tensor:
        return raw.reshape(shape)

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
    """M1: deterministic mean/log-variance heads with no uncertainty GAN."""
    secondary_kind = "none"

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.encoder = P2IEncoder(length=self.length, base=self.base, dropout=0.0)
        tail = dict(base=self.base, length=self.length, height=self.height, width=self.width,
                    num_res=self.num_res, dropout=0.0, residual_scale=self.residual_scale,
                    norm_groups=self.residual_norm_groups)
        self.mean_decoder = P2IDecoder(**tail, output_relu=True)
        self.logvar_decoder = P2IDecoder(**tail, output_relu=False)
        if self._init_requested:
            self.init_weights()

    def forward(self, masked, mask, **_) -> UncertaintyOutput:
        f = self.encoder(masked, mask)
        mean = self._mean(self.mean_decoder(f), masked.shape)
        logvar = self._logvar(self.logvar_decoder(f), mean.shape)
        return self._pack(mean.unsqueeze(0), logvar.unsqueeze(0), kind=self.secondary_kind)


class P2IBNNJointGenerator(_P2IUncertaintyBase):
    """Deterministic P2I encoder with Bayesian mean and log-variance decoders."""
    secondary_kind = "bnn_weight"

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        model = config.get("model", {})
        self.prior_scale = float(model.get("prior_scale", 0.10))
        self.posterior_init_scale = float(model.get("posterior_init_scale", 0.005))
        self.encoder = P2IEncoder(length=self.length, base=self.base, dropout=0.0)
        tail = dict(
            base=self.base,
            length=self.length,
            height=self.height,
            width=self.width,
            num_res=self.num_res,
            residual_scale=self.residual_scale,
            norm_groups=self.residual_norm_groups,
            prior_scale=self.prior_scale,
            posterior_init_scale=self.posterior_init_scale,
        )
        self.mean_decoder = BayesianP2IDecoder(**tail, output_relu=True)
        self.logvar_decoder = BayesianP2IDecoder(**tail, output_relu=False)
        if self._init_requested:
            # BayesianConv2d initializes its posterior mean and scale itself;
            # this initializes only the unchanged deterministic P2I encoder.
            self.init_weights()

    def initialize_from_joint(self, source: P2IJointGenerator) -> None:
        """Use a trained deterministic joint model as the posterior-mean start."""
        self.encoder.load_state_dict(source.encoder.state_dict())
        self.mean_decoder.initialize_from_deterministic(source.mean_decoder)
        self.logvar_decoder.initialize_from_deterministic(source.logvar_decoder)

    def kl_divergence(self) -> torch.Tensor:
        return (
            bayesian_kl_divergence(self.mean_decoder)
            + bayesian_kl_divergence(self.logvar_decoder)
        )

    def forward(self, masked, mask, *, num_members: int = 1,
                sample_seeds: Optional[torch.Tensor] = None, **_) -> UncertaintyOutput:
        features = self.encoder(masked, mask)
        means, logvars = [], []
        member_count = max(1, num_members)
        if sample_seeds is not None and sample_seeds.numel() != member_count:
            raise ValueError(
                f"sample_seeds must contain {member_count} values, "
                f"got {sample_seeds.numel()}"
            )
        for index in range(member_count):
            generator = None
            if sample_seeds is not None:
                generator = torch.Generator(device=masked.device)
                generator.manual_seed(int(sample_seeds[index].item()))
            with sampled_bayesian_weights(
                self.mean_decoder,
                self.logvar_decoder,
                generator=generator,
            ):
                mean = self._mean(self.mean_decoder(features), masked.shape)
                logvar = self._logvar(self.logvar_decoder(features), masked.shape)
            means.append(mean)
            logvars.append(logvar)
        return self._pack(
            torch.stack(means),
            torch.stack(logvars),
            kind=self.secondary_kind,
        )


class P2IDecoderEnsembleGenerator(_P2IUncertaintyBase):
    """M2: independent mean decoders with one shared aleatoric-variance decoder."""
    secondary_kind = "between_decoder"

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        heads = int(config.get("model", {}).get("num_heads", 4))
        self.encoder = P2IEncoder(length=self.length, base=self.base, dropout=0.0)
        tail = dict(base=self.base, length=self.length, height=self.height, width=self.width,
                    num_res=self.num_res, dropout=0.0, residual_scale=self.residual_scale,
                    norm_groups=self.residual_norm_groups)
        self.mean_decoders = nn.ModuleList([
            P2IDecoder(**tail, output_relu=True) for _ in range(heads)
        ])
        self.logvar_decoder = P2IDecoder(**tail, output_relu=False)
        if self._init_requested:
            self.init_weights()

    def forward(self, masked, mask, **_) -> UncertaintyOutput:
        f = self.encoder(masked, mask)
        means = torch.stack([self._mean(d(f), masked.shape) for d in self.mean_decoders])
        logvar = self._logvar(self.logvar_decoder(f), means[0].shape)
        return self._pack(
            means,
            logvar.unsqueeze(0).expand_as(means),
            kind=self.secondary_kind,
        )


class P2IMCDropoutGenerator(_P2IUncertaintyBase):
    """M3: one dual-head network sampled repeatedly with MC dropout."""
    secondary_kind = "mc_dropout"

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.encoder = P2IEncoder(length=self.length, base=self.base, dropout=self.dropout_rate)
        tail = dict(base=self.base, length=self.length, height=self.height, width=self.width,
                    num_res=self.num_res, dropout=self.dropout_rate, residual_scale=self.residual_scale,
                    norm_groups=self.residual_norm_groups)
        self.mean_decoder = P2IDecoder(**tail, output_relu=True)
        self.logvar_decoder = P2IDecoder(**tail, output_relu=False)
        if self._init_requested:
            self.init_weights()

    def forward(self, masked, mask, *, num_members: int = 1,
                sample_seeds: Optional[torch.Tensor] = None, **_) -> UncertaintyOutput:
        means, logvars = [], []
        member_count = max(1, num_members)
        if sample_seeds is not None and sample_seeds.numel() != member_count:
            raise ValueError(
                f"sample_seeds must contain {member_count} values, "
                f"got {sample_seeds.numel()}"
            )

        def sample_member():
            features = self.encoder(masked, mask)
            mean = self._mean(self.mean_decoder(features), masked.shape)
            means.append(mean)
            logvars.append(self._logvar(self.logvar_decoder(features), mean.shape))

        for index in range(member_count):
            if sample_seeds is None:
                sample_member()
                continue
            cuda_devices = []
            if masked.is_cuda:
                cuda_devices = [
                    masked.device.index
                    if masked.device.index is not None
                    else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(int(sample_seeds[index].item()))
                sample_member()
        return self._pack(torch.stack(means), torch.stack(logvars), kind=self.secondary_kind)


class P2IConditionCoModGenerator(_P2IUncertaintyBase):
    """M4: MC dropout with condition-routed spatial gain/bias experts."""
    secondary_kind = "mc_dropout_condition"

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        model = config.get("model", {})
        self.num_modulation_experts = int(model.get("num_modulation_experts", 16))
        self.condition_embedding_dim = int(model.get("condition_embedding_dim", 32))
        if self.condition_embedding_dim < 1:
            raise ValueError("condition_embedding_dim must be positive")
        legacy_scale = float(model.get("modulation_scale", 0.10))
        comod_scale = float(model.get("comod_scale", legacy_scale))
        self.encoder = P2IEncoder(
            length=self.length, base=self.base, dropout=self.dropout_rate
        )

        # Each encoder scale is first reduced to one channel per frame using a
        # parameter-free within-frame mean.  Mean/max/std regional summaries
        # from all four scales become three 32-D stage embeddings.  A small,
        # stage-specific linear router maps each embedding to 16 expert logits.
        self.condition_grid_size = int(model.get("condition_grid_size", 4))
        if self.condition_grid_size < 1:
            raise ValueError("condition_grid_size must be positive")
        for channels in (self.base, self.base * 2, self.base * 4, self.base * 8):
            if channels % self.length:
                raise ValueError(
                    f"encoder channels ({channels}) must be divisible by "
                    f"sample length ({self.length}) for condition compression"
                )
        condition_channels = 3 * 4 * self.length
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(condition_channels, 4 * self.length, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                4 * self.length,
                self.condition_embedding_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(
                self.condition_embedding_dim
                * self.condition_grid_size
                * self.condition_grid_size,
                3 * self.condition_embedding_dim,
            ),
            nn.SiLU(inplace=True),
        )
        self.condition_routers = nn.ModuleList([
            nn.Linear(self.condition_embedding_dim, self.num_modulation_experts)
            for _ in range(3)
        ])
        self.mean_decoder = CoModDecoder(base=self.base, length=self.length,
                                         height=self.height, width=self.width,
                                         num_res=self.num_res,
                                         comod_scale=comod_scale,
                                         dropout=self.dropout_rate,
                                         num_experts=self.num_modulation_experts,
                                         residual_scale=self.residual_scale,
                                         norm_groups=self.residual_norm_groups)
        self.logvar_decoder = P2IDecoder(base=self.base, length=self.length, height=self.height, width=self.width,
                                         num_res=self.num_res, dropout=self.dropout_rate,
                                         residual_scale=self.residual_scale,
                                         norm_groups=self.residual_norm_groups,
                                         output_relu=False)
        if self._init_requested:
            self.init_weights()
            # Zero condition embeddings make every router logit zero at the
            # first forward pass, hence uniform expert weights.  Because the
            # routers themselves retain non-zero weights, gradients reach this
            # output layer immediately instead of being blocked by a zero
            # router matrix.
            nn.init.zeros_(self.condition_encoder[-2].weight)
            nn.init.zeros_(self.condition_encoder[-2].bias)

    def _compress_condition_scale(self, feature: torch.Tensor):
        """Compress ``[B,C,H,W]`` into three ``[B,T,G,G]`` statistics."""
        b, channels, height, width = feature.shape
        if channels % self.length:
            raise ValueError(
                f"feature channels ({channels}) must be divisible by "
                f"sample length ({self.length})"
            )
        feature = feature.reshape(
            b, self.length, channels // self.length, height, width
        ).mean(dim=2)
        output_size = (self.condition_grid_size, self.condition_grid_size)
        regional_mean = F.adaptive_avg_pool2d(feature, output_size)
        regional_max = F.adaptive_max_pool2d(feature, output_size)
        regional_mean_square = F.adaptive_avg_pool2d(feature.square(), output_size)
        regional_std = (
            regional_mean_square - regional_mean.square()
        ).clamp_min(0.0).add(1e-6).sqrt()
        return regional_mean, regional_max, regional_std

    def _condition_distribution(self, features):
        """Map four-scale statistics to embeddings and expert probabilities.

        With the default ``T=16``, ``G=4``, embedding size 32 and 16 experts:
        statistics are ``[B,192,4,4]``, embeddings are ``[B,3,32]``, and
        returned expert probabilities are ``[B,3,16]``.
        """
        summaries = [self._compress_condition_scale(feature) for feature in features]
        statistics = torch.cat([
            torch.cat([summary[index] for summary in summaries], dim=1)
            for index in range(3)
        ], dim=1)
        embeddings = self.condition_encoder(statistics).reshape(
            -1, 3, self.condition_embedding_dim
        )
        logits = torch.stack([
            router(embeddings[:, stage])
            for stage, router in enumerate(self.condition_routers)
        ], dim=1)
        return embeddings, F.softmax(logits, dim=-1)

    def forward(self, masked, mask, *, num_members: int = 1,
                sample_seeds: Optional[torch.Tensor] = None, **_) -> UncertaintyOutput:
        means, logvars, embeddings, expert_weights = [], [], [], []
        member_count = max(1, num_members)
        if sample_seeds is not None and sample_seeds.numel() != member_count:
            raise ValueError(
                f"sample_seeds must contain {member_count} values, "
                f"got {sample_seeds.numel()}"
            )

        def sample_member():
            # Re-running the full stochastic path gives every member its own
            # dropout sample and therefore its own condition reconstruction.
            features = self.encoder(masked, mask)
            condition_embedding, condition_weights = self._condition_distribution(features)
            mean = self._mean(
                self.mean_decoder(features, condition_weights), masked.shape
            )
            means.append(mean)
            logvars.append(self._logvar(self.logvar_decoder(features), mean.shape))
            embeddings.append(condition_embedding)
            expert_weights.append(condition_weights)

        for index in range(member_count):
            if sample_seeds is None:
                sample_member()
                continue
            cuda_devices = []
            if masked.is_cuda:
                cuda_devices = [
                    masked.device.index
                    if masked.device.index is not None
                    else torch.cuda.current_device()
                ]
            # Reusing these seeds for every sliding window keeps member i tied
            # to the same MC-dropout mask sequence across the complete event.
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(int(sample_seeds[index].item()))
                sample_member()
        return self._pack(
            torch.stack(means),
            torch.stack(logvars),
            kind=self.secondary_kind,
            aux={
                "condition_embedding": torch.stack(embeddings),
                "condition_expert_weights": torch.stack(expert_weights),
            },
        )
