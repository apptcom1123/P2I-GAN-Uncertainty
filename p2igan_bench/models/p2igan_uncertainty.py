"""P2I-GAN backbone adapted for heteroscedastic uncertainty estimation."""

from __future__ import annotations

import torch
import torch.nn as nn

from p2igan_bench.modules.layer import (
    BaseNetwork,
    BasicConv_do,
    BasicConv_do_eval,
    ResBlock_do,
    ResBlock_do_eval,
    InputBlock,
    DownsampleDuplicateChannels,
    UPPos,
    UPPosMod,
)


class EBlock(nn.Module):
    def __init__(self, out_channel: int, num_res: int = 8, res_block: type[nn.Module] = ResBlock_do):
        super().__init__()
        self.layers = nn.Sequential(*[res_block(out_channel) for _ in range(num_res)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _ExpertGateMLP(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _BranchDecoder(nn.Module):
    def __init__(
        self,
        *,
        base_channel: int,
        num_res: int,
        res_block: type[nn.Module],
        conv_block: type[nn.Module],
        height: int,
        width: int,
        length: int,
        dropout_rate: float,
        use_mod: bool = False,
        mod_noise_rate: float = 0.0,
        number_of_experts: int = 6,
        mod_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.use_mod = bool(use_mod)
        self.mod_noise_rate = float(mod_noise_rate)
        self.number_of_experts = int(number_of_experts)
        self.blocks = nn.ModuleList(
            [
                EBlock(base_channel, num_res, res_block=res_block),
                EBlock(base_channel * 2, num_res, res_block=res_block),
                EBlock(base_channel * 4, num_res, res_block=res_block),
                EBlock(base_channel * 8, num_res, res_block=res_block),
            ]
        )
        if self.use_mod:
            self.ups = nn.ModuleList(
                [
                    UPPosMod(
                        in_ch=base_channel * 2,
                        out_ch=base_channel,
                        H=height,
                        W=width,
                        T=length,
                        number_of_experts=self.number_of_experts,
                    ),
                    UPPosMod(
                        in_ch=base_channel * 4,
                        out_ch=base_channel * 2,
                        H=height // 2,
                        W=width // 2,
                        T=length,
                        number_of_experts=self.number_of_experts,
                    ),
                    UPPosMod(
                        in_ch=base_channel * 8,
                        out_ch=base_channel * 4,
                        H=height // 4,
                        W=width // 4,
                        T=length,
                        number_of_experts=self.number_of_experts,
                    ),
                ]
            )
            self.modulator = _ExpertGateMLP(
                output_dim=3 * self.number_of_experts,
                hidden_dim=int(mod_hidden_dim),
            )
        else:
            self.ups = nn.ModuleList(
                [
                    UPPos(in_ch=base_channel * 2, out_ch=base_channel, H=height, W=width, T=length),
                    UPPos(in_ch=base_channel * 4, out_ch=base_channel * 2, H=height // 2, W=width // 2, T=length),
                    UPPos(in_ch=base_channel * 8, out_ch=base_channel * 4, H=height // 4, W=width // 4, T=length),
                ]
            )
            self.modulator = None
        self.output = conv_block(base_channel, length, kernel_size=1, relu=False, stride=1, groups=4)

    def forward(self, x0: torch.Tensor, x2: torch.Tensor, x4: torch.Tensor, x8: torch.Tensor) -> torch.Tensor:
        gates = None
        if self.use_mod and self.modulator is not None:
            w = self.modulator(x8)
            noise = 2.0 * torch.rand_like(w) - 1.0
            g = w + self.mod_noise_rate * noise
            gates = torch.chunk(g, 3, dim=-1)

        res1 = self.dropout(self.blocks[3](x8))
        res1 = res1 + torch.zeros_like(x8, requires_grad=True)
        res1 = self.ups[2](res1, gates[2]) if gates is not None else self.ups[2](res1)

        h4 = x4 + res1
        h4 = h4 + torch.zeros_like(h4, requires_grad=True)
        res2 = self.dropout(self.blocks[2](h4))
        res2 = self.ups[1](res2, gates[1]) if gates is not None else self.ups[1](res2)

        h2 = res2
        h2 = h2 + torch.zeros_like(h2, requires_grad=True)
        res3 = self.dropout(self.blocks[1](h2))
        res3 = self.ups[0](res3, gates[0]) if gates is not None else self.ups[0](res3)

        h0 = res3
        h0 = h0 + torch.zeros_like(h0, requires_grad=True)
        out = self.dropout(self.blocks[0](h0))
        out = out + torch.zeros_like(out, requires_grad=True)
        return self.output(out)


class P2IUncertaintyGenerator(BaseNetwork):
    """Shared P2I-GAN encoder with separate mean/log-variance decoder branches."""

    def __init__(
        self,
        config,
        length: int = 16,
        num_res: int = 4,
        inference: bool = False,
        init_weights: bool = True,
    ) -> None:
        super().__init__()

        data_cfg = config.get("data_loader") or config["data"]["train"]
        model_cfg = config.get("model", {})
        self.keep = data_cfg.get("mask", {}).get("keep", 0)
        self.H = data_cfg["h"]
        self.W = data_cfg["w"]
        self.length = data_cfg.get("sample_length", length)
        self.output_activation = model_cfg.get("output_activation", "sigmoid").lower()
        self.dropout_rate = float(model_cfg.get("mc_dropout_rate", 0.10))
        self.use_mod = bool(model_cfg.get("mod", False))
        self.mod_noise_rate = float(model_cfg.get("mod_noise_rate", 0.0))
        self.number_of_experts = int(model_cfg.get("number_of_experts", 6))
        self.mod_hidden_dim = int(model_cfg.get("mod_hidden_dim", max(128, int(model_cfg.get("base_channels", 64)) * 2)))
        loss_cfg = config.get("loss", {})
        self.logvar_clamp_min = loss_cfg.get("logvar_clamp_min")
        self.logvar_clamp_max = loss_cfg.get("logvar_clamp_max")

        if not inference:
            res_block = ResBlock_do
            conv_block = BasicConv_do
        else:
            res_block = ResBlock_do_eval
            conv_block = BasicConv_do_eval

        self.input = InputBlock(depth=2, k=4, rho=2.0, tau=0.05, chunk=16384)
        self.base_channel = int(model_cfg.get("base_channels", 64))
        self.conv_in = conv_block(self.length, self.base_channel, kernel_size=3, relu=False, stride=1, groups=4)
        self.downsample = DownsampleDuplicateChannels(length=self.length)

        branch_kwargs = {
            "base_channel": self.base_channel,
            "num_res": num_res,
            "res_block": res_block,
            "conv_block": conv_block,
            "height": self.H,
            "width": self.W,
            "length": self.length,
            "dropout_rate": self.dropout_rate,
        }
        self.mean_decoder = _BranchDecoder(
            **branch_kwargs,
            use_mod=self.use_mod,
            mod_noise_rate=self.mod_noise_rate,
            number_of_experts=self.number_of_experts,
            mod_hidden_dim=self.mod_hidden_dim,
        )
        self.logvar_decoder = _BranchDecoder(**branch_kwargs)
        self.shared_dropout = nn.Dropout2d(p=self.dropout_rate)

        if init_weights:
            self.init_weights()

    def _activate_mean(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "sigmoid":
            return torch.sigmoid(prediction)
        if self.output_activation == "tanh":
            return torch.tanh(prediction)
        if self.output_activation == "identity":
            return prediction
        raise ValueError(f"Unsupported output activation: {self.output_activation}")

    def forward(self, masked_frames: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = masked_frames.size()
        masked_frames = masked_frames.view(b, c * t, h, w)
        masks = masks.view(b, c * t, h, w)

        x = self.input(masked_frames, masks).float()
        x0 = self.conv_in(x) + x.repeat_interleave(4, dim=1)
        x0 = self.shared_dropout(x0)
        x0 = x0.view(b, t * c * 4, h, w)
        x2 = self.downsample(x0)
        x4 = self.downsample(x2)
        x8 = self.downsample(x4)

        use_amp = masked_frames.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            x0 = x0.float()
            x2 = x2.float()
            x4 = x4.float()
            x8 = x8.float()

        mean = self.mean_decoder(x0, x2, x4, x8)
        logvar = self.logvar_decoder(x0, x2, x4, x8)
        if self.logvar_clamp_min is not None or self.logvar_clamp_max is not None:
            min_value = -float("inf") if self.logvar_clamp_min is None else float(self.logvar_clamp_min)
            max_value = float("inf") if self.logvar_clamp_max is None else float(self.logvar_clamp_max)
            logvar = logvar.clamp(min=min_value, max=max_value)

        mean = self._activate_mean(mean).view(b, t, c, h, w)
        logvar = logvar.view(b, t, c, h, w)
        return torch.stack([mean, logvar], dim=-1)
