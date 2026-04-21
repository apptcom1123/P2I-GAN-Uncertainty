"""Model registry for the benchmark."""

from typing import Any, Dict

import torch.nn as nn

from .dk import DKGenerator
from .p2igan import P2IDiscriminator, P2IGenerator
from .p2igan_uncertainty import P2IUncertaintyGenerator
from .p2iganwopos import P2IDiscriminator as P2IDiscriminatorWoPos
from .p2iganwopos import P2IGenerator as P2IGeneratorWoPos
from .simple import SimpleDiscriminator, SimpleGenerator
from .stdk import STDKGenerator


def build_generator(cfg: Dict[str, Any]) -> nn.Module:
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "simple").lower()
    in_channels = model_cfg.get("in_channels", 1)
    out_channels = model_cfg.get("out_channels", in_channels)
    base_channels = model_cfg.get("base_channels", 64)

    if model_name == "p2igan":
        return P2IGenerator(cfg)
    if model_name == "p2igan_uncertainty":
        return P2IUncertaintyGenerator(cfg)
    if model_name == "p2iganwopos":
        return P2IGeneratorWoPos(cfg)
    if model_name == "dk":
        data_cfg = cfg.get("data_loader") or cfg.get("data", {}).get("train", {})
        sample_length = data_cfg.get("sample_length", 16)
        return DKGenerator(cfg, length=sample_length)
    if model_name == "stdk":
        data_cfg = cfg.get("data_loader") or cfg.get("data", {}).get("train", {})
        sample_length = data_cfg.get("sample_length", 16)
        return STDKGenerator(cfg, length=sample_length)

    return SimpleGenerator(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)


def build_discriminator(cfg: Dict[str, Any]) -> nn.Module:
    discriminator_cfg = cfg.get("discriminator", {})
    model_cfg = cfg.get("model", {})
    model_name = discriminator_cfg.get("name", model_cfg.get("name", "simple")).lower()
    in_channels = discriminator_cfg.get("in_channels", model_cfg.get("in_channels", 1))
    base_channels = discriminator_cfg.get("base_channels", model_cfg.get("base_channels", 64))
    init_weights = bool(discriminator_cfg.get("init_weights", True))

    if model_name == "p2igan":
        data_cfg = cfg.get("data_loader") or cfg.get("data", {}).get("train", {})
        sample_length = int(discriminator_cfg.get("sample_length", data_cfg.get("sample_length", 16)))
        seq_channels = in_channels * sample_length
        return P2IDiscriminator(in_channels=seq_channels, init_weights=init_weights)
    if model_name == "p2iganwopos":
        data_cfg = cfg.get("data_loader") or cfg.get("data", {}).get("train", {})
        sample_length = int(discriminator_cfg.get("sample_length", data_cfg.get("sample_length", 16)))
        seq_channels = in_channels * sample_length
        return P2IDiscriminatorWoPos(in_channels=seq_channels, init_weights=init_weights)

    return SimpleDiscriminator(in_channels=in_channels, base_channels=base_channels)


__all__ = [
    "build_generator",
    "build_discriminator",
    "SimpleGenerator",
    "SimpleDiscriminator",
    "P2IGenerator",
    "P2IUncertaintyGenerator",
    "P2IDiscriminator",
    "P2IGeneratorWoPos",
    "P2IDiscriminatorWoPos",
    "DKGenerator",
    "STDKGenerator",
]
