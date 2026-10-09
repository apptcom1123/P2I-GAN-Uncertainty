"""Model registry for the benchmark."""

from typing import Any, Dict

import torch.nn as nn

from .p2igan import P2IDiscriminator
from .p2igan_uncertainty import (
    P2IBNNJointGenerator,
    P2IDecoderEnsembleGenerator,
    P2IJointGenerator,
    P2IConditionCoModGenerator,
    P2IMCDropoutGenerator,
    UncertaintyOutput,
)


def build_generator(cfg: Dict[str, Any]) -> nn.Module:
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "")).lower()
    if model_name == "p2igan_joint":
        return P2IJointGenerator(cfg)
    if model_name == "p2igan_bnn_joint":
        return P2IBNNJointGenerator(cfg)
    if model_name == "p2igan_decoder_ensemble":
        return P2IDecoderEnsembleGenerator(cfg)
    if model_name == "p2igan_mc_dropout":
        return P2IMCDropoutGenerator(cfg)
    if model_name == "p2igan_condition_comod":
        return P2IConditionCoModGenerator(cfg)
    supported = (
        "p2igan_joint", "p2igan_bnn_joint", "p2igan_decoder_ensemble",
        "p2igan_mc_dropout", "p2igan_condition_comod",
    )
    raise ValueError(f"Unsupported model.name={model_name!r}; choose one of {supported}")


def build_discriminator(cfg: Dict[str, Any]) -> nn.Module:
    discriminator_cfg = cfg.get("discriminator", {})
    model_cfg = cfg.get("model", {})
    model_name = str(discriminator_cfg.get("name", model_cfg.get("name", ""))).lower()
    in_channels = discriminator_cfg.get("in_channels", model_cfg.get("in_channels", 1))
    init_weights = bool(discriminator_cfg.get("init_weights", True))

    if model_name in {
        "p2igan_joint", "p2igan_bnn_joint", "p2igan_decoder_ensemble",
        "p2igan_mc_dropout", "p2igan_condition_comod",
    }:
        data_cfg = cfg.get("data_loader") or cfg.get("data", {}).get("train", {})
        sample_length = int(discriminator_cfg.get("sample_length", data_cfg.get("sample_length", 16)))
        feature_channels = int(discriminator_cfg.get("feature_channels", in_channels))
        seq_channels = feature_channels * sample_length
        return P2IDiscriminator(
            in_channels=seq_channels,
            feature_channels=feature_channels,
            init_weights=init_weights,
        )
    supported = (
        "p2igan_joint", "p2igan_bnn_joint", "p2igan_decoder_ensemble",
        "p2igan_mc_dropout", "p2igan_condition_comod",
    )
    raise ValueError(f"Unsupported discriminator/model name={model_name!r}; choose one of {supported}")


__all__ = [
    "build_generator",
    "build_discriminator",
    "P2IBNNJointGenerator",
    "P2IJointGenerator",
    "P2IDecoderEnsembleGenerator",
    "P2IMCDropoutGenerator",
    "P2IConditionCoModGenerator",
    "UncertaintyOutput",
    "P2IDiscriminator",
]
