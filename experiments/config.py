"""Typed, JSON-loadable configuration for the four-model evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class MethodConfig:
    path: str
    calibration_path: Optional[str] = None
    group: str = "in-domain"
    model_variant: Optional[str] = None
    uncertainty_kind: str = "unspecified"
    secondary_kind: Optional[str] = None
    loss_profile: str = "unspecified"
    expected_raw_members: Optional[int] = None
    enabled: bool = True


def _default_methods() -> Dict[str, MethodConfig]:
    root = "inference/four_models"
    return {
        "M1-joint": MethodConfig(
            path=f"{root}/p2igan_joint.zarr", model_variant="p2igan_joint",
            uncertainty_kind="aleatoric", secondary_kind="none",
            loss_profile="gaussian_nll+temporal_kl+uncertainty_smoothness+field_gan",
            expected_raw_members=1,
        ),
        "M2-decoder-ensemble": MethodConfig(
            path=f"{root}/p2igan_decoder_ensemble.zarr", model_variant="p2igan_decoder_ensemble",
            uncertainty_kind="between_decoder",
            loss_profile="poisson_member_nll+temporal_kl+uncertainty_smoothness+field_gan",
            secondary_kind="between_decoder",
            expected_raw_members=4,
        ),
        "M3-mc-dropout": MethodConfig(
            path=f"{root}/p2igan_mc_dropout.zarr", model_variant="p2igan_mc_dropout",
            uncertainty_kind="mc_dropout",
            loss_profile="member_nll+l2+temporal_kl+uncertainty_smoothness+field_gan",
            secondary_kind="mc_dropout",
            expected_raw_members=16,
        ),
        "M4-dropout-condition-comod": MethodConfig(
            path=f"{root}/p2igan_condition_comod.zarr", model_variant="p2igan_condition_comod",
            uncertainty_kind="mc_dropout_condition",
            loss_profile="member_nll+l2+temporal_kl+uncertainty_smoothness+condition_experts+field_gan",
            secondary_kind="mc_dropout_condition",
            expected_raw_members=16,
        ),
    }


@dataclass
class ExperimentConfig:
    experiment_name: str = "p2igan_four_model_uncertainty"
    description: str = "Controlled four-model uncertainty comparison."
    save_dir: str = "experiments/results"
    run_exp1: bool = True
    run_exp2: bool = True
    run_exp3: bool = True
    methods: Dict[str, MethodConfig] = field(default_factory=_default_methods)

    evaluation_region: str = "missing"
    crop_size: Optional[int] = 128
    encoded_to_mmhr: bool = True
    encoding_divisor: float = 3.0
    rain_thresholds_mmhr: Tuple[float, ...] = (0.5, 2.0, 4.0, 8.0, 16.0)
    coverage_levels: Tuple[float, ...] = (0.50, 0.70, 0.80, 0.90, 0.95)
    fss_scales: Tuple[int, ...] = (1, 2, 4, 8)

    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 2024
    bad_error_quantile: float = 0.90
    risk_coverages: Tuple[float, ...] = (
        0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
    )
    case_study_events: Tuple[str, ...] = ()
    case_study_frame: Optional[int] = None
    max_scatter_points: int = 50000

    # Evaluation cost / integrity controls.
    energy_score_pair_samples: int = 128
    energy_score_max_points: int = 16384
    require_aligned_events: bool = True
    alignment_check_events: int = 3
    require_raw_members: bool = True
    fail_on_variant_mismatch: bool = True


_TUPLE_FIELDS = {
    "rain_thresholds_mmhr", "coverage_levels", "fss_scales", "risk_coverages",
    "case_study_events",
}


def build_config(path: Optional[str] = None) -> ExperimentConfig:
    """Return defaults or merge a JSON experiment manifest over them."""
    cfg = ExperimentConfig()
    if path is None:
        return cfg
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    methods = payload.pop("methods", None)
    for key, value in payload.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown experiment config key: {key}")
        setattr(cfg, key, tuple(value) if key in _TUPLE_FIELDS else value)
    if methods is not None:
        cfg.methods = {
            name: MethodConfig(**method)
            for name, method in methods.items()
            if method.get("enabled", True)
        }
    if not cfg.methods:
        raise ValueError("Experiment config contains no enabled methods")
    return cfg
