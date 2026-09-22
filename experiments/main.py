"""Run the three uncertainty-aware evaluation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from .config import build_config
from .exp1 import run_exp1, scorecard_lines
from .exp2 import run_exp2
from .exp3 import run_exp3
from .io import ensure_dir, load_uncertainty_store, save_config_snapshot, save_json, save_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _load_stores(cfg):
    stores: Dict[str, object] = {}
    calibration: Dict[str, object] = {}
    for name, method in cfg.methods.items():
        path = _resolve(method.path)
        if not path.exists():
            raise FileNotFoundError(
                f"Inference store for {name!r} does not exist: {path}\n"
                "Create it first with scripts/infer.py --output <path> --overwrite."
            )
        print(f"[experiments] loading {name}: {path}")
        stores[name] = load_uncertainty_store(str(path), crop_size=cfg.crop_size)
        if method.calibration_path:
            calibration_path = _resolve(method.calibration_path)
            if not calibration_path.exists():
                raise FileNotFoundError(
                    f"Calibration store for {name!r} does not exist: {calibration_path}"
                )
            calibration[name] = load_uncertainty_store(
                str(calibration_path), crop_size=cfg.crop_size
            )
    return stores, calibration


def _preflight(stores, cfg):
    """Validate model identity, raw-member contract, and fair event alignment."""
    report = {}
    reference_name = next(iter(stores))
    reference_events = tuple(stores[reference_name]["events"])
    for name, store in stores.items():
        method = cfg.methods[name]
        attrs = store.get("attrs", {})
        events = tuple(store["events"])
        issues = []
        actual_variant = attrs.get("model_variant")
        if method.model_variant and actual_variant != method.model_variant:
            issues.append(f"model_variant={actual_variant!r}, expected {method.model_variant!r}")
        actual_secondary = attrs.get("secondary_kind")
        if method.secondary_kind and actual_secondary != method.secondary_kind:
            issues.append(f"secondary_kind={actual_secondary!r}, expected {method.secondary_kind!r}")
        if cfg.require_aligned_events and events != reference_events:
            issues.append("event keys/order differ from the reference method")
        first = store["events"][events[0]]
        raw_members = int(first["member_mean"].shape[0]) if "member_mean" in first else 0
        if cfg.require_raw_members and raw_members == 0:
            issues.append("member_mean is missing")
        if method.expected_raw_members is not None and raw_members != method.expected_raw_members:
            issues.append(f"raw_members={raw_members}, expected {method.expected_raw_members}")
        report[name] = {
            "model_variant": actual_variant,
            "uncertainty_kind": method.uncertainty_kind,
            "secondary_kind": actual_secondary,
            "loss_profile": method.loss_profile,
            "events": len(events),
            "raw_members": raw_members,
            "issues": issues,
        }
    if cfg.require_aligned_events:
        for event_name in reference_events[:max(0, cfg.alignment_check_events)]:
            reference = stores[reference_name]["events"][event_name]
            for name, store in stores.items():
                if name == reference_name or event_name not in store["events"]:
                    continue
                candidate = store["events"][event_name]
                if not np.array_equal(reference["mask"], candidate["mask"]):
                    report[name]["issues"].append(f"mask mismatch at {event_name}")
                if not np.allclose(reference["target"], candidate["target"], atol=1e-5):
                    report[name]["issues"].append(f"target mismatch at {event_name}")
    failures = {name: values["issues"] for name, values in report.items() if values["issues"]}
    if failures and cfg.fail_on_variant_mismatch:
        raise ValueError(f"Experiment preflight failed: {failures}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Run uncertainty evaluation experiments")
    parser.add_argument("--config", default="experiments/configs/four_models.json")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main(args=None) -> None:
    parsed = args or parse_args()
    cfg = build_config(parsed.config)
    stores, calibration_stores = _load_stores(cfg)
    output_root = _resolve(cfg.save_dir) / cfg.experiment_name
    ensure_dir(str(output_root))
    save_config_snapshot(str(output_root / "config.json"), cfg)
    preflight = _preflight(stores, cfg)
    save_json(str(output_root / "preflight.json"), preflight)
    print(f"[experiments] preflight passed for {len(stores)} methods")
    if parsed.check_only:
        return

    if cfg.run_exp1:
        output = output_root / "exp1_scorecard"
        ensure_dir(str(output))
        results = run_exp1(stores, cfg)
        save_json(str(output / "metrics.json"), results)
        save_text(str(output / "scorecard.tsv"), scorecard_lines(results))
        print(f"[EXP-1] scorecard saved to {output}")

    if cfg.run_exp2:
        output = output_root / "exp2_calibration"
        ensure_dir(str(output))
        results = run_exp2(
            stores, cfg, str(output), calibration_stores=calibration_stores
        )
        save_json(str(output / "diagnostics.json"), results)
        print(f"[EXP-2] calibration diagnostics saved to {output}")

    if cfg.run_exp3:
        output = output_root / "exp3_stress_utility"
        ensure_dir(str(output))
        results = run_exp3(stores, cfg, str(output))
        save_json(str(output / "metrics.json"), results)
        print(f"[EXP-3] stress/utility results saved to {output}")


if __name__ == "__main__":
    main()
