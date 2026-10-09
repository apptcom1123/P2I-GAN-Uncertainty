"""Small synthetic smoke tests for the uncertainty evaluation pipeline."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import torch
import zarr

from p2igan_bench.modules.monitoring import QUALITY_METRIC_KEYS, quality_metrics

from .config import MethodConfig, build_config
from .exp1 import run_exp1
from .exp2 import run_exp2
from .exp3 import run_exp3
from .io import load_uncertainty_store
from .main import _preflight


def _synthetic_store(seed: int = 4):
    rng = np.random.default_rng(seed)
    events = {}
    for event_index in range(3):
        target = rng.uniform(0, 90, size=(4, 8, 8))
        mean = np.clip(target + rng.normal(0, 5, size=target.shape), 0, 255)
        predictive_std = np.full_like(target, 7.0)
        ensemble = np.clip(
            mean[None] + rng.normal(0, 7, size=(12, *target.shape)), 0, 255
        )
        mask = np.zeros_like(target)
        mask[:, ::3, ::3] = 1.0
        events[f"event_{event_index + 1:02d}"] = {
            "target": target,
            "masked_input": target * mask,
            "mask": mask,
            "mean": mean,
            "aleatoric_std": predictive_std * 0.8,
            "epistemic_std": predictive_std * 0.6,
            "predictive_std": predictive_std,
            "ensemble": ensemble,
        }
    return {"attrs": {"value_scale": 255.0}, "events": events}


class EvaluationSmokeTest(unittest.TestCase):
    def setUp(self):
        self.cfg = build_config()
        self.cfg.crop_size = None
        self.cfg.bootstrap_replicates = 20
        self.cfg.max_scatter_points = 500
        self.cfg.methods = {}
        self.stores = {"synthetic": _synthetic_store()}

    def test_exp1_scorecard(self):
        result = run_exp1(self.stores, self.cfg)
        summary = result["synthetic"]["summary"]
        self.assertIn("CRPS", summary)
        self.assertIn("intervals.0.90.coverage", summary)
        self.assertGreater(summary["RMSE"]["mean"], 0.0)

    def test_plotting_and_utility(self):
        with tempfile.TemporaryDirectory() as output:
            calibration = {"synthetic": _synthetic_store(seed=5)}
            exp2 = run_exp2(self.stores, self.cfg, output, calibration)
            exp3 = run_exp3(self.stores, self.cfg, output)
        self.assertIsNotNone(exp2["synthetic"]["std_scale"])
        self.assertIn("risk_coverage", exp3["synthetic"]["utility"])

    def test_inference_zarr_loader(self):
        source = _synthetic_store()["events"]["event_01"]
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/inference.zarr"
            root = zarr.open_group(path, mode="w")
            root.attrs["value_scale"] = 255.0
            event = root.create_group("event_01")
            for key, value in source.items():
                event.create_dataset(key, data=value, shape=value.shape)
            loaded = load_uncertainty_store(path)
            self.assertEqual(loaded["events"]["event_01"]["mean"].shape, (4, 8, 8))
            self.assertEqual(loaded["events"]["event_01"]["ensemble"].shape, (12, 4, 8, 8))

    def test_four_model_preflight_contract(self):
        variants = {
            "M1": ("p2igan_joint", 1, "aleatoric"),
            "M2": ("p2igan_decoder_ensemble", 4, "between_decoder"),
            "M3": ("p2igan_mc_dropout", 16, "mc_dropout"),
            "M4": ("p2igan_condition_comod", 16, "mc_dropout_condition"),
        }
        stores, methods = {}, {}
        for name, (variant, members, kind) in variants.items():
            store = _synthetic_store()
            for event in store["events"].values():
                event["secondary_std"] = event["epistemic_std"]
                event["member_mean"] = np.repeat(event["mean"][None], members, axis=0)
                event["member_logvar"] = np.zeros_like(event["member_mean"])
            store["attrs"]["model_variant"] = variant
            store["attrs"]["secondary_kind"] = "none" if name == "M1" else kind
            stores[name] = store
            methods[name] = MethodConfig(
                path="unused", model_variant=variant, uncertainty_kind=kind,
                secondary_kind="none" if name == "M1" else kind,
                expected_raw_members=members,
            )
        self.cfg.methods = methods
        report = _preflight(stores, self.cfg)
        self.assertTrue(all(not item["issues"] for item in report.values()))

    def test_training_monitoring_metrics_are_finite(self):
        target = torch.linspace(0.0, 0.8, 2 * 4 * 8 * 8).reshape(2, 4, 1, 8, 8)
        prediction = (target + 0.05).clamp(0.0, 1.0)
        predictive_var = torch.full_like(target, 0.01)
        observation_mask = torch.zeros_like(target)
        observation_mask[..., ::3, ::3] = 1.0
        metrics = quality_metrics(target, prediction, predictive_var, observation_mask)
        self.assertEqual(set(metrics), set(QUALITY_METRIC_KEYS))
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        self.assertGreater(float(metrics["coverage_90"]), 0.0)


if __name__ == "__main__":
    unittest.main()
