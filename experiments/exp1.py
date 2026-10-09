"""EXP-1: in-domain reconstruction and probabilistic scorecard."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
from scipy.ndimage import uniform_filter

from .io import convert_rain, evaluation_mask
from .distributions import (
    deterministic_subsample,
    mixture_crps,
    mixture_intervals,
    mixture_nll,
    selected_components,
)


EPS = 1e-8


def categorical_metrics(pred: np.ndarray, truth: np.ndarray, threshold: float) -> Dict[str, float]:
    pred_bin = pred >= threshold
    truth_bin = truth >= threshold
    hits = np.logical_and(pred_bin, truth_bin).sum()
    misses = np.logical_and(~pred_bin, truth_bin).sum()
    false = np.logical_and(pred_bin, ~truth_bin).sum()
    correct = np.logical_and(~pred_bin, ~truth_bin).sum()
    pod = hits / (hits + misses + EPS)
    far = false / (hits + false + EPS)
    csi = hits / (hits + misses + false + EPS)
    hss = 2 * (hits * correct - misses * false) / (
        misses**2 + false**2 + 2 * hits * correct + (misses + false) * (hits + correct) + EPS
    )
    return {"POD": float(pod), "FAR": float(far), "CSI": float(csi), "HSS": float(hss)}


def fss(pred: np.ndarray, truth: np.ndarray, selector: np.ndarray, threshold: float, scale: int) -> float:
    pred_fraction = uniform_filter((pred >= threshold).astype(np.float32), size=(1, scale, scale))
    truth_fraction = uniform_filter((truth >= threshold).astype(np.float32), size=(1, scale, scale))
    numerator = np.mean((pred_fraction[selector] - truth_fraction[selector]) ** 2)
    denominator = np.mean(pred_fraction[selector] ** 2 + truth_fraction[selector] ** 2)
    return float(1.0 - numerator / (denominator + EPS))


def gaussian_nll(mean: np.ndarray, std: np.ndarray, truth: np.ndarray, selector: np.ndarray) -> float:
    mu, sigma, y = mean[selector], np.maximum(std[selector], 1e-6), truth[selector]
    valid = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(y)
    mu, sigma, y = mu[valid], sigma[valid], y[valid]
    return float(np.mean(0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * ((y - mu) / sigma) ** 2))


def _masked_ssim(pred: np.ndarray, truth: np.ndarray, selector: np.ndarray) -> float:
    scores = []
    for frame in range(pred.shape[0]):
        selected = selector[frame]
        x, y = pred[frame][selected], truth[frame][selected]
        if x.size < 2:
            continue
        mx, my = x.mean(), y.mean()
        vx, vy = np.mean((x - mx) ** 2), np.mean((y - my) ** 2)
        covariance = np.mean((x - mx) * (y - my))
        dynamic = max(float(np.nanmax(y) - np.nanmin(y)), 1.0)
        c1, c2 = (0.01 * dynamic) ** 2, (0.03 * dynamic) ** 2
        scores.append(((2 * mx * my + c1) * (2 * covariance + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2) + EPS))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_event(event: Mapping[str, np.ndarray], cfg) -> Dict[str, object]:
    selector = evaluation_mask(event["mask"], cfg.evaluation_region)
    truth = convert_rain(event["target"], cfg.encoded_to_mmhr, cfg.encoding_divisor)
    prediction = convert_rain(event["mean"], cfg.encoded_to_mmhr, cfg.encoding_divisor)
    truth_values, pred_values = truth[selector], prediction[selector]
    valid = np.isfinite(truth_values) & np.isfinite(pred_values)
    truth_values, pred_values = truth_values[valid], pred_values[valid]
    error = pred_values - truth_values
    metrics: Dict[str, object] = {
        "n_pixels": int(valid.sum()),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "SSIM": _masked_ssim(prediction, truth, selector),
        # NLL remains in the stored encoded domain because nonlinear rain-unit
        # conversion does not preserve a Gaussian density.
        "Gaussian_NLL_encoded": gaussian_nll(
            event["mean"], event["predictive_std"], event["target"], selector
        ),
        "Aleatoric_NLL_encoded": gaussian_nll(
            event["mean"], event.get("aleatoric_std", event["predictive_std"]),
            event["target"], selector
        ),
        "mean_aleatoric_std_encoded": float(np.mean(event.get("aleatoric_std", event["predictive_std"])[selector])),
        "mean_secondary_std_encoded": float(np.mean(event.get("secondary_std", event.get("epistemic_std", np.zeros_like(event["predictive_std"])))[selector])),
        "mean_secondary_variance_fraction": float(np.mean(
            event.get("secondary_std", np.zeros_like(event["predictive_std"]))[selector] ** 2
            / np.maximum(event["predictive_std"][selector] ** 2, EPS)
        )),
        "mean_out_of_range_fraction": float(
            np.mean((event["mean"][selector] < 0.0) | (event["mean"][selector] > 255.0))
        ),
    }
    for threshold in cfg.rain_thresholds_mmhr:
        metrics[f"CAT_{threshold:g}"] = categorical_metrics(pred_values, truth_values, threshold)
        for scale in cfg.fss_scales:
            metrics[f"FSS_{threshold:g}_{scale}x{scale}"] = fss(
                prediction, truth, selector, threshold, scale
            )
    if "member_mean" not in event or "member_logvar" not in event:
        raise ValueError("Analytic probabilistic scoring requires member_mean and member_logvar")
    mixture_truth, mixture_means, mixture_stds = selected_components(event, selector)
    metrics["Gaussian_Mixture_NLL_encoded"] = mixture_nll(
        mixture_truth, mixture_means, mixture_stds
    )
    score_truth, score_means, score_stds = deterministic_subsample(
        mixture_truth, mixture_means, mixture_stds,
        cfg.distribution_score_max_points, cfg.bootstrap_seed,
    )
    metrics["Gaussian_Mixture_CRPS_encoded"] = mixture_crps(
        score_truth, score_means, score_stds
    )
    metrics["distribution_score_points"] = int(score_truth.size)
    metrics["intervals"] = mixture_intervals(
        score_truth, score_means, score_stds, cfg.coverage_levels,
        lambda value: convert_rain(value, cfg.encoded_to_mmhr, cfg.encoding_divisor),
    )
    metrics["mean_predictive_std_encoded"] = float(
        np.mean(event["predictive_std"][selector])
    )
    predictive_spread_mmhr = convert_rain(
        event["mean"][selector] + event["predictive_std"][selector],
        cfg.encoded_to_mmhr, cfg.encoding_divisor,
    ) - convert_rain(
        event["mean"][selector], cfg.encoded_to_mmhr, cfg.encoding_divisor,
    )
    metrics["spread_skill_ratio"] = float(
        np.sqrt(np.mean(predictive_spread_mmhr ** 2)) / max(metrics["RMSE"], EPS)
    )
    metrics["raw_member_spread_encoded"] = float(
        np.mean(np.std(event["member_mean"][:, selector], axis=0))
    )
    return metrics


def _flatten_scalars(metrics: Mapping[str, object], prefix: str = "") -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            output.update(_flatten_scalars(value, name))
        elif isinstance(value, (int, float, np.number)) and key != "n_pixels":
            output[name] = float(value)
    return output


def _bootstrap_summary(
    per_event: Mapping[str, Mapping[str, object]], replicates: int, seed: int
) -> Dict[str, Dict[str, float]]:
    event_names = list(per_event)
    flattened = {name: _flatten_scalars(per_event[name]) for name in event_names}
    keys = sorted(set.intersection(*(set(values) for values in flattened.values()))) if flattened else []
    rng = np.random.default_rng(seed)
    result: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.asarray([flattened[name][key] for name in event_names], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        if values.size == 1 or replicates <= 0:
            low = high = float(values.mean())
        else:
            indices = rng.integers(0, values.size, size=(replicates, values.size))
            boot = values[indices].mean(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])
        result[key] = {"mean": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high)}
    return result


def run_exp1(stores: Mapping[str, Mapping[str, object]], cfg) -> Dict[str, object]:
    results: Dict[str, object] = {}
    for method, store in stores.items():
        per_event = {
            event_name: evaluate_event(event, cfg)
            for event_name, event in store["events"].items()
        }
        results[method] = {
            "store_attrs": store.get("attrs", {}),
            "per_event": per_event,
            "summary": _bootstrap_summary(
                per_event, cfg.bootstrap_replicates, cfg.bootstrap_seed
            ),
        }
    return results


def scorecard_lines(results: Mapping[str, object]) -> Sequence[str]:
    keys = (
        "MAE", "RMSE", "bias", "SSIM", "Gaussian_Mixture_CRPS_encoded",
        "Gaussian_Mixture_NLL_encoded", "Gaussian_NLL_encoded",
        "Aleatoric_NLL_encoded", "spread_skill_ratio",
        "intervals.0.90.coverage", "intervals.0.90.width",
    )
    lines = ["method\t" + "\t".join(keys)]
    for method, payload in results.items():
        summary = payload["summary"]
        cells = [method] + [f"{summary.get(key, {}).get('mean', float('nan')):.6g}" for key in keys]
        lines.append("\t".join(cells))
    return lines
