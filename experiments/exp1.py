"""EXP-1: in-domain reconstruction and probabilistic scorecard."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import uniform_filter

from .io import convert_rain, evaluation_mask


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


def ensemble_crps(samples: np.ndarray, truth: np.ndarray, selector: np.ndarray) -> float:
    """Empirical CRPS using an O(E log E) sorted-sample identity."""
    selected = np.asarray(samples, dtype=np.float64)[:, selector]
    observed = np.asarray(truth, dtype=np.float64)[selector]
    finite = np.isfinite(observed) & np.all(np.isfinite(selected), axis=0)
    selected = selected[:, finite]
    observed = observed[finite]
    if observed.size == 0:
        return float("nan")
    first = np.mean(np.abs(selected - observed[None, :]), axis=0)
    ordered = np.sort(selected, axis=0)
    members = ordered.shape[0]
    coefficients = 2.0 * np.arange(1, members + 1) - members - 1.0
    half_pairwise = np.sum(coefficients[:, None] * ordered, axis=0) / (members**2)
    return float(np.mean(first - half_pairwise))


def energy_score(samples: np.ndarray, truth: np.ndarray, selector: np.ndarray,
                 max_pairs: int, max_points: int, seed: int) -> float:
    """Sampled multivariate Energy Score over the selected spatio-temporal field."""
    draws = np.asarray(samples, dtype=np.float64)[:, selector]
    observed = np.asarray(truth, dtype=np.float64)[selector]
    valid = np.isfinite(observed) & np.all(np.isfinite(draws), axis=0)
    draws, observed = draws[:, valid], observed[valid]
    if observed.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    if observed.size > max_points:
        chosen = rng.choice(observed.size, size=max_points, replace=False)
        draws, observed = draws[:, chosen], observed[chosen]
    normalizer = np.sqrt(observed.size)
    accuracy = np.mean([
        np.linalg.norm(member - observed) / normalizer for member in draws
    ])
    total_pairs = draws.shape[0] ** 2
    pair_count = min(max(1, int(max_pairs)), total_pairs)
    left = rng.integers(0, draws.shape[0], size=pair_count)
    right = rng.integers(0, draws.shape[0], size=pair_count)
    spread = np.mean([
        np.linalg.norm(draws[i] - draws[j]) / normalizer for i, j in zip(left, right)
    ])
    return float(accuracy - 0.5 * spread)


def interval_metrics(
    samples: np.ndarray,
    truth: np.ndarray,
    selector: np.ndarray,
    levels: Iterable[float],
) -> Dict[str, Dict[str, float]]:
    values = np.asarray(truth)[selector]
    draws = np.asarray(samples)[:, selector]
    valid = np.isfinite(values) & np.all(np.isfinite(draws), axis=0)
    values, draws = values[valid], draws[:, valid]
    result: Dict[str, Dict[str, float]] = {}
    for level in levels:
        tail = 1.0 - float(level)
        lower, upper = np.quantile(draws, [tail / 2.0, 1.0 - tail / 2.0], axis=0)
        covered = (values >= lower) & (values <= upper)
        width = upper - lower
        winkler = width.copy()
        winkler += (2.0 / tail) * (lower - values) * (values < lower)
        winkler += (2.0 / tail) * (values - upper) * (values > upper)
        result[f"{level:.2f}"] = {
            "coverage": float(np.mean(covered)),
            "width": float(np.mean(width)),
            "winkler": float(np.mean(winkler)),
        }
    return result


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
    if "ensemble" in event and event["ensemble"].shape[0] >= 2:
        ensemble = convert_rain(event["ensemble"], cfg.encoded_to_mmhr, cfg.encoding_divisor)
        metrics["CRPS"] = ensemble_crps(ensemble, truth, selector)
        metrics["Energy_Score"] = energy_score(
            ensemble, truth, selector, cfg.energy_score_pair_samples,
            cfg.energy_score_max_points, cfg.bootstrap_seed
        )
        metrics["intervals"] = interval_metrics(ensemble, truth, selector, cfg.coverage_levels)
        metrics["mean_predictive_std"] = float(np.mean(np.std(ensemble[:, selector], axis=0)))
        ensemble_spread = float(np.sqrt(np.mean(np.var(ensemble[:, selector], axis=0))))
        metrics["spread_skill_ratio"] = ensemble_spread / max(metrics["RMSE"], EPS)
        ensemble_encoded = np.asarray(event["ensemble"][:, selector], dtype=np.float64)
        metrics["ensemble_mean_consistency_RMSE_encoded"] = float(
            np.sqrt(np.mean((ensemble_encoded.mean(axis=0) - event["mean"][selector]) ** 2))
        )
        metrics["ensemble_boundary_fraction"] = float(
            np.mean((ensemble_encoded <= 0.0) | (ensemble_encoded >= 255.0))
        )
        if "member_mean" in event:
            metrics["raw_member_spread_encoded"] = float(
                np.mean(np.std(event["member_mean"][:, selector], axis=0))
            )
    else:
        metrics["CRPS"] = float("nan")
        metrics["Energy_Score"] = float("nan")
        metrics["intervals"] = {}
        metrics["mean_predictive_std"] = float("nan")
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
    keys = ("MAE", "RMSE", "bias", "SSIM", "CRPS", "Energy_Score",
            "Gaussian_NLL_encoded", "Aleatoric_NLL_encoded", "spread_skill_ratio",
            "intervals.0.90.coverage", "intervals.0.90.width",
            "ensemble_mean_consistency_RMSE_encoded")
    lines = ["method\t" + "\t".join(keys)]
    for method, payload in results.items():
        summary = payload["summary"]
        cells = [method] + [f"{summary.get(key, {}).get('mean', float('nan')):.6g}" for key in keys]
        lines.append("\t".join(cells))
    return lines
