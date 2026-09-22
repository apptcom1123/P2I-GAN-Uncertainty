"""EXP-3: heavy-rain stratification, UQ usefulness, and stress comparisons."""

from __future__ import annotations

import os
from typing import Dict, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import convert_rain, ensure_dir, evaluation_mask


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _ranking_metrics(score: np.ndarray, positive: np.ndarray) -> Dict[str, float]:
    positive = positive.astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return {"AUROC_bad_error": float("nan"), "AUPRC_bad_error": float("nan")}
    order = np.argsort(-score, kind="mergesort")
    labels = positive[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(~labels)
    tpr = np.concatenate(([0.0], tp / n_pos, [1.0]))
    fpr = np.concatenate(([0.0], fp / n_neg, [1.0]))
    auroc = float(np.trapz(tpr, fpr))
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    auprc = float(np.sum((recall - np.concatenate(([0.0], recall[:-1]))) * precision))
    return {"AUROC_bad_error": auroc, "AUPRC_bad_error": auprc}


def _risk_curves(error: np.ndarray, uncertainty: np.ndarray, coverages) -> Dict[str, list]:
    coverages = np.asarray(coverages, dtype=np.float64)
    uq_order = np.argsort(uncertainty)
    oracle_order = np.argsort(error)
    uq_risk, oracle_risk = [], []
    for coverage in coverages:
        count = max(1, int(np.ceil(error.size * coverage)))
        uq_risk.append(float(np.mean(error[uq_order[:count]])))
        oracle_risk.append(float(np.mean(error[oracle_order[:count]])))
    random_risk = [float(np.mean(error))] * coverages.size
    return {
        "coverage": coverages.tolist(),
        "uq_risk": uq_risk,
        "oracle_risk": oracle_risk,
        "random_risk": random_risk,
        "AURC": float(np.trapz(uq_risk, coverages)),
        "excess_AURC_vs_oracle": float(np.trapz(np.asarray(uq_risk) - np.asarray(oracle_risk), coverages)),
    }


def _evaluate_method(store: Mapping[str, object], cfg) -> Dict[str, object]:
    thresholds = (0.0,) + tuple(cfg.rain_thresholds_mmhr)
    sums = {threshold: {"absolute": 0.0, "squared": 0.0, "count": 0} for threshold in thresholds}
    errors, predictive_u, aleatoric_u, secondary_u = [], [], [], []
    rng = np.random.default_rng(cfg.bootstrap_seed)
    remaining = cfg.max_scatter_points

    for event in store["events"].values():
        selector = evaluation_mask(event["mask"], cfg.evaluation_region)
        truth = convert_rain(event["target"], cfg.encoded_to_mmhr, cfg.encoding_divisor)
        mean = convert_rain(event["mean"], cfg.encoded_to_mmhr, cfg.encoding_divisor)
        y, mu = truth[selector], mean[selector]
        valid = np.isfinite(y) & np.isfinite(mu)
        y, mu = y[valid], mu[valid]
        absolute = np.abs(mu - y)
        for threshold in thresholds:
            chosen = y >= threshold
            sums[threshold]["absolute"] += float(np.sum(absolute[chosen]))
            sums[threshold]["squared"] += float(np.sum((mu[chosen] - y[chosen]) ** 2))
            sums[threshold]["count"] += int(chosen.sum())

        if remaining <= 0:
            continue
        if "ensemble" in event and event["ensemble"].shape[0] >= 2:
            ensemble = convert_rain(event["ensemble"], cfg.encoded_to_mmhr, cfg.encoding_divisor)
            total_u = np.std(ensemble[:, selector], axis=0)[valid]
        else:
            total_u = event["predictive_std"][selector][valid]
        au = event.get("aleatoric_std", event["predictive_std"])[selector][valid]
        eu = event.get(
            "secondary_std", event.get("epistemic_std", np.zeros_like(event["predictive_std"]))
        )[selector][valid]
        sample_count = min(remaining, absolute.size)
        indices = rng.choice(absolute.size, size=sample_count, replace=False)
        errors.append(absolute[indices])
        predictive_u.append(total_u[indices])
        aleatoric_u.append(au[indices])
        secondary_u.append(eu[indices])
        remaining -= sample_count

    error = np.concatenate(errors) if errors else np.empty(0)
    total_u = np.concatenate(predictive_u) if predictive_u else np.empty(0)
    au = np.concatenate(aleatoric_u) if aleatoric_u else np.empty(0)
    eu = np.concatenate(secondary_u) if secondary_u else np.empty(0)
    strata: Dict[str, Dict[str, float]] = {}
    for threshold, values in sums.items():
        count = max(values["count"], 1)
        strata[f">={threshold:g}_mmhr"] = {
            "n_pixels": int(values["count"]),
            "MAE": values["absolute"] / count,
            "RMSE": float(np.sqrt(values["squared"] / count)),
        }
    if error.size == 0:
        return {"rain_strata": strata, "utility": {}}
    bad_threshold = float(np.quantile(error, cfg.bad_error_quantile))
    utility: Dict[str, object] = {
        "sample_size": int(error.size),
        "bad_error_threshold": bad_threshold,
        "spearman_predictive": _rank_correlation(total_u, error),
        "spearman_aleatoric_encoded": _rank_correlation(au, error),
        "spearman_secondary_encoded": _rank_correlation(eu, error),
    }
    utility.update(_ranking_metrics(total_u, error >= bad_threshold))
    utility["risk_coverage"] = _risk_curves(error, total_u, cfg.risk_coverages)
    return {"rain_strata": strata, "utility": utility}


def _plot_risk(results: Mapping[str, object], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)
    for method, payload in results.items():
        curve = payload["utility"].get("risk_coverage")
        if not curve:
            continue
        ax.plot(curve["coverage"], curve["uq_risk"], marker="o", label=f"{method} UQ")
        ax.plot(curve["coverage"], curve["oracle_risk"], linestyle=":", alpha=0.7, label=f"{method} oracle")
    ax.set(xlabel="Retained coverage", ylabel="Selective MAE (mm/h)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, "risk_coverage.pdf"), dpi=200)
    plt.close(fig)


def _plot_strata(results: Mapping[str, object], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    for method, payload in results.items():
        labels = list(payload["rain_strata"])
        values = [payload["rain_strata"][key]["RMSE"] for key in labels]
        ax.plot(range(len(labels)), values, marker="o", label=method)
    if results:
        labels = list(next(iter(results.values()))["rain_strata"])
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set(xlabel="Truth intensity stratum", ylabel="RMSE (mm/h)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, "heavy_rain_strata.pdf"), dpi=200)
    plt.close(fig)


def _plot_stress(results: Mapping[str, object], groups: Mapping[str, str], out_dir: str) -> None:
    labels = list(results)
    rmse = [results[name]["rain_strata"][">=0_mmhr"]["RMSE"] for name in labels]
    correlation = [results[name]["utility"].get("spearman_predictive", np.nan) for name in labels]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = ["#087f7b" if groups.get(name) == "in-domain" else "#d85f32" for name in labels]
    axes[0].bar(labels, rmse, color=colors)
    axes[0].set_ylabel("RMSE (mm/h)")
    axes[1].bar(labels, correlation, color=colors)
    axes[1].set_ylabel("Spearman(U, |error|)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(os.path.join(out_dir, "scenario_comparison.pdf"), dpi=200)
    plt.close(fig)


def run_exp3(stores: Mapping[str, Mapping[str, object]], cfg, out_dir: str) -> Dict[str, object]:
    ensure_dir(out_dir)
    results = {method: _evaluate_method(store, cfg) for method, store in stores.items()}
    groups = {name: cfg.methods[name].group for name in stores if name in cfg.methods}
    _plot_risk(results, out_dir)
    _plot_strata(results, out_dir)
    _plot_stress(results, groups, out_dir)
    return results
