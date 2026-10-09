"""EXP-2: calibration diagnostics and spatial uncertainty case studies."""

from __future__ import annotations

import os
from typing import Dict, Mapping, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import convert_rain, ensure_dir, evaluation_mask
from .distributions import (
    deterministic_subsample,
    mixture_intervals,
    mixture_pit,
    selected_components,
)


def fit_std_scale(store: Mapping[str, object], cfg) -> float:
    """Fit one variance scale on an independent calibration store."""
    total, count = 0.0, 0
    for event in store["events"].values():
        selected = evaluation_mask(event["mask"], cfg.evaluation_region)
        residual = event["target"][selected] - event["mean"][selected]
        std = np.maximum(event["predictive_std"][selected], 1e-6)
        ratio2 = (residual / std) ** 2
        valid = np.isfinite(ratio2)
        total += float(np.sum(ratio2[valid]))
        count += int(valid.sum())
    return float(np.sqrt(total / max(count, 1)))


def _diagnostics(store: Mapping[str, object], cfg, scale: float = 1.0) -> Dict[str, object]:
    levels = np.asarray(cfg.coverage_levels, dtype=np.float64)
    covered = np.zeros(levels.size, dtype=np.float64)
    widths = np.zeros(levels.size, dtype=np.float64)
    width_count = 0
    count = 0
    rank_counts = np.zeros(int(cfg.pit_bins), dtype=np.int64)
    scatter_error, scatter_uncertainty = [], []
    event_mae: Dict[str, float] = {}
    rng = np.random.default_rng(cfg.bootstrap_seed)

    for event_name, event in store["events"].items():
        selector = evaluation_mask(event["mask"], cfg.evaluation_region)
        truth_encoded, means, stds = selected_components(event, selector, scale=scale)
        y = convert_rain(truth_encoded, cfg.encoded_to_mmhr, cfg.encoding_divisor)
        center_encoded = means.mean(axis=0)
        mu = convert_rain(center_encoded, cfg.encoded_to_mmhr, cfg.encoding_divisor)
        if y.size == 0:
            continue
        event_mae[event_name] = float(np.mean(np.abs(mu - y)))
        count += y.size
        pit = mixture_pit(truth_encoded, means, stds)
        for index, level in enumerate(levels):
            tail = 1.0 - level
            covered[index] += np.sum((pit >= tail / 2.0) & (pit <= 1.0 - tail / 2.0))
        sampled_y, sampled_means, sampled_stds = deterministic_subsample(
            truth_encoded, means, stds, cfg.distribution_score_max_points,
            cfg.bootstrap_seed,
        )
        sampled_intervals = mixture_intervals(
            sampled_y, sampled_means, sampled_stds, levels,
            lambda value: convert_rain(value, cfg.encoded_to_mmhr, cfg.encoding_divisor),
        )
        width_count += sampled_y.size
        for index, level in enumerate(levels):
            widths[index] += sampled_intervals[f"{level:.2f}"]["width"] * sampled_y.size

        rank_counts += np.histogram(pit, bins=int(cfg.pit_bins), range=(0.0, 1.0))[0]

        predictive_std_encoded = np.sqrt(np.mean(stds ** 2 + means ** 2, axis=0) - center_encoded ** 2)
        uncertainty = convert_rain(
            center_encoded + predictive_std_encoded,
            cfg.encoded_to_mmhr, cfg.encoding_divisor,
        ) - mu
        absolute_error = np.abs(mu - y)
        remaining = max(0, cfg.max_scatter_points - sum(x.size for x in scatter_error))
        if remaining:
            chosen = rng.choice(y.size, size=min(remaining, y.size), replace=False)
            scatter_error.append(absolute_error[chosen])
            scatter_uncertainty.append(uncertainty[chosen])

    empirical = covered / max(count, 1)
    # Widths are evaluated on a deterministic subset because mixture quantiles
    # require numerical inversion; coverage above uses every valid pixel via PIT.
    width = widths / max(width_count, 1)
    errors = np.concatenate(scatter_error) if scatter_error else np.empty(0)
    uncertainty = np.concatenate(scatter_uncertainty) if scatter_uncertainty else np.empty(0)
    correlation = float("nan")
    if errors.size > 1:
        # Spearman without a SciPy dependency; ties are rare for continuous fields.
        error_rank = np.argsort(np.argsort(errors))
        uncertainty_rank = np.argsort(np.argsort(uncertainty))
        correlation = float(np.corrcoef(error_rank, uncertainty_rank)[0, 1])
    return {
        "nominal": levels,
        "empirical": empirical,
        "width": width,
        "calibration_error": float(np.mean(np.abs(empirical - levels))),
        "rank_counts": rank_counts,
        "scatter_error": errors,
        "scatter_uncertainty": uncertainty,
        "spearman": correlation,
        "event_mae": event_mae,
    }


def _plot_reliability(all_diagnostics: Mapping[str, Mapping[str, object]], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.0), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "--", color="0.55", label="ideal")
    for label, values in all_diagnostics.items():
        ax.plot(values["nominal"], values["empirical"], marker="o", label=label)
    ax.set(xlabel="Nominal coverage", ylabel="Empirical coverage", xlim=(0.45, 1.0), ylim=(0.45, 1.0))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, "reliability.pdf"), dpi=200)
    plt.close(fig)


def _plot_rank_hist(all_diagnostics: Mapping[str, Mapping[str, object]], out_dir: str) -> None:
    count = len(all_diagnostics)
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 3.8), squeeze=False, constrained_layout=True)
    for ax, (label, values) in zip(axes[0], all_diagnostics.items()):
        ranks = values["rank_counts"]
        ax.bar(np.arange(ranks.size), ranks / max(ranks.sum(), 1), color="#087f7b")
        ax.axhline(1.0 / max(ranks.size, 1), color="0.4", linestyle="--")
        ax.set(title=label, xlabel="Gaussian-mixture PIT bin", ylabel="Frequency")
    fig.savefig(os.path.join(out_dir, "rank_histogram.pdf"), dpi=200)
    plt.close(fig)


def _plot_uncertainty_error(all_diagnostics: Mapping[str, Mapping[str, object]], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.6), constrained_layout=True)
    for label, values in all_diagnostics.items():
        uncertainty = values["scatter_uncertainty"]
        error = values["scatter_error"]
        if uncertainty.size == 0:
            continue
        edges = np.quantile(uncertainty, np.linspace(0, 1, 11))
        x, y = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            selected = (uncertainty >= low) & (uncertainty <= high)
            if np.any(selected):
                x.append(float(np.mean(uncertainty[selected])))
                y.append(float(np.mean(error[selected])))
        ax.plot(x, y, marker="o", label=f"{label} (rho={values['spearman']:.2f})")
    ax.set(xlabel="Predictive spread (mm/h)", ylabel="Mean absolute error (mm/h)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, "uncertainty_vs_error.pdf"), dpi=200)
    plt.close(fig)


def _case_events(diagnostics: Mapping[str, object], requested: Tuple[str, ...]) -> Tuple[str, ...]:
    if requested:
        return requested
    ordered = sorted(diagnostics["event_mae"], key=diagnostics["event_mae"].get)
    if not ordered:
        return ()
    return tuple(dict.fromkeys((ordered[0], ordered[len(ordered) // 2], ordered[-1])))


def _plot_cases(method: str, store: Mapping[str, object], diagnostics: Mapping[str, object], cfg, out_dir: str) -> None:
    for event_name in _case_events(diagnostics, cfg.case_study_events):
        event = store["events"].get(event_name)
        if event is None:
            continue
        truth = convert_rain(event["target"], cfg.encoded_to_mmhr, cfg.encoding_divisor)
        mean = convert_rain(event["mean"], cfg.encoded_to_mmhr, cfg.encoding_divisor)
        masked = convert_rain(event.get("masked_input", event["target"]), cfg.encoded_to_mmhr, cfg.encoding_divisor)
        uncertainty = convert_rain(
            event["mean"] + event["predictive_std"],
            cfg.encoded_to_mmhr, cfg.encoding_divisor,
        ) - mean
        frame = cfg.case_study_frame
        if frame is None:
            frame = int(np.argmax(np.mean(truth, axis=(1, 2))))
        frame = min(max(int(frame), 0), truth.shape[0] - 1)
        fields = (masked[frame], truth[frame], mean[frame], np.abs(mean[frame] - truth[frame]), uncertainty[frame])
        titles = ("Sparse input", "Target", "Predictive mean", "Absolute error", "Predictive spread")
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), constrained_layout=True)
        rain_max = max(float(np.nanpercentile(truth[frame], 99)), 1.0)
        uncertainty_max = max(float(np.nanpercentile(uncertainty[frame], 99)), 1e-3)
        for index, (ax, field, title) in enumerate(zip(axes, fields, titles)):
            vmax = rain_max if index < 4 else uncertainty_max
            image = ax.imshow(field, vmin=0, vmax=vmax, cmap="turbo" if index < 3 else "magma")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
        safe_method = method.replace("/", "_").replace(" ", "_")
        fig.suptitle(f"{method} · {event_name} · frame {frame}")
        fig.savefig(os.path.join(out_dir, f"case_{safe_method}_{event_name}.png"), dpi=180)
        plt.close(fig)


def run_exp2(
    stores: Mapping[str, Mapping[str, object]],
    cfg,
    out_dir: str,
    calibration_stores: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Dict[str, object]:
    ensure_dir(out_dir)
    calibration_stores = calibration_stores or {}
    plotted: Dict[str, Mapping[str, object]] = {}
    serializable: Dict[str, object] = {}
    for method, store in stores.items():
        raw = _diagnostics(store, cfg)
        plotted[f"{method} raw"] = raw
        serializable[method] = {
            "raw": {k: v for k, v in raw.items() if not k.startswith("scatter_") and k != "event_mae"},
            "std_scale": None,
        }
        if method in calibration_stores:
            scale = fit_std_scale(calibration_stores[method], cfg)
            calibrated = _diagnostics(store, cfg, scale=scale)
            plotted[f"{method} calibrated"] = calibrated
            serializable[method]["std_scale"] = scale
            serializable[method]["calibrated"] = {
                k: v for k, v in calibrated.items() if not k.startswith("scatter_") and k != "event_mae"
            }
        _plot_cases(method, store, raw, cfg, out_dir)
    _plot_reliability(plotted, out_dir)
    _plot_rank_hist({k: v for k, v in plotted.items() if k.endswith(" raw")}, out_dir)
    _plot_uncertainty_error({k: v for k, v in plotted.items() if k.endswith(" raw")}, out_dir)
    return serializable
