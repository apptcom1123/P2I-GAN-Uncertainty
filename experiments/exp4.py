"""EXP-4: reproducible event clips rendered as combined GIF animations."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .io import convert_rain, ensure_dir


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def _segment_indices(length: int, event_name: str, cfg) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    segment_length = int(cfg.visualization_segment_length)
    if segment_length < 1 or segment_length > length:
        raise ValueError(
            f"visualization_segment_length must be in [1, {length}], got {segment_length}"
        )
    max_start = length - segment_length
    requested = tuple(dict.fromkeys(int(x) for x in cfg.visualization_segment_starts))
    if requested:
        invalid = [start for start in requested if start < 0 or start > max_start]
        if invalid:
            raise ValueError(
                f"visualization segment starts {invalid} exceed valid range [0, {max_start}]"
            )
        starts = requested
    else:
        count = int(cfg.visualization_random_segments)
        if count < 1:
            raise ValueError("visualization_random_segments must be >= 1")
        digest = hashlib.sha256(
            f"{int(cfg.visualization_seed)}:{event_name}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        candidates = rng.permutation(max_start + 1).tolist()
        selected = []
        for start in candidates:
            if all(abs(start - other) >= segment_length for other in selected):
                selected.append(int(start))
                if len(selected) == count:
                    break
        if len(selected) < count:
            for start in candidates:
                if start not in selected:
                    selected.append(int(start))
                    if len(selected) == count:
                        break
        starts = tuple(selected)
    indices = tuple(
        frame for start in starts for frame in range(start, start + segment_length)
    )
    return starts, indices


def _robust_max(values: np.ndarray, percentile: float = 99.5, floor: float = 1e-3) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return max(float(np.percentile(finite, percentile)), floor) if finite.size else floor


def _convert_std(std: np.ndarray, mean: np.ndarray, cfg) -> np.ndarray:
    """Map encoded-space standard deviation to a local mm/h increment."""
    if not cfg.encoded_to_mmhr:
        return np.asarray(std, dtype=np.float64)
    center = convert_rain(mean, True, cfg.encoding_divisor)
    upper = convert_rain(np.asarray(mean) + np.asarray(std), True, cfg.encoding_divisor)
    return np.maximum(upper - center, 0.0)


def _overview_frame(
    *, method: str, event: str, frame: int, segment: int,
    forecast_fields: Sequence[np.ndarray], uncertainty_fields: Sequence[np.ndarray],
    ensemble_fields: Sequence[np.ndarray], ensemble_members: Sequence[int],
    rain_max: float, error_max: float, uncertainty_max: float,
) -> np.ndarray:
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 9.4), constrained_layout=True)
    rows = (
        (forecast_fields, ("Sparse input", "Target", "Predictive mean", "Absolute error"),
         ("turbo", "turbo", "turbo", "magma"),
         (rain_max, rain_max, rain_max, error_max)),
        (uncertainty_fields, ("Aleatoric std", "Secondary std", "Predictive std"),
         ("magma", "magma", "magma"),
         (uncertainty_max, uncertainty_max, uncertainty_max)),
        (ensemble_fields, tuple(f"Bayesian mean member {x}" for x in ensemble_members),
         tuple("turbo" for _ in ensemble_fields), tuple(rain_max for _ in ensemble_fields)),
    )
    for row_index, (fields, titles, cmaps, vmaxes) in enumerate(rows):
        for column, ax in enumerate(axes[row_index]):
            if column >= len(fields):
                ax.axis("off")
                continue
            image = ax.imshow(fields[column], vmin=0.0, vmax=vmaxes[column], cmap=cmaps[column])
            ax.set_title(titles[column], fontsize=9)
            ax.axis("off")
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
            colorbar.ax.tick_params(labelsize=7)
            colorbar.set_label("mm/h", fontsize=7)
        axes[row_index, 0].text(
            -0.16, 0.5, ("Forecast", "Uncertainty", "Ensemble")[row_index],
            rotation=90, va="center", ha="center",
            transform=axes[row_index, 0].transAxes, fontsize=10, fontweight="bold",
        )
    fig.suptitle(
        f"{method} · {event} · segment {segment} · frame {frame}", fontsize=12
    )
    fig.canvas.draw()
    rendered = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rendered


def _save_gif(path: Path, frames: Sequence[np.ndarray], duration_ms: int) -> None:
    if frames:
        images = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
        images[0].save(
            path, save_all=True, append_images=images[1:],
            duration=max(1, int(duration_ms)), loop=0, disposal=2,
        )


def _render_event(method: str, event_name: str, event: Mapping[str, np.ndarray], cfg,
                  output: Path) -> Dict[str, object]:
    truth = convert_rain(event["target"], cfg.encoded_to_mmhr, cfg.encoding_divisor)
    mean = convert_rain(event["mean"], cfg.encoded_to_mmhr, cfg.encoding_divisor)
    masked = convert_rain(
        event.get("masked_input", event["target"]), cfg.encoded_to_mmhr,
        cfg.encoding_divisor,
    )
    aleatoric = _convert_std(
        event.get("aleatoric_std", event["predictive_std"]), event["mean"], cfg
    )
    secondary = _convert_std(
        event.get("secondary_std", np.zeros_like(event["predictive_std"])),
        event["mean"], cfg,
    )
    predictive = _convert_std(event["predictive_std"], event["mean"], cfg)
    starts, indices = _segment_indices(truth.shape[0], event_name, cfg)
    rain_max = _robust_max(truth)
    error_max = _robust_max(np.abs(mean - truth), percentile=99.0)
    uncertainty_max = _robust_max(predictive, percentile=99.0)

    selected_members: Tuple[int, ...] = ()
    ensemble = None
    if "member_mean" in event and event["member_mean"].shape[0] > 0:
        member_count = event["member_mean"].shape[0]
        selected_members = tuple(dict.fromkeys(
            int(index) for index in cfg.visualization_member_indices
            if 0 <= int(index) < member_count
        ))
        if selected_members:
            ensemble = convert_rain(
                event["member_mean"][list(selected_members)], cfg.encoded_to_mmhr,
                cfg.encoding_divisor,
            )

    segment_length = int(cfg.visualization_segment_length)
    overview_frames = []
    for position, frame in enumerate(indices):
        overview_frames.append(_overview_frame(
            method=method, event=event_name, frame=frame,
            segment=position // segment_length + 1,
            forecast_fields=(
                masked[frame], truth[frame], mean[frame], np.abs(mean[frame] - truth[frame])
            ),
            uncertainty_fields=(aleatoric[frame], secondary[frame], predictive[frame]),
            ensemble_fields=tuple(ensemble[:, frame]) if ensemble is not None else (),
            ensemble_members=selected_members,
            rain_max=rain_max, error_max=error_max, uncertainty_max=uncertainty_max,
        ))
    overview_path = output / "overview.gif"
    _save_gif(overview_path, overview_frames, cfg.visualization_frame_duration_ms)
    return {
        "segment_starts": list(starts),
        "segment_length": segment_length,
        "frames": list(indices),
        "frame_duration_ms": int(cfg.visualization_frame_duration_ms),
        "member_indices": list(selected_members),
        "files": [overview_path.name],
        "rain_vmax_mmhr": rain_max,
        "error_vmax_mmhr": error_max,
        "uncertainty_vmax_mmhr": uncertainty_max,
    }


def run_exp4(stores: Mapping[str, Mapping[str, object]], cfg, out_dir: str) -> Dict[str, object]:
    """Render configured events and reproducible consecutive clips for every method."""
    ensure_dir(out_dir)
    requested_events = tuple(cfg.visualization_events)
    if not requested_events:
        raise ValueError("visualization_events must contain at least one event name")
    manifest: Dict[str, object] = {}
    for method, store in stores.items():
        manifest[method] = {}
        for event_name in requested_events:
            if event_name not in store["events"]:
                available = ", ".join(store["events"])
                raise ValueError(
                    f"EXP-4 event {event_name!r} is missing for {method!r}; available: {available}"
                )
            output = Path(out_dir) / _safe_name(method) / _safe_name(event_name)
            output.mkdir(parents=True, exist_ok=True)
            manifest[method][event_name] = _render_event(
                method, event_name, store["events"][event_name], cfg, output
            )
    return manifest
