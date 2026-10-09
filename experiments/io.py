"""I/O helpers for the uncertainty Zarr format exported by scripts/infer.py."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from collections.abc import Iterator, Mapping
from typing import Dict, Iterable, Optional

import numpy as np


FIELD_KEYS = (
    "target", "masked_input", "mask", "mean", "aleatoric_std",
    "secondary_std", "epistemic_std", "style_std", "proxy_secondary_std",
    "predictive_std", "member_mean", "member_logvar",
)
MEMBER_KEYS = {"member_mean", "member_logvar"}


def ensure_thw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got shape {arr.shape}")
    return arr


def ensure_ethw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 6 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 5 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 5 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 4:
        raise ValueError(f"Expected ensemble [E,T,H,W], got shape {arr.shape}")
    return arr


def crop_last2(arr: np.ndarray, size: Optional[int]) -> np.ndarray:
    arr = np.asarray(arr)
    if size is None:
        return arr
    h, w = arr.shape[-2:]
    if size > min(h, w):
        raise ValueError(f"crop size {size} exceeds input {h}x{w}")
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[..., top:top + size, left:left + size]


def _read_cropped(array, size: Optional[int]) -> np.ndarray:
    if size is None:
        return np.asarray(array)
    h, w = array.shape[-2:]
    if size > min(h, w):
        raise ValueError(f"crop size {size} exceeds input {h}x{w}")
    top = (h - size) // 2
    left = (w - size) // 2
    return np.asarray(array[..., top:top + size, left:left + size])


class LazyEventMapping(Mapping):
    """Read one event at a time so large ensembles do not fill system memory."""

    def __init__(self, root, path: str, crop_size: Optional[int], stored_scale: float):
        self.root = root
        self.path = path
        self.crop_size = crop_size
        self.to_encoded_255 = 255.0 / stored_scale
        self.names = tuple(sorted(root.group_keys()))

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, event_key: str) -> Dict[str, np.ndarray]:
        if event_key not in self.names:
            raise KeyError(event_key)
        group = self.root[event_key]
        available = set(group.array_keys())
        missing = {"target", "mask", "mean", "predictive_std"} - available
        if missing:
            raise ValueError(f"{self.path}/{event_key} is missing arrays: {sorted(missing)}")
        event: Dict[str, np.ndarray] = {}
        for key in FIELD_KEYS:
            if key not in available:
                continue
            # Slice first and materialize second. This keeps peak memory near one event.
            raw = _read_cropped(group[key], self.crop_size)
            event[key] = ensure_ethw(raw) if key in MEMBER_KEYS else ensure_thw(raw)
            if key not in {"mask", "member_logvar"} and self.to_encoded_255 != 1.0:
                event[key] = event[key].astype(np.float64) * self.to_encoded_255
        target_shape = event["target"].shape
        for key, value in event.items():
            compatible = value.shape[1:] == target_shape if key in MEMBER_KEYS else value.shape == target_shape
            if not compatible:
                raise ValueError(
                    f"{self.path}/{event_key}/{key} shape {value.shape} does not match target {target_shape}"
                )
        return event


def load_uncertainty_store(path: str, crop_size: Optional[int] = None) -> Dict[str, object]:
    """Open an inference store and lazily materialize one event at a time."""
    import zarr

    root = zarr.open_group(path, mode="r")
    attrs = dict(root.attrs)
    stored_scale = float(attrs.get("value_scale", 255.0))
    if stored_scale <= 0:
        raise ValueError(f"Invalid value_scale={stored_scale} in {path}")
    events = LazyEventMapping(root, path, crop_size, stored_scale)
    if not events:
        raise ValueError(f"No event groups found in uncertainty store: {path}")
    attrs["source_value_scale"] = stored_scale
    attrs["evaluation_value_scale"] = 255.0
    return {"attrs": attrs, "events": events}


def evaluation_mask(mask: np.ndarray, region: str) -> np.ndarray:
    """Return a [T,H,W] selector; inference mask=1 means observed."""
    observed = ensure_thw(mask) > 0.5
    normalized = region.lower().strip()
    if normalized == "missing":
        return ~observed
    if normalized == "observed":
        return observed
    if normalized == "all":
        return np.ones_like(observed, dtype=bool)
    raise ValueError(f"Unknown evaluation_region={region!r}; use missing, observed, or all")


def encoded_to_mmhr(arr: np.ndarray, divisor: float = 3.0) -> np.ndarray:
    """Convert the repository's stored 0--255 rain encoding to mm h^-1."""
    values = np.asarray(arr, dtype=np.float64) / float(divisor)
    return np.clip(0.036 * np.power(10.0, values * 0.0625), 0.0, 200.0)


def convert_rain(arr: np.ndarray, enabled: bool, divisor: float) -> np.ndarray:
    return encoded_to_mmhr(arr, divisor=divisor) if enabled else np.asarray(arr, dtype=np.float64)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: str, payload: Mapping) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default, allow_nan=True)


def save_config_snapshot(path: str, cfg) -> None:
    save_json(path, asdict(cfg) if is_dataclass(cfg) else cfg)


def save_text(path: str, lines: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).rstrip() + "\n")
