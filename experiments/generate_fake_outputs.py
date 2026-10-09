"""Generate small, schema-accurate Zarr stores for end-to-end experiment tests."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter


VARIANTS = {
    "p2igan_joint": (1, "none", 5.0, 0.0),
    "p2igan_decoder_ensemble": (4, "between_decoder", 4.5, 3.5),
    "p2igan_mc_dropout": (16, "mc_dropout", 5.0, 2.8),
    "p2igan_condition_comod": (16, "mc_dropout_condition", 4.7, 4.2),
}


def _field(seed: int, time_steps: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    frames = []
    for step in range(time_steps):
        phase = 2 * np.pi * step / time_steps
        cx, cy = 0.35 * np.cos(phase), 0.3 * np.sin(phase)
        core = 125 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.10)
        band = 45 * np.exp(-((xx + 0.45) ** 2 + (yy - 0.2 * np.sin(phase)) ** 2) / 0.22)
        texture = gaussian_filter(rng.normal(size=(size, size)), sigma=2.0) * 4
        frames.append(np.clip(core + band + texture, 0, 255))
    return np.asarray(frames, dtype=np.float32)


def _mask(time_steps: int, size: int) -> np.ndarray:
    mask = np.zeros((time_steps, size, size), dtype=np.float32)
    mask[:, 2::6, 2::6] = 1.0
    return mask


def _correlated_noise(rng, shape, scale):
    return gaussian_filter(rng.normal(size=shape), sigma=(0.7, 1.2, 1.2)) * scale


def _write(group, name, value):
    value = np.asarray(value, dtype=np.float32)
    group.create_dataset(name, shape=value.shape, chunks=value.shape, dtype="float32")[:] = value


def _event(variant: str, event_seed: int, time_steps: int, size: int):
    members_count, secondary_kind, error_scale, structural_scale = VARIANTS[variant]
    rng = np.random.default_rng(event_seed + sum(map(ord, variant)))
    target = _field(event_seed, time_steps, size)
    mask = _mask(time_steps, size)
    base_error = _correlated_noise(rng, target.shape, error_scale)
    base_mean = np.clip(target + base_error, 0, 255)
    raw_members = []
    for _ in range(members_count):
        perturbation = _correlated_noise(rng, target.shape, structural_scale)
        member = np.clip(base_mean + perturbation, 0, 255)
        member = mask * target + (1 - mask) * member
        raw_members.append(member)
    member_mean = np.asarray(raw_members, dtype=np.float32)
    mean = member_mean.mean(axis=0)
    secondary_std = member_mean.std(axis=0)
    aleatoric_std = (3.0 + 0.035 * target).astype(np.float32)
    predictive_std = np.sqrt(aleatoric_std ** 2 + secondary_std ** 2)
    member_logvar = np.log(np.maximum((aleatoric_std / 255.0) ** 2, 1e-8))
    member_logvar = np.repeat(member_logvar[None], members_count, axis=0)
    ensemble = []
    for index in range(24):
        center = member_mean[index % members_count]
        sample = np.clip(center + _correlated_noise(rng, target.shape, 1.0) * aleatoric_std, 0, 255)
        ensemble.append(mask * target + (1 - mask) * sample)
    arrays = {
        "target": target[:, None], "masked_input": (target * mask)[:, None],
        "mask": mask[:, None], "mean": mean[:, None],
        "aleatoric_std": aleatoric_std[:, None], "secondary_std": secondary_std[:, None],
        "predictive_std": predictive_std[:, None], "member_mean": member_mean[:, :, None],
        "member_logvar": member_logvar[:, :, None], "ensemble": np.asarray(ensemble)[:, :, None],
    }
    if secondary_kind in {"between_decoder", "mc_dropout", "mc_dropout_condition"}:
        arrays["epistemic_std"] = arrays["secondary_std"]
    else:
        arrays["proxy_secondary_std"] = arrays["secondary_std"]
    return arrays, secondary_kind


def generate_store(path: Path, variant: str, events: int, seed: int, time_steps: int, size: int):
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update({
        "model_variant": variant, "secondary_kind": VARIANTS[variant][1],
        "value_scale": 255.0, "raw_members_saved": True, "synthetic": True,
    })
    for index in range(events):
        arrays, _ = _event(variant, seed + index, time_steps, size)
        event = root.create_group(f"event_{index + 1:02d}")
        for name, value in arrays.items():
            _write(event, name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/fake_outputs"))
    parser.add_argument("--events", type=int, default=4)
    parser.add_argument("--calibration-events", type=int, default=3)
    parser.add_argument("--time-steps", type=int, default=8)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for split, count, seed in (("test", args.events, 100), ("calibration", args.calibration_events, 900)):
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            path = split_dir / f"{variant}.zarr"
            if path.exists():
                if not args.overwrite:
                    raise FileExistsError(path)
                shutil.rmtree(path)
            generate_store(path, variant, count, seed, args.time_steps, args.size)
            print(f"[fake-output] wrote {path}")


if __name__ == "__main__":
    main()
