"""Inference for one or many uncertainty configs; each config writes its own Zarr."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import zarr

from p2igan_bench.data.dataloader import P2IDataModule
from p2igan_bench.models import UncertaintyOutput, build_generator
from p2igan_bench.modules import enable_mc_dropout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer one or more P2I uncertainty models")
    parser.add_argument("--config", type=Path, nargs="+", required=True, help="One or more model config files")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Only valid with one config")
    parser.add_argument("--model-dir", type=Path, default=None, help="Only valid with one config")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Only valid with one config")
    parser.add_argument("--output-dir", type=Path, default=Path("inference"))
    parser.add_argument("--predict-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--max-events", type=int, default=None,
        help="Optionally stop after this many test events (useful for preliminary comparisons)",
    )
    parser.add_argument("--overlap", type=int, default=12)
    parser.add_argument("--output-scale", type=float, default=255.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix in {".yaml", ".yml"}:
            import yaml
            data = yaml.safe_load(handle)
        else:
            data = json.load(handle)
    parent = data.pop("extends", None)
    if not parent:
        return data
    base = load_config((path.parent / parent).resolve())
    def merge(left, right):
        result = dict(left)
        for key, value in right.items():
            result[key] = merge(result[key], value) if key in result and isinstance(result[key], dict) and isinstance(value, dict) else value
        return result
    return merge(base, data)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def prepare_batch(batch: Iterable[torch.Tensor], device: torch.device):
    return tuple(x.permute(0, 1, 4, 2, 3).to(device, non_blocking=True) for x in batch)


def resolve_checkpoint(cfg, explicit: Optional[Path] = None, model_dir: Optional[Path] = None) -> Path:
    if explicit:
        return explicit
    root = Path(model_dir or cfg.get("save_dir", "weights"))
    for name in ("best.pt", "latest.pt"):
        if (root / name).exists():
            return root / name
    raise FileNotFoundError(f"No best.pt/latest.pt under {root}")


def _write(group, name, tensor, scale):
    value = tensor.detach().cpu().numpy().astype(np.float32)
    if name not in {"mask", "member_logvar"}:
        value *= scale
    group.create_dataset(name, shape=value.shape, chunks=value.shape, dtype="float32", overwrite=True)[:] = value


class UncertaintyInterpolationPredictor:
    def __init__(self, cfg, checkpoint: Path, device_override=None):
        self.cfg = cfg
        seed_everything(int(cfg.get("seed", 42)))
        self.variant = cfg["model"]["name"]
        self.device = torch.device(device_override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        dm = P2IDataModule(cfg)
        self.loader = dm.test_dataloader()
        if self.loader is None:
            raise RuntimeError("Test dataloader is not configured")
        self.network = build_generator(cfg).to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.checkpoint_epoch = state.get("epoch")
        self.checkpoint_global_step = state.get("global_step")
        checkpoint_model = state.get("config", {}).get("model", {})
        checkpoint_variant = checkpoint_model.get("name")
        if checkpoint_variant and checkpoint_variant != self.variant:
            raise ValueError(
                f"Checkpoint model {checkpoint_variant!r} does not match config model {self.variant!r}"
            )
        expected_stack = cfg.get("model", {}).get("stack_version", "legacy_unscaled")
        checkpoint_stack = checkpoint_model.get("stack_version", "legacy_unscaled")
        if checkpoint_stack != expected_stack:
            raise ValueError(
                f"Checkpoint stack {checkpoint_stack!r} does not match config stack "
                f"{expected_stack!r}; use a checkpoint trained with the current residual stack"
            )
        self.network.load_state_dict(state["network"])
        self.network.eval()
        if self.variant in {"p2igan_mc_dropout", "p2igan_condition_comod"}:
            enable_mc_dropout(self.network)
        unc = cfg.get("uncertainty", {})
        data = cfg.get("data_loader") or cfg["data"]["train"]
        self.predict_samples = int(unc.get("predict_samples", 16))
        self.window = int(data.get("sample_length", 16))

    def _forward(self, masked, masks, count, sample_seeds=None) -> UncertaintyOutput:
        members = count if self.variant in {
            "p2igan_bnn_joint", "p2igan_mc_dropout", "p2igan_condition_comod"
        } else 1
        with torch.no_grad():
            return self.network(
                masked,
                masks,
                num_members=members,
                sample_seeds=sample_seeds,
            )

    def predict_event(self, target, masked, masks, predict_samples, overlap):
        event_len, step = target.shape[1], max(1, self.window - overlap)
        member_sum = logvar_sum = None
        event_sample_seeds = None
        if self.variant in {
            "p2igan_bnn_joint", "p2igan_mc_dropout", "p2igan_condition_comod"
        }:
            event_sample_seeds = torch.randint(
                0, 2 ** 31 - 1, (predict_samples,), device="cpu"
            )
        counts = torch.zeros((event_len, 1, 1, 1), device=self.device, dtype=target.dtype)
        for start in range(0, event_len, step):
            stop, valid = start + self.window, min(self.window, event_len - start)
            current_masked, current_masks = masked[:, start:min(stop, event_len)], masks[:, start:min(stop, event_len)]
            if valid < self.window:
                pad = self.window - valid
                current_masked = torch.cat([current_masked, current_masked[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1)
                current_masks = torch.cat([current_masks, current_masks[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1)
            out = self._forward(
                current_masked,
                current_masks,
                predict_samples,
                sample_seeds=event_sample_seeds,
            )
            current_mean = out.member_mean[:, 0, :valid]
            current_logvar = out.member_logvar[:, 0, :valid]
            if member_sum is None:
                shape = (current_mean.shape[0], event_len, *current_mean.shape[2:])
                member_sum = torch.zeros(shape, device=self.device, dtype=target.dtype)
                logvar_sum = torch.zeros_like(member_sum)
            member_sum[:, start:start + valid] += current_mean
            logvar_sum[:, start:start + valid] += current_logvar
            counts[start:start + valid] += 1
        members = member_sum / counts.unsqueeze(0)
        member_logvar = logvar_sum / counts.unsqueeze(0)
        mean = members.mean(0)
        aleatoric_var = member_logvar.exp().mean(0)
        secondary_var = members.var(0, unbiased=False)
        predictive_var = aleatoric_var + secondary_var

        secondary_name = "epistemic_std" if self.variant in {
            "p2igan_bnn_joint", "p2igan_decoder_ensemble", "p2igan_mc_dropout",
            "p2igan_condition_comod",
        } else "proxy_secondary_std"
        result = {
            "target": target[0], "masked_input": masked[0], "mask": masks[0], "mean": mean,
            "aleatoric_std": aleatoric_var.clamp_min(1e-6).sqrt(),
            "secondary_std": secondary_var.clamp_min(0).sqrt(),
            "predictive_std": predictive_var.clamp_min(1e-6).sqrt(),
            "member_mean": members, "member_logvar": member_logvar,
        }
        result[secondary_name] = result["secondary_std"]
        return result


def run_one(config_path: Path, parsed, output_path: Path):
    cfg = load_config(config_path)
    if parsed.data_root:
        cfg.setdefault("data", {}).setdefault("test", {})["data_root"] = str(parsed.data_root)
    checkpoint = resolve_checkpoint(cfg, parsed.checkpoint, parsed.model_dir)
    predictor = UncertaintyInterpolationPredictor(cfg, checkpoint, parsed.device)
    if parsed.max_events is not None and parsed.max_events < 1:
        raise ValueError("max_events must be >= 1 when provided")
    if output_path.exists():
        if not parsed.overwrite:
            raise FileExistsError(output_path)
        shutil.rmtree(output_path)
    group = zarr.open_group(str(output_path), mode="w")
    predict_samples = predictor.predict_samples if parsed.predict_samples is None else parsed.predict_samples
    if predict_samples < 1:
        raise ValueError("predict_samples must be >= 1")
    group.attrs.update({"config_path": str(config_path), "checkpoint": str(checkpoint),
                        "model_variant": predictor.variant, "predict_samples": predict_samples,
                        "value_scale": parsed.output_scale,
                        "raw_members_saved": True,
                        "probabilistic_representation": "equal_weight_gaussian_mixture",
                        "checkpoint_epoch": predictor.checkpoint_epoch,
                        "checkpoint_global_step": predictor.checkpoint_global_step,
                        "max_events": parsed.max_events,
                        "secondary_kind": getattr(predictor.network, "secondary_kind", "unknown"),
                        "member_logvar_scale": 1.0,
                        "model_config_json": json.dumps(cfg.get("model", {}), sort_keys=True),
                        "loss_config_json": json.dumps(cfg.get("loss", {}), sort_keys=True)})
    started = time.time()
    dataset = predictor.loader.dataset
    for index, batch in enumerate(predictor.loader):
        if parsed.max_events is not None and index >= parsed.max_events:
            break
        target, masked, masks = prepare_batch(batch, predictor.device)
        result = predictor.predict_event(
            target, masked, masks, predict_samples, max(0, parsed.overlap)
        )
        event = group.create_group(f"event_{index + 1:02d}")
        for key, value in result.items():
            _write(event, key, value, parsed.output_scale)
        if hasattr(dataset, "video_files"):
            event.attrs["source"] = str(dataset.video_files[index])
        if (index + 1) % max(1, parsed.log_every) == 0:
            logging.info("%s: %d/%d events (%.2f events/s)", predictor.variant, index + 1,
                         len(dataset), (index + 1) / max(time.time() - started, 1e-6))
    logging.info("Saved %s", output_path)


def main(args: Optional[argparse.Namespace] = None):
    parsed = args or parse_args()
    if len(parsed.config) > 1 and (parsed.output or parsed.checkpoint or parsed.model_dir):
        raise ValueError("--output/--checkpoint/--model-dir can only be used with one config")
    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    for config_path in parsed.config:
        cfg = load_config(config_path)
        variant = cfg.get("model", {}).get("name", config_path.stem)
        output = parsed.output if len(parsed.config) == 1 and parsed.output else parsed.output_dir / f"{variant}.zarr"
        run_one(config_path, parsed, Path(output))


if __name__ == "__main__":
    cli = parse_args()
    logging.basicConfig(level=getattr(logging, cli.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s")
    main(cli)
