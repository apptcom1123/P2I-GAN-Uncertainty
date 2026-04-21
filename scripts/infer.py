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
from p2igan_bench.models import build_generator
from p2igan_bench.modules import enable_mc_dropout, generate_ensemble_from_uncertainty, summarize_uncertainty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for P2IGAN uncertainty interpolation ensembles")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("p2igan_bench/config/p2igan_uncertainty.json"),
        help="Path to JSON/YAML config file.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to model checkpoint (.pt).")
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory containing best.pt/latest.pt.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override data.test.data_root.")
    parser.add_argument("--output", type=Path, default=None, help="Output zarr path.")
    parser.add_argument("--predict-samples", type=int, default=None, help="Override raw MC sample count.")
    parser.add_argument("--ensemble-size", type=int, default=None, help="Override final ensemble member count.")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g. cuda:0, cpu).")
    parser.add_argument("--log-every", type=int, default=10, help="Log progress every N samples.")
    parser.add_argument("--stride", type=int, default=None, help="Sliding window length.")
    parser.add_argument("--overlap", type=int, default=12, help="Sliding window overlap.")
    parser.add_argument("--output-scale", type=float, default=255.0, help="Scale factor for stored outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output zarr.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(f)
        return json.load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_batch(batch: Iterable[torch.Tensor], device: torch.device):
    frames, masked_frames, masks = batch
    frames = frames.permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
    masked_frames = masked_frames.permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
    masks = masks.permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
    return frames, masked_frames, masks


def resolve_checkpoint(cfg: Dict[str, Any], args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return args.checkpoint

    base_dir = Path(args.model_dir or cfg.get("save_dir", "weights/p2igan_uncertainty"))
    if base_dir.is_file():
        return base_dir
    for candidate_name in ("best.pt", "latest.pt"):
        candidate = base_dir / candidate_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve checkpoint under {base_dir}")


def _write_array(group, name: str, value: np.ndarray) -> None:
    group.create_dataset(
        name,
        shape=value.shape,
        chunks=value.shape,
        dtype="float32",
        overwrite=True,
    )[:] = value.astype(np.float32)


class UncertaintyInterpolationPredictor:
    def __init__(self, cfg: Dict[str, Any], checkpoint_path: Path, device_override: Optional[str] = None) -> None:
        self.cfg = cfg
        seed_everything(int(cfg.get("seed", 42)))
        if device_override is not None:
            cfg["device"] = device_override

        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but unavailable. Falling back to CPU.")
            self.device = torch.device("cpu")

        if cfg.get("model", {}).get("name") != "p2igan_uncertainty":
            raise ValueError("This script only supports model.name = 'p2igan_uncertainty'.")

        self.data_module = P2IDataModule(cfg)
        self.test_loader = self.data_module.test_dataloader()
        if self.test_loader is None:
            raise RuntimeError("Test dataloader is not configured.")

        self.network = build_generator(cfg).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.network.eval()

        uncertainty_cfg = cfg.get("uncertainty", {})
        train_data_cfg = cfg.get("data_loader") or cfg["data"]["train"]
        self.predict_samples = int(uncertainty_cfg.get("predict_samples", 16))
        self.ensemble_size = int(uncertainty_cfg.get("ensemble_size", 48))
        self.window_length = int(train_data_cfg.get("sample_length", 16))
        self.temporal_mode = str(
            uncertainty_cfg.get("correlated_noise", {}).get("temporal_mode", "per_frame")
        ).lower()
        self.use_correlated_noise = bool(
            uncertainty_cfg.get("correlated_noise", {}).get("enabled", True)
        )

    def _sample_raw_predictions(self, masked_frames: torch.Tensor, masks: torch.Tensor, sample_count: int) -> torch.Tensor:
        self.network.eval()
        enable_mc_dropout(self.network)
        outputs = []
        with torch.no_grad():
            for _ in range(sample_count):
                outputs.append(self.network(masked_frames, masks).detach())
        return torch.stack(outputs, dim=0)

    def predict_event(
        self,
        frames: torch.Tensor,
        masked_frames: torch.Tensor,
        masks: torch.Tensor,
        *,
        predict_samples: int,
        ensemble_size: int,
        overlap: int,
    ) -> Dict[str, torch.Tensor]:
        event_len = int(frames.shape[1])
        window = self.window_length
        step = max(1, window - overlap)

        accum = {
            "mean": torch.zeros_like(frames[0]),
            "aleatoric_var": torch.zeros_like(frames[0]),
            "epistemic_var": torch.zeros_like(frames[0]),
            "predictive_var": torch.zeros_like(frames[0]),
        }
        counts = torch.zeros((event_len, 1, 1, 1), device=self.device, dtype=frames.dtype)

        for start_idx in range(0, event_len, step):
            end_idx = start_idx + window
            if end_idx > event_len:
                pad_len = end_idx - event_len

                def _pad(x: torch.Tensor):
                    return torch.cat([x, x[:, -1:, :, :, :].repeat(1, pad_len, 1, 1, 1)], dim=1)

                current_masked = _pad(masked_frames[:, start_idx:event_len])
                current_masks = _pad(masks[:, start_idx:event_len])
                valid_len = event_len - start_idx
            else:
                current_masked = masked_frames[:, start_idx:end_idx]
                current_masks = masks[:, start_idx:end_idx]
                valid_len = window

            raw_samples = self._sample_raw_predictions(current_masked, current_masks, predict_samples)
            summary = summarize_uncertainty(raw_samples[:, 0, :valid_len])

            sl = slice(start_idx, start_idx + valid_len)
            accum["mean"][sl] += summary["mean"]
            accum["aleatoric_var"][sl] += summary["aleatoric_var"]
            accum["epistemic_var"][sl] += summary["epistemic_var"]
            accum["predictive_var"][sl] += summary["predictive_var"]
            counts[sl] += 1.0

        counts = counts.clamp_min(1.0)
        mean = accum["mean"] / counts
        aleatoric_var = accum["aleatoric_var"] / counts
        epistemic_var = accum["epistemic_var"] / counts
        predictive_var = accum["predictive_var"] / counts

        if ensemble_size > 0:
            if self.use_correlated_noise:
                ensemble = generate_ensemble_from_uncertainty(
                    mean,
                    predictive_var,
                    masked_frames[0],
                    num_members=ensemble_size,
                    temporal_mode=self.temporal_mode,
                    clip_min=0.0,
                    clip_max=1.0,
                )
            else:
                noise = torch.randn((ensemble_size, *mean.shape), device=self.device, dtype=mean.dtype)
                ensemble = mean.unsqueeze(0) + noise * torch.sqrt(predictive_var.clamp_min(1e-6)).unsqueeze(0)
                ensemble = ensemble.clamp_(0.0, 1.0)
        else:
            ensemble = torch.empty((0, *mean.shape), device=self.device, dtype=mean.dtype)

        return {
            "target": frames[0],
            "masked_input": masked_frames[0],
            "mask": masks[0],
            "mean": mean,
            "aleatoric_std": torch.sqrt(aleatoric_var.clamp_min(1e-6)),
            "epistemic_std": torch.sqrt(epistemic_var.clamp_min(0.0)),
            "predictive_std": torch.sqrt(predictive_var.clamp_min(1e-6)),
            "ensemble": ensemble,
        }


def main(args: Optional[argparse.Namespace] = None) -> None:
    parsed = args or parse_args()
    logging.info("Loading config from %s", parsed.config)
    cfg = load_config(parsed.config)
    if parsed.data_root is not None:
        cfg.setdefault("data", {}).setdefault("test", {})["data_root"] = str(parsed.data_root)

    checkpoint_path = resolve_checkpoint(cfg, parsed)
    predictor = UncertaintyInterpolationPredictor(cfg, checkpoint_path, device_override=parsed.device)

    predict_samples = parsed.predict_samples or predictor.predict_samples
    ensemble_size = parsed.ensemble_size or predictor.ensemble_size
    output_path = parsed.output or (Path(cfg.get("save_dir", "weights/p2igan_uncertainty")) / "test_uncertainty.zarr")
    output_path = Path(output_path)
    if output_path.exists():
        if parsed.overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"Output already exists: {output_path}")

    output_scale = float(parsed.output_scale)
    group = zarr.open_group(str(output_path), mode="w")
    group.attrs.update(
        {
            "config_path": str(parsed.config),
            "checkpoint": str(checkpoint_path),
            "predict_samples": int(predict_samples),
            "ensemble_size": int(ensemble_size),
            "value_scale": output_scale,
            "task": "spatio-temporal interpolation uncertainty",
            "epistemic_method": "mc_dropout",
            "aleatoric_method": "heteroscedastic_gaussian_nll",
        }
    )

    dataset = predictor.test_loader.dataset
    total_events = len(dataset)
    overlap = max(0, int(parsed.overlap))
    start = time.time()
    for batch_idx, batch in enumerate(predictor.test_loader):
        frames, masked_frames, masks = prepare_batch(batch, predictor.device)
        result = predictor.predict_event(
            frames,
            masked_frames,
            masks,
            predict_samples=predict_samples,
            ensemble_size=ensemble_size,
            overlap=overlap,
        )

        event_name = f"event_{batch_idx + 1:02d}"
        event_group = group.create_group(event_name)
        for key, value in result.items():
            array = value.detach().cpu().numpy()
            if key != "mask":
                array = array * output_scale
            _write_array(event_group, key, array)
        if hasattr(dataset, "video_files"):
            event_group.attrs["source"] = str(dataset.video_files[batch_idx])

        if (batch_idx + 1) % max(1, int(parsed.log_every)) == 0 or (batch_idx + 1) == total_events:
            elapsed = max(time.time() - start, 1e-6)
            logging.info(
                "predicted %d/%d events | %.2f events/sec",
                batch_idx + 1,
                total_events,
                (batch_idx + 1) / elapsed,
            )

    logging.info("Inference completed. Output saved to %s", output_path)


if __name__ == "__main__":
    cli_args = parse_args()
    logging.basicConfig(
        level=getattr(logging, cli_args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main(cli_args)
