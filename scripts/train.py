from __future__ import annotations

import argparse
import json
import logging
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import mlflow
import numpy as np
import torch
from torch.optim import Adam
from torchvision.utils import make_grid, save_image

from p2igan_bench.data.dataloader import P2IDataModule
from p2igan_bench.models import build_discriminator, build_generator
from p2igan_bench.modules import enable_mc_dropout, gan_loss, heteroscedastic_gaussian_nll, summarize_uncertainty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P2IGAN uncertainty model with heteroscedastic loss + MC dropout")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("p2igan_bench/config/p2igan_uncertainty.json"),
        help="Path to JSON/YAML config file.",
    )
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


def flatten_dict(data: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key))
        elif isinstance(value, (list, tuple)):
            items[new_key] = json.dumps(value)
        elif value is not None:
            items[new_key] = value
    return items


def prepare_batch(batch: Iterable[torch.Tensor], device: torch.device):
    frames, masked_frames, masks = batch
    frames = frames.permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
    masked_frames = masked_frames.permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
    masks = masks.permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
    return frames, masked_frames, masks


class UncertaintyInterpolationTrainer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        seed_everything(int(cfg.get("seed", 42)))
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but unavailable. Falling back to CPU.")
            self.device = torch.device("cpu")

        self.data_module = P2IDataModule(cfg)
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()

        if cfg.get("model", {}).get("name") != "p2igan_uncertainty":
            raise ValueError("This script only supports model.name = 'p2igan_uncertainty'.")
        self.network = build_generator(cfg).to(self.device)

        opt_cfg = cfg["train"]["optimizer"]
        self.optimizer = Adam(
            self.network.parameters(),
            lr=float(opt_cfg.get("lr", 1e-4)),
            betas=(float(opt_cfg.get("beta1", 0.0)), float(opt_cfg.get("beta2", 0.99))),
        )

        discriminator_cfg = cfg.get("discriminator", {})
        self.use_discriminator = bool(discriminator_cfg.get("enabled", False))
        self.discriminator = build_discriminator(cfg).to(self.device) if self.use_discriminator else None
        self.disc_optimizer = None
        if self.use_discriminator:
            disc_opt_cfg = cfg["train"].get("discriminator_optimizer", opt_cfg)
            self.disc_optimizer = Adam(
                self.discriminator.parameters(),
                lr=float(disc_opt_cfg.get("lr", opt_cfg.get("lr", 1e-4))),
                betas=(
                    float(disc_opt_cfg.get("beta1", opt_cfg.get("beta1", 0.0))),
                    float(disc_opt_cfg.get("beta2", opt_cfg.get("beta2", 0.99))),
                ),
            )

        uncertainty_cfg = cfg.get("uncertainty", {})
        loss_cfg = cfg.get("loss", {})
        self.train_samples = int(uncertainty_cfg.get("train_samples", 2))
        self.val_samples = int(uncertainty_cfg.get("val_samples", self.train_samples))
        self.loss_beta = float(loss_cfg.get("beta", 0.0))
        self.adv_weight = float(loss_cfg.get("adversarial_weight", 0.0))
        self.gan_loss_type = str(loss_cfg.get("gan_loss", "hinge")).lower()
        self.target_real_label = float(loss_cfg.get("target_real_label", 1.0))
        self.target_fake_label = float(loss_cfg.get("target_fake_label", 0.0))
        self.use_adversarial = self.use_discriminator and self.adv_weight > 0.0
        self.max_epochs = int(cfg["train"].get("max_epochs", 30))
        self.log_every = int(cfg["train"].get("log_step", 100))
        self.use_validation = bool(cfg["train"].get("use_validation", True))
        self.save_dir = Path(cfg.get("save_dir", "weights/p2igan_uncertainty"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.latest_checkpoint_path = self.save_dir / "latest.pt"
        self.best_checkpoint_path = self.save_dir / "best.pt"
        self.artifact_dir = self.save_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.start_epoch = 1
        self.auto_resume = bool(cfg["train"].get("auto_resume", True))
        self.visualize_every = int(cfg["train"].get("visualize_every", 1))
        self.visualize_samples = int(cfg["train"].get("visualize_samples", 1))
        self.visualize_timesteps = int(cfg["train"].get("visualize_timesteps", 4))
        self.enable_mlflow = bool(cfg.get("mlflow", {}).get("enabled", True))
        self.tracking_uri = cfg.get("mlflow", {}).get("tracking_uri") or cfg.get("tracking_uri")
        self.experiment_name = cfg.get("mlflow", {}).get("experiment_name") or cfg.get("experiment_name")
        self.run_name = cfg.get("mlflow", {}).get("run_name") or cfg.get("run_name")

        if self.auto_resume:
            self._auto_resume_if_available()

    def _forward_raw(self, masked_frames: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.network(masked_frames, masks)

    @staticmethod
    def _set_requires_grad(module: Optional[torch.nn.Module], value: bool) -> None:
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad_(value)

    def _generator_losses(self, raw: torch.Tensor, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        prediction = raw[..., 0]
        logvar = raw[..., 1]
        recon_loss = heteroscedastic_gaussian_nll(prediction, frames, logvar, beta=self.loss_beta)
        gen_adv_loss = torch.zeros((), device=frames.device, dtype=frames.dtype)
        if self.use_adversarial and self.discriminator is not None:
            logits_fake = self.discriminator(prediction)
            gen_adv_loss = self.adv_weight * gan_loss(
                logits_fake,
                True,
                loss_type=self.gan_loss_type,
                is_disc=False,
                target_real_label=self.target_real_label,
                target_fake_label=self.target_fake_label,
            )
        total_loss = recon_loss + gen_adv_loss
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "gen_adv_loss": gen_adv_loss,
        }

    def _discriminator_loss(self, frames: torch.Tensor, fake_prediction: torch.Tensor) -> torch.Tensor:
        if self.discriminator is None:
            return torch.zeros((), device=frames.device, dtype=frames.dtype)

        logits_real = self.discriminator(frames)
        logits_fake = self.discriminator(fake_prediction.detach())
        real_loss = gan_loss(
            logits_real,
            True,
            loss_type=self.gan_loss_type,
            is_disc=True,
            target_real_label=self.target_real_label,
            target_fake_label=self.target_fake_label,
        )
        fake_loss = gan_loss(
            logits_fake,
            False,
            loss_type=self.gan_loss_type,
            is_disc=True,
            target_real_label=self.target_real_label,
            target_fake_label=self.target_fake_label,
        )
        return 0.5 * (real_loss + fake_loss)

    def _discriminator_loss_multi_sample(
        self,
        frames: torch.Tensor,
        masked_frames: torch.Tensor,
        masks: torch.Tensor,
        *,
        sample_count: int,
    ) -> torch.Tensor:
        if self.discriminator is None:
            return torch.zeros((), device=frames.device, dtype=frames.dtype)

        disc_loss = torch.zeros((), device=frames.device, dtype=frames.dtype)
        for _ in range(sample_count):
            with torch.no_grad():
                raw_disc = self._forward_raw(masked_frames, masks)
                fake_prediction = raw_disc[..., 0]
            disc_loss = disc_loss + self._discriminator_loss(frames, fake_prediction)
        return disc_loss / float(sample_count)

    def _sample_predictions(self, masked_frames: torch.Tensor, masks: torch.Tensor, sample_count: int) -> torch.Tensor:
        self.network.eval()
        enable_mc_dropout(self.network)
        outputs = []
        with torch.no_grad():
            for _ in range(sample_count):
                outputs.append(self._forward_raw(masked_frames, masks).detach())
        return torch.stack(outputs, dim=0)

    def _normalize_frame(self, frame: torch.Tensor) -> torch.Tensor:
        frame = frame.float()
        min_value = frame.min()
        max_value = frame.max()
        denom = (max_value - min_value).clamp_min(1e-6)
        return ((frame - min_value) / denom).clamp(0.0, 1.0)

    def _select_timesteps(self, total_steps: int) -> list[int]:
        target = max(1, min(self.visualize_timesteps, total_steps))
        if target == 1:
            return [0]
        indices = np.linspace(0, total_steps - 1, num=target)
        return sorted({int(round(idx)) for idx in indices})

    def _build_visual_grid(
        self,
        masked_input: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        aleatoric_std: torch.Tensor,
        epistemic_std: torch.Tensor,
    ) -> torch.Tensor:
        rows = []
        indices = self._select_timesteps(int(target.shape[0]))
        sources = [masked_input, target, mean, aleatoric_std, epistemic_std]
        for src in sources:
            frames = []
            for idx in indices:
                frame = src[idx]
                if frame.dim() == 3:
                    frame = frame.mean(dim=0, keepdim=True)
                elif frame.dim() == 2:
                    frame = frame.unsqueeze(0)
                frame = self._normalize_frame(frame)
                frames.append(frame.repeat(3, 1, 1))
            rows.append(make_grid(frames, nrow=len(frames), padding=2))
        return make_grid(rows, nrow=1, padding=4)

    def _log_visualization(self, loader, epoch: int, prefix: str) -> None:
        if loader is None:
            return
        try:
            batch = next(iter(loader))
        except StopIteration:
            return

        frames, masked_frames, masks = prepare_batch(batch, self.device)
        raw_samples = self._sample_predictions(masked_frames, masks, max(2, self.val_samples))

        num_examples = min(int(frames.shape[0]), self.visualize_samples)
        for example_idx in range(num_examples):
            summary = summarize_uncertainty(raw_samples[:, example_idx])
            grid = self._build_visual_grid(
                masked_frames[example_idx].detach().cpu(),
                frames[example_idx].detach().cpu(),
                summary["mean"].detach().cpu(),
                summary["aleatoric_std"].detach().cpu(),
                summary["epistemic_std"].detach().cpu(),
            )
            out_path = self.artifact_dir / f"{prefix}_epoch{epoch:03d}_sample{example_idx:02d}.png"
            save_image(grid, out_path)
            if mlflow.active_run() is not None:
                mlflow.log_artifact(str(out_path), artifact_path="visualizations")

    def _run_loader(self, loader, *, training: bool, sample_count: int) -> Dict[str, float]:
        if loader is None:
            return {
                "loss": 0.0,
                "recon_loss": 0.0,
                "gen_adv_loss": 0.0,
                "disc_loss": 0.0,
            }

        if training:
            self.network.train()
            if self.discriminator is not None:
                self.discriminator.train()
        else:
            self.network.eval()
            enable_mc_dropout(self.network)
            if self.discriminator is not None:
                self.discriminator.eval()

        totals = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "gen_adv_loss": 0.0,
            "disc_loss": 0.0,
        }
        batches = 0

        for batch_idx, batch in enumerate(loader):
            frames, masked_frames, masks = prepare_batch(batch, self.device)
            disc_loss_value = torch.zeros((), device=self.device, dtype=frames.dtype)

            if training and self.use_adversarial and self.discriminator is not None and self.disc_optimizer is not None:
                self.disc_optimizer.zero_grad(set_to_none=True)
                disc_loss_value = self._discriminator_loss_multi_sample(
                    frames,
                    masked_frames,
                    masks,
                    sample_count=sample_count,
                )
                disc_loss_value.backward()
                self.disc_optimizer.step()

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            if self.use_discriminator and self.discriminator is not None:
                self._set_requires_grad(self.discriminator, False)

            loss = torch.zeros((), device=self.device, dtype=frames.dtype)
            recon_loss_value = torch.zeros((), device=self.device, dtype=frames.dtype)
            gen_adv_loss_value = torch.zeros((), device=self.device, dtype=frames.dtype)
            with torch.set_grad_enabled(training):
                for _ in range(sample_count):
                    raw = self._forward_raw(masked_frames, masks)
                    loss_parts = self._generator_losses(raw, frames)
                    loss = loss + loss_parts["total_loss"]
                    recon_loss_value = recon_loss_value + loss_parts["recon_loss"]
                    gen_adv_loss_value = gen_adv_loss_value + loss_parts["gen_adv_loss"]
                loss = loss / float(sample_count)
                recon_loss_value = recon_loss_value / float(sample_count)
                gen_adv_loss_value = gen_adv_loss_value / float(sample_count)

            if training:
                loss.backward()
                self.optimizer.step()
                self.global_step += 1

            if self.use_discriminator and self.discriminator is not None:
                self._set_requires_grad(self.discriminator, True)

            totals["loss"] += float(loss.detach())
            totals["recon_loss"] += float(recon_loss_value.detach())
            totals["gen_adv_loss"] += float(gen_adv_loss_value.detach())
            totals["disc_loss"] += float(disc_loss_value.detach())
            batches += 1

            if training and self.global_step % self.log_every == 0:
                logging.info(
                    "step=%d loss=%.6f recon=%.6f gen_adv=%.6f disc=%.6f batch=%d/%d",
                    self.global_step,
                    float(loss.detach()),
                    float(recon_loss_value.detach()),
                    float(gen_adv_loss_value.detach()),
                    float(disc_loss_value.detach()),
                    batch_idx + 1,
                    len(loader),
                )
                if mlflow.active_run() is not None:
                    mlflow.log_metric("train/step_loss", float(loss.detach()), step=self.global_step)
                    mlflow.log_metric("train/step_recon_loss", float(recon_loss_value.detach()), step=self.global_step)
                    mlflow.log_metric("train/step_gen_adv_loss", float(gen_adv_loss_value.detach()), step=self.global_step)
                    if self.use_adversarial:
                        mlflow.log_metric("train/step_disc_loss", float(disc_loss_value.detach()), step=self.global_step)

        denom = float(max(1, batches))
        return {key: value / denom for key, value in totals.items()}

    def _save_checkpoint(self, epoch: int, path: Path) -> None:
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "config": self.cfg,
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "discriminator": self.discriminator.state_dict() if self.discriminator is not None else None,
                "disc_optimizer": self.disc_optimizer.state_dict() if self.disc_optimizer is not None else None,
            },
            path,
        )

    def _auto_resume_if_available(self) -> None:
        if not self.latest_checkpoint_path.exists():
            return

        checkpoint = torch.load(self.latest_checkpoint_path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        discriminator_state = checkpoint.get("discriminator")
        if self.discriminator is not None and discriminator_state is not None:
            self.discriminator.load_state_dict(discriminator_state)

        disc_optimizer_state = checkpoint.get("disc_optimizer")
        if self.disc_optimizer is not None and disc_optimizer_state is not None:
            self.disc_optimizer.load_state_dict(disc_optimizer_state)

        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        last_epoch = int(checkpoint.get("epoch", 0))
        self.start_epoch = last_epoch + 1

        logging.info(
            "Resumed from %s (epoch=%d, next_epoch=%d, global_step=%d, best_val_loss=%.6f)",
            self.latest_checkpoint_path,
            last_epoch,
            self.start_epoch,
            self.global_step,
            self.best_val_loss,
        )

    def train(self) -> None:
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.enable_mlflow and self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

        run_context = mlflow.start_run(run_name=self.run_name) if self.enable_mlflow else nullcontext()
        with run_context:
            if mlflow.active_run() is not None:
                mlflow.log_params(flatten_dict(self.cfg))

            if self.start_epoch > self.max_epochs:
                logging.info(
                    "Checkpoint already reached max_epochs (start_epoch=%d, max_epochs=%d). Nothing to do.",
                    self.start_epoch,
                    self.max_epochs,
                )
                return

            for epoch in range(self.start_epoch, self.max_epochs + 1):
                train_metrics = self._run_loader(self.train_loader, training=True, sample_count=self.train_samples)
                logging.info(
                    "epoch=%d train_loss=%.6f recon=%.6f gen_adv=%.6f disc=%.6f",
                    epoch,
                    train_metrics["loss"],
                    train_metrics["recon_loss"],
                    train_metrics["gen_adv_loss"],
                    train_metrics["disc_loss"],
                )
                self._save_checkpoint(epoch, self.latest_checkpoint_path)

                if self.use_validation and self.val_loader is not None:
                    with torch.no_grad():
                        val_metrics = self._run_loader(self.val_loader, training=False, sample_count=self.val_samples)
                    logging.info(
                        "epoch=%d val_loss=%.6f recon=%.6f gen_adv=%.6f",
                        epoch,
                        val_metrics["loss"],
                        val_metrics["recon_loss"],
                        val_metrics["gen_adv_loss"],
                    )
                else:
                    val_metrics = dict(train_metrics)

                if mlflow.active_run() is not None:
                    mlflow.log_metric("train/epoch_loss", train_metrics["loss"], step=epoch)
                    mlflow.log_metric("train/epoch_recon_loss", train_metrics["recon_loss"], step=epoch)
                    mlflow.log_metric("train/epoch_gen_adv_loss", train_metrics["gen_adv_loss"], step=epoch)
                    if self.use_adversarial:
                        mlflow.log_metric("train/epoch_disc_loss", train_metrics["disc_loss"], step=epoch)
                    mlflow.log_metric("valid/epoch_loss", val_metrics["loss"], step=epoch)
                    mlflow.log_metric("valid/epoch_recon_loss", val_metrics["recon_loss"], step=epoch)
                    mlflow.log_metric("valid/epoch_gen_adv_loss", val_metrics["gen_adv_loss"], step=epoch)
                    mlflow.log_metric("train/lr", float(self.optimizer.param_groups[0]["lr"]), step=epoch)
                    if self.disc_optimizer is not None:
                        mlflow.log_metric("train/disc_lr", float(self.disc_optimizer.param_groups[0]["lr"]), step=epoch)
                    mlflow.log_artifact(str(self.latest_checkpoint_path), artifact_path="checkpoints")

                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self._save_checkpoint(epoch, self.best_checkpoint_path)
                    if mlflow.active_run() is not None:
                        mlflow.log_metric("valid/best_loss", self.best_val_loss, step=epoch)
                        mlflow.log_artifact(str(self.best_checkpoint_path), artifact_path="checkpoints")

                if epoch % self.visualize_every == 0:
                    self._log_visualization(self.train_loader, epoch, prefix="train")
                    if self.val_loader is not None:
                        self._log_visualization(self.val_loader, epoch, prefix="valid")


def main(args: Optional[argparse.Namespace] = None) -> None:
    parsed = args or parse_args()
    logging.info("Loading config from %s", parsed.config)
    cfg = load_config(parsed.config)
    trainer = UncertaintyInterpolationTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    cli_args = parse_args()
    logging.basicConfig(
        level=getattr(logging, cli_args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main(cli_args)
