"""Train any of the four P2I uncertainty variants."""

from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import mlflow
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from p2igan_bench.data.dataloader import P2IDataModule
from p2igan_bench.models import P2IDiscriminator, UncertaintyOutput, build_generator
from p2igan_bench.modules import (
    ActivationMonitor,
    QUALITY_METRIC_KEYS,
    calibration_figure,
    enable_mc_dropout,
    ensemble_crps_loss,
    gaussian_mixture_nll,
    gan_loss,
    members_figure,
    member_gaussian_nll,
    multiscale_energy_score,
    prediction_figure,
    quality_metrics,
    tensor_finite_summary,
)
from p2igan_bench.modules.losses import pm_loss, weighted_l1_distance


MODEL_NAMES = {
    "p2igan_joint", "p2igan_decoder_ensemble", "p2igan_mc_dropout",
    "p2igan_latent_comod",
}
LOSS_METRIC_KEYS = (
    "loss", "nll", "mixture_nll", "weighted_l1", "pm", "crps", "energy",
    "gen_adv_field", "gen_adv_joint", "uncertainty_temporal", "disc_field",
    "disc_joint", "member_spread", "aleatoric_std", "predictive_std",
)
HEALTH_METRIC_KEYS = (
    "output_abs_max", "nonfinite_count", "gradient_norm", "amp_scale",
    "learning_rate", "disc_learning_rate",
)
METRIC_KEYS = LOSS_METRIC_KEYS + QUALITY_METRIC_KEYS + HEALTH_METRIC_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train four-model P2I uncertainty benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--smoke-test", action="store_true", help="Run a few real-data optimizer steps without checkpoints")
    parser.add_argument("--smoke-batches", type=int, default=1)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, name))
        elif value is not None:
            out[name] = json.dumps(value) if isinstance(value, (list, tuple)) else value
    return out


def prepare_batch(batch: Iterable[torch.Tensor], device: torch.device):
    frames, masked, masks = batch
    return tuple(x.permute(0, 1, 4, 2, 3).to(device, non_blocking=True) for x in (frames, masked, masks))


class UncertaintyInterpolationTrainer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        seed_everything(int(cfg.get("seed", 42)))
        self.variant = str(cfg.get("model", {}).get("name", ""))
        if self.variant not in MODEL_NAMES:
            raise ValueError(f"Unsupported uncertainty model: {self.variant}")
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA unavailable; using CPU")
            self.device = torch.device("cpu")

        train_cfg = cfg["train"]
        performance = cfg.get("performance", {})
        self.amp_enabled = bool(performance.get("amp", True)) and self.device.type == "cuda"
        amp_name = str(performance.get("amp_dtype", "float16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
        self.accumulate_steps = max(1, int(train_cfg.get("accumulate_steps", 1)))
        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = bool(performance.get("cudnn_benchmark", True))
            torch.backends.cuda.matmul.allow_tf32 = bool(performance.get("allow_tf32", True))
            torch.backends.cudnn.allow_tf32 = bool(performance.get("allow_tf32", True))
            torch.set_float32_matmul_precision(str(performance.get("matmul_precision", "high")))

        dm = P2IDataModule(cfg)
        self.train_loader, self.val_loader = dm.train_dataloader(), dm.val_dataloader()
        self.network = build_generator(cfg).to(self.device)
        loss = cfg.get("loss", {})
        opt = train_cfg["optimizer"]
        fused_adam = bool(performance.get("fused_adam", True)) and self.device.type == "cuda"
        self.optimizer = Adam(self.network.parameters(), lr=float(opt.get("lr", 1e-4)),
                              betas=(float(opt.get("beta1", 0.0)), float(opt.get("beta2", 0.99))),
                              fused=fused_adam)

        disc_cfg = cfg.get("discriminator", {})
        length = int((cfg.get("data_loader") or cfg["data"]["train"]).get("sample_length", 16))
        self.discriminators = torch.nn.ModuleDict()
        if bool(disc_cfg.get("field_enabled", disc_cfg.get("enabled", True))):
            self.discriminators["field"] = P2IDiscriminator(length * 3, feature_channels=3).to(self.device)
        if self.variant == "p2igan_joint" and bool(disc_cfg.get("joint_enabled", True)):
            self.discriminators["joint"] = P2IDiscriminator(length * 4, feature_channels=4).to(self.device)
        disc_opt = train_cfg.get("discriminator_optimizer", opt)
        self.disc_optimizer = None
        if self.discriminators:
            self.disc_optimizer = Adam(self.discriminators.parameters(), lr=float(disc_opt.get("lr", 1e-4)),
                                       betas=(float(disc_opt.get("beta1", 0.0)), float(disc_opt.get("beta2", 0.99))),
                                       fused=fused_adam)

        self.weights = {
            "weighted_l1": float(loss.get("lambda_w", 1.0)),
            "pm": float(loss.get("lambda_pm", 0.0)),
            "crps": float(loss.get("lambda_crps", 0.0)),
            "energy": float(loss.get("lambda_energy", 0.0)),
            "adv_field": float(loss.get("lambda_adv_field", loss.get("lambda_adv", 0.0))),
            "adv_joint": float(loss.get("lambda_adv_joint", 0.0)),
            "uncertainty_temporal": float(loss.get("lambda_uncertainty_temporal", 0.0)),
        }
        self.beta = float(loss.get("beta", 0.0))
        self.pm_omega = float(loss.get("pm_omega", 10.0))
        self.gan_type = str(loss.get("gan_loss", "hinge"))
        uncertainty = cfg.get("uncertainty", {})
        self.train_members = int(uncertainty.get("train_samples", 4))
        self.val_members = int(uncertainty.get("val_samples", self.train_members))
        self.max_epochs = int(train_cfg.get("max_epochs", 100))
        self.log_every = max(1, int(train_cfg.get("log_step", 100)))
        schedule = cfg.get("schedule", {})
        self.distribution_start = float(schedule.get("distribution_start", 0.10))
        self.adversarial_start = float(schedule.get("adversarial_start", 0.40))
        self.variant_start = float(schedule.get("variant_start", 0.70))

        monitoring = cfg.get("monitoring", {})
        self.fail_on_nonfinite = bool(monitoring.get("fail_on_nonfinite", True))
        self.visualize_every = max(0, int(monitoring.get("visualize_every", 10)))
        self.visualize_samples = max(1, int(monitoring.get("visualize_samples", 1)))
        self.visualize_timesteps = tuple(
            int(value) for value in monitoring.get("visualize_timesteps", [0, 4, 8, 12, 15])
        )
        self.visualize_members = max(1, int(monitoring.get("visualize_members", 8)))
        self.log_checkpoint_artifact = bool(monitoring.get("log_checkpoint_artifact", True))
        self.checkpoint_every = max(1, int(monitoring.get("checkpoint_every", 5)))
        self.best_min_delta = max(0.0, float(monitoring.get("best_min_delta", 1e-5)))
        self.activation_warmup_steps = max(0, int(monitoring.get("activation_warmup_steps", 20)))
        self.max_activation_abs = float(monitoring.get("max_activation_abs", 1000.0))
        self.activation_monitor = ActivationMonitor(self.network)
        self._fixed_validation_batch = None

        self.save_dir = Path(cfg.get("save_dir", f"weights/{self.variant}"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.latest, self.best = self.save_dir / "latest.pt", self.save_dir / "best.pt"
        self.best_loss, self.start_epoch, self.global_step = float("inf"), 1, 0
        if bool(train_cfg.get("auto_resume", True)) and self.latest.exists():
            self._load(self.latest)

    def _phase(self, epoch: int) -> tuple[bool, float, bool]:
        progress = epoch / max(1, self.max_epochs)
        distribution = progress >= self.distribution_start
        if progress < self.adversarial_start:
            adv_scale = 0.0
        else:
            adv_scale = min(1.0, (progress - self.adversarial_start) / max(0.05, self.variant_start - self.adversarial_start))
        return distribution, adv_scale, progress >= self.variant_start

    def _forward(self, masked, masks, members: int) -> UncertaintyOutput:
        count = members if self.variant in {"p2igan_mc_dropout", "p2igan_latent_comod"} else 1
        return self.network(masked, masks, num_members=count)

    @staticmethod
    def _condition(masked, masks, candidate):
        return torch.cat([masked, masks, candidate], dim=2)

    @staticmethod
    def _uncertainty_proxy(target, prediction):
        residual = (target - prediction.detach()).abs().permute(0, 2, 1, 3, 4)
        proxy = F.avg_pool3d(residual, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        return proxy.permute(0, 2, 1, 3, 4).clamp(0.0, 1.0)

    def _joint(self, masked, masks, field, uncertainty):
        return torch.cat([masked, masks, field, uncertainty], dim=2)

    def _discriminator_losses(self, out, target, masked, masks, adv_scale):
        zero = target.new_zeros(())
        values = {"disc_field": zero, "disc_joint": zero}
        if adv_scale <= 0 or not self.discriminators:
            return values
        if "field" in self.discriminators:
            d = self.discriminators["field"]
            real = gan_loss(d(self._condition(masked, masks, target)), True, loss_type=self.gan_type, is_disc=True)
            fake = torch.stack([gan_loss(d(self._condition(masked, masks, m.detach())), False,
                                         loss_type=self.gan_type, is_disc=True) for m in out.member_mean]).mean()
            values["disc_field"] = 0.5 * (real + fake)
        if "joint" in self.discriminators:
            d = self.discriminators["joint"]
            proxy = self._uncertainty_proxy(target, out.mean)
            sigma = torch.exp(0.5 * out.member_logvar[0]).clamp(0.0, 1.0)
            real = gan_loss(d(self._joint(masked, masks, target, proxy)), True, loss_type=self.gan_type, is_disc=True)
            fake = gan_loss(d(self._joint(masked, masks, out.mean.detach(), sigma.detach())), False,
                            loss_type=self.gan_type, is_disc=True)
            values["disc_joint"] = 0.5 * (real + fake)
        return values

    def _generator_losses(self, out, target, masked, masks, epoch):
        distribution, adv_scale, variant_phase = self._phase(epoch)
        zero = target.new_zeros(())
        values = {key: zero for key in LOSS_METRIC_KEYS}
        values["weighted_l1"] = self.weights["weighted_l1"] * weighted_l1_distance(out.mean, target)
        if distribution:
            if self.variant == "p2igan_latent_comod":
                values["mixture_nll"] = gaussian_mixture_nll(out.member_mean, target, out.member_logvar)
            elif self.variant == "p2igan_decoder_ensemble" and self.network.training:
                means, logvars = out.member_mean.float(), out.member_logvar.float()
                variance = logvars.exp().clamp_min(1e-6)
                pixel_nll = 0.5 * ((target.float().unsqueeze(0) - means).square() / variance + logvars)
                per_head_sample = pixel_nll.flatten(2).mean(2)
                # Online Poisson bootstrap gives each decoder a different empirical training set.
                bootstrap = torch.poisson(torch.ones_like(per_head_sample))
                bootstrap[:, 0] += (bootstrap.sum(1) == 0).to(bootstrap.dtype)
                values["nll"] = (per_head_sample * bootstrap).sum() / bootstrap.sum().clamp_min(1.0)
            else:
                values["nll"] = member_gaussian_nll(out.member_mean, target, out.member_logvar, beta=self.beta)
        if variant_phase:
            if self.weights["pm"]:
                values["pm"] = self.weights["pm"] * pm_loss(out.mean, target, omega=self.pm_omega)
            if self.variant == "p2igan_decoder_ensemble" and self.weights["crps"]:
                values["crps"] = self.weights["crps"] * ensemble_crps_loss(out.member_mean, target)
            if self.variant == "p2igan_latent_comod" and self.weights["energy"]:
                values["energy"] = self.weights["energy"] * multiscale_energy_score(out.member_mean, target)
            if self.variant == "p2igan_joint" and self.weights["uncertainty_temporal"]:
                sigma = torch.exp(0.5 * out.member_logvar[0])
                values["uncertainty_temporal"] = self.weights["uncertainty_temporal"] * (sigma[:, 1:] - sigma[:, :-1]).abs().mean()
        if adv_scale and "field" in self.discriminators:
            losses = [gan_loss(self.discriminators["field"](self._condition(masked, masks, member)), True,
                               loss_type=self.gan_type, is_disc=False) for member in out.member_mean]
            values["gen_adv_field"] = self.weights["adv_field"] * adv_scale * torch.stack(losses).mean()
        if adv_scale and "joint" in self.discriminators:
            sigma = torch.exp(0.5 * out.member_logvar[0]).clamp(0.0, 1.0)
            logits = self.discriminators["joint"](self._joint(masked, masks, out.mean, sigma))
            values["gen_adv_joint"] = self.weights["adv_joint"] * adv_scale * gan_loss(
                logits, True, loss_type=self.gan_type, is_disc=False)
        values["member_spread"] = out.secondary_var.sqrt().mean().detach()
        values["aleatoric_std"] = out.aleatoric_var.sqrt().mean().detach()
        values["predictive_std"] = out.predictive_var.sqrt().mean().detach()
        optimized = ("nll", "mixture_nll", "weighted_l1", "pm", "crps", "energy",
                     "gen_adv_field", "gen_adv_joint", "uncertainty_temporal")
        values["loss"] = sum((values[k] for k in optimized), zero)
        return values

    @staticmethod
    def _requires_grad(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled)

    def _optimizer_step(self, optimizer, parameters):
        gradient_norm = 0.0
        if self.grad_clip_norm > 0:
            self.scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip_norm)
            gradient_norm = float(norm.detach())
        self.scaler.step(optimizer)
        return gradient_norm

    def _fail_nonfinite(self, *, epoch: int, batch_index: int, reason: str) -> None:
        message = f"Non-finite training state at epoch={epoch} batch={batch_index}: {reason}"
        logging.error(message)
        debug_path = self.save_dir / f"debug_nonfinite_e{epoch:04d}_b{batch_index:05d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "batch_index": batch_index,
                "global_step": self.global_step,
                "reason": reason,
                "config": self.cfg,
                "network": self.network.state_dict(),
                "discriminators": self.discriminators.state_dict(),
                "grad_scaler": self.scaler.state_dict(),
            },
            debug_path,
        )
        if mlflow.active_run():
            mlflow.set_tags({"status": "failed", "failure_reason": reason})
            mlflow.log_text(message, "debug/nonfinite.txt")
        raise FloatingPointError(message)

    def _check_finite(self, tensor: torch.Tensor, *, epoch: int, batch_index: int, name: str) -> None:
        if self.fail_on_nonfinite and not torch.isfinite(tensor).all():
            count = int((~torch.isfinite(tensor)).sum().item())
            self._fail_nonfinite(
                epoch=epoch,
                batch_index=batch_index,
                reason=f"{name} contains {count} NaN/Inf values",
            )

    def _run(self, loader, training: bool, members: int, epoch: int, max_batches: Optional[int] = None):
        self.network.train(training)
        if not training and self.variant == "p2igan_mc_dropout":
            enable_mc_dropout(self.network)
        self.discriminators.train(training)
        totals = {key: 0.0 for key in METRIC_KEYS}
        metric_counts = {key: 0 for key in METRIC_KEYS}
        activation_abs_max = 0.0
        activation_nonfinite_count = 0.0
        batches = 0
        samples = 0
        run_started = time.perf_counter()
        previous_end = run_started
        data_wait_seconds = 0.0
        if training:
            self.optimizer.zero_grad(set_to_none=True)
            if self.disc_optimizer is not None:
                self.disc_optimizer.zero_grad(set_to_none=True)
        selected_loader = loader if max_batches is None else islice(loader, max_batches)
        for index, batch in enumerate(selected_loader):
            data_wait_seconds += time.perf_counter() - previous_end
            target, masked, masks = prepare_batch(batch, self.device)
            samples += int(target.shape[0])
            distribution, adv_scale, variant_phase = self._phase(epoch)
            final_available_batch = min(len(loader), max_batches) if max_batches is not None else len(loader)
            should_step = (index + 1) % self.accumulate_steps == 0 or (index + 1) == final_available_batch
            next_step = self.global_step + (1 if training and should_step else 0)
            capture_activations = bool(
                training and should_step and (
                    next_step <= self.activation_warmup_steps
                    or
                    next_step % self.log_every == 0
                    or (index + 1) == final_available_batch
                )
            )
            disc_values = {"disc_field": target.new_zeros(()), "disc_joint": target.new_zeros(())}
            disc_gradient_norm = 0.0
            if training and self.disc_optimizer is not None and adv_scale > 0:
                with self._autocast():
                    with torch.no_grad():
                        disc_out = self._forward(masked, masks, members)
                    disc_values = self._discriminator_losses(disc_out, target, masked, masks, adv_scale)
                disc_total = disc_values["disc_field"] + disc_values["disc_joint"]
                self._check_finite(
                    disc_total,
                    epoch=epoch,
                    batch_index=index,
                    name="discriminator loss",
                )
                self.scaler.scale(disc_total / self.accumulate_steps).backward()
                if should_step:
                    disc_gradient_norm = self._optimizer_step(
                        self.disc_optimizer,
                        self.discriminators.parameters(),
                    )

            self._requires_grad(self.discriminators, False)
            if capture_activations:
                self.activation_monitor.start()
            with torch.set_grad_enabled(training):
                with self._autocast():
                    out = self._forward(masked, masks, members)
                    values = self._generator_losses(out, target, masked, masks, epoch)
            activation_stats = self.activation_monitor.stop() if capture_activations else {
                "activation_abs_max": 0.0,
                "activation_nonfinite_count": 0.0,
            }
            activation_abs_max = max(activation_abs_max, activation_stats["activation_abs_max"])
            activation_nonfinite_count += activation_stats["activation_nonfinite_count"]
            if (
                capture_activations
                and self.max_activation_abs > 0
                and activation_stats["activation_abs_max"] > self.max_activation_abs
            ):
                self._fail_nonfinite(
                    epoch=epoch,
                    batch_index=index,
                    reason=(
                        f"activation abs max {activation_stats['activation_abs_max']:.6g} "
                        f"exceeds limit {self.max_activation_abs:.6g}"
                    ),
                )

            monitor_batch = not training or capture_activations
            if monitor_batch:
                output_health = tensor_finite_summary(
                    (out.mean, out.member_logvar, out.predictive_var)
                )
                if self.fail_on_nonfinite and output_health["nonfinite_count"]:
                    self._fail_nonfinite(
                        epoch=epoch,
                        batch_index=index,
                        reason=f"model output contains {int(output_health['nonfinite_count'])} NaN/Inf values",
                    )
                values.update(quality_metrics(target, out.mean, out.predictive_var, masks))
                values["output_abs_max"] = target.new_tensor(output_health["output_abs_max"])
                values["nonfinite_count"] = target.new_tensor(output_health["nonfinite_count"])
            values["gradient_norm"] = target.new_zeros(())
            values["amp_scale"] = target.new_tensor(float(self.scaler.get_scale()))
            values["learning_rate"] = target.new_tensor(self.optimizer.param_groups[0]["lr"])
            values["disc_learning_rate"] = target.new_tensor(
                self.disc_optimizer.param_groups[0]["lr"] if self.disc_optimizer else 0.0
            )
            self._check_finite(values["loss"], epoch=epoch, batch_index=index, name="generator loss")

            if training:
                self.scaler.scale(values["loss"] / self.accumulate_steps).backward()
                if should_step:
                    gradient_norm = self._optimizer_step(self.optimizer, self.network.parameters())
                    values["gradient_norm"] = target.new_tensor(gradient_norm)
                    if not np.isfinite(gradient_norm):
                        self._fail_nonfinite(
                            epoch=epoch,
                            batch_index=index,
                            reason="generator gradient norm is NaN/Inf",
                        )
                    if disc_gradient_norm and not np.isfinite(disc_gradient_norm):
                        self._fail_nonfinite(
                            epoch=epoch,
                            batch_index=index,
                            reason="discriminator gradient norm is NaN/Inf",
                        )
                    self.scaler.update()
                    values["amp_scale"] = target.new_tensor(float(self.scaler.get_scale()))
                    if self.scaler.is_enabled() and self.scaler.get_scale() <= 0:
                        self._fail_nonfinite(
                            epoch=epoch,
                            batch_index=index,
                            reason="AMP GradScaler reached zero",
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.disc_optimizer is not None:
                        self.disc_optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
            self._requires_grad(self.discriminators, True)
            values.update(disc_values)
            for key in METRIC_KEYS:
                if key in values:
                    totals[key] += float(values[key].detach())
                    metric_counts[key] += 1
            batches += 1
            if training and should_step and self.global_step % self.log_every == 0:
                logging.info("step=%d total=%.5f nll=%.5f mix=%.5f crps=%.5f energy=%.5f adv=(%.5f,%.5f)",
                             self.global_step, totals["loss"] / batches, totals["nll"] / batches,
                             totals["mixture_nll"] / batches, totals["crps"] / batches,
                             totals["energy"] / batches, totals["gen_adv_field"] / batches,
                             totals["gen_adv_joint"] / batches)
                if mlflow.active_run():
                    step_metrics = {
                        f"step/train/{key}": float(value.detach())
                        for key, value in values.items()
                        if torch.is_tensor(value) and value.numel() == 1
                    }
                    step_metrics.update({
                        "step/train/activation_abs_max": activation_stats["activation_abs_max"],
                        "step/train/activation_nonfinite_count": activation_stats["activation_nonfinite_count"],
                        "step/train/discriminator_gradient_norm": disc_gradient_norm,
                        "step/phase/distribution_enabled": float(distribution),
                        "step/phase/adversarial_scale": adv_scale,
                        "step/phase/variant_enabled": float(variant_phase),
                    })
                    mlflow.log_metrics(step_metrics, step=self.global_step)
            previous_end = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - run_started
        result = {
            key: value / max(1, metric_counts[key])
            for key, value in totals.items()
        }
        result.update({
            "epoch_seconds": elapsed,
            "data_wait_seconds": data_wait_seconds,
            "data_wait_fraction": data_wait_seconds / max(elapsed, 1e-9),
            "samples_per_second": samples / max(elapsed, 1e-9),
            "activation_abs_max": activation_abs_max,
            "activation_nonfinite_count": activation_nonfinite_count,
            "gpu_memory_gib": (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                if self.device.type == "cuda" else 0.0
            ),
            "phase_distribution_enabled": float(self._phase(epoch)[0]),
            "phase_adversarial_scale": self._phase(epoch)[1],
            "phase_variant_enabled": float(self._phase(epoch)[2]),
        })
        return result

    def _save(self, epoch, path):
        torch.save({"epoch": epoch, "global_step": self.global_step, "best_val_loss": self.best_loss,
                    "config": self.cfg, "network": self.network.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "discriminators": self.discriminators.state_dict(),
                    "disc_optimizer": self.disc_optimizer.state_dict() if self.disc_optimizer else None,
                    "grad_scaler": self.scaler.state_dict()}, path)

    def _load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        saved_model = state.get("config", {}).get("model", {})
        active_model = self.cfg.get("model", {})
        saved_stack = saved_model.get("stack_version", "legacy_unscaled")
        active_stack = active_model.get("stack_version", "legacy_unscaled")
        if saved_stack != active_stack:
            raise ValueError(
                f"Checkpoint {path} uses stack_version={saved_stack!r}, but the active "
                f"model uses {active_stack!r}. Start a clean run; legacy collapsed "
                "checkpoints are intentionally not loaded into the stable residual stack."
            )
        self.network.load_state_dict(state["network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.discriminators.load_state_dict(state.get("discriminators", {}), strict=False)
        if self.disc_optimizer and state.get("disc_optimizer"):
            self.disc_optimizer.load_state_dict(state["disc_optimizer"])
        if state.get("grad_scaler"):
            self.scaler.load_state_dict(state["grad_scaler"])
        if self.scaler.is_enabled() and (
            not np.isfinite(self.scaler.get_scale()) or self.scaler.get_scale() <= 0
        ):
            raise ValueError(
                f"Checkpoint {path} contains invalid AMP scale={self.scaler.get_scale()}; "
                "start from a clean checkpoint or disable AMP"
            )
        self.start_epoch = int(state.get("epoch", 0)) + 1
        self.global_step = int(state.get("global_step", 0))
        self.best_loss = float(state.get("best_val_loss", float("inf")))

    def _fixed_validation(self):
        if self.val_loader is None:
            return None
        if self._fixed_validation_batch is None:
            batch = next(iter(self.val_loader))
            self._fixed_validation_batch = tuple(
                tensor[:self.visualize_samples].clone() for tensor in batch
            )
        return prepare_batch(self._fixed_validation_batch, self.device)

    def _log_visualizations(self, epoch: int) -> None:
        if not mlflow.active_run() or self.visualize_every <= 0:
            return
        if epoch != self.start_epoch and epoch % self.visualize_every != 0:
            return
        batch = self._fixed_validation()
        if batch is None:
            return
        target, masked, masks = batch
        was_training = self.network.training
        self.network.eval()
        if self.variant == "p2igan_mc_dropout":
            enable_mc_dropout(self.network)
        with torch.no_grad(), self._autocast():
            output = self._forward(masked, masks, self.val_members)
        self._check_finite(output.mean, epoch=epoch, batch_index=-1, name="visualization output")
        for sample in range(min(self.visualize_samples, target.shape[0])):
            figure = prediction_figure(
                target,
                masked,
                output,
                sample=sample,
                timesteps=self.visualize_timesteps,
            )
            mlflow.log_figure(
                figure,
                f"visualizations/epoch_{epoch:04d}/sample_{sample:02d}_prediction.png",
            )
            plt.close(figure)
            member_panel = members_figure(
                output,
                sample=sample,
                timesteps=self.visualize_timesteps,
                max_members=self.visualize_members,
            )
            if member_panel is not None:
                mlflow.log_figure(
                    member_panel,
                    f"visualizations/epoch_{epoch:04d}/sample_{sample:02d}_members.png",
                )
                plt.close(member_panel)
        calibration = calibration_figure(target, output.mean, output.predictive_var, masks)
        mlflow.log_figure(
            calibration,
            f"calibration/epoch_{epoch:04d}_reliability.png",
        )
        plt.close(calibration)
        self.network.train(was_training)

    def _log_run_metadata(self) -> None:
        if not mlflow.active_run():
            return
        precision = (
            str(self.amp_dtype).replace("torch.", "")
            if self.amp_enabled else "float32"
        )
        train_root = str(self.cfg.get("data", {}).get("train", {}).get("data_root", ""))
        mlflow.set_tags({
            "model_variant": self.variant,
            "uncertainty_kind": getattr(self.network, "secondary_kind", "unknown"),
            "precision": precision,
            "dataset": Path(train_root).name if train_root else "unknown",
            "seed": str(self.cfg.get("seed", 42)),
            "run_group": "four-model-comparison",
            "status": "running",
        })
        mlflow.log_dict(self.cfg, "config/resolved_config.json")
        trainable = sum(parameter.numel() for parameter in self.network.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.network.parameters())
        summary = (
            f"variant: {self.variant}\n"
            f"parameters_total: {total}\n"
            f"parameters_trainable: {trainable}\n\n"
            f"{self.network}\n"
        )
        mlflow.log_text(summary, "model/model_summary.txt")

    def train(self):
        tracking = self.cfg.get("mlflow", {})
        if tracking.get("tracking_uri"):
            mlflow.set_tracking_uri(tracking["tracking_uri"])
        if tracking.get("enabled", True) and tracking.get("experiment_name"):
            mlflow.set_experiment(tracking["experiment_name"])
        context = mlflow.start_run(run_name=tracking.get("run_name")) if tracking.get("enabled", True) else nullcontext()
        with context:
            if mlflow.active_run():
                mlflow.log_params(flatten_dict(self.cfg))
                self._log_run_metadata()
            try:
                for epoch in range(self.start_epoch, self.max_epochs + 1):
                    if self.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(self.device)
                    train = self._run(self.train_loader, True, self.train_members, epoch)
                    with torch.no_grad():
                        valid = self._run(self.val_loader, False, self.val_members, epoch) if self.val_loader else train
                    logging.info(
                        "epoch=%d train=%.6f valid=%.6f hidden_rmse=%.6f coverage90=%.4f "
                        "spread=%.6f predictive_std=%.6f",
                        epoch, train["loss"], valid["loss"], valid["rmse_hidden"],
                        valid["coverage_90"], valid["member_spread"], valid["predictive_std"],
                    )
                    improved = valid["loss"] < self.best_loss - self.best_min_delta
                    if improved:
                        self.best_loss = valid["loss"]
                    if (
                        epoch == self.start_epoch
                        or epoch % self.checkpoint_every == 0
                        or epoch == self.max_epochs
                    ):
                        self._save(epoch, self.latest)
                    if improved:
                        self._save(epoch, self.best)
                    if mlflow.active_run():
                        for split, metrics in (("train", train), ("valid", valid)):
                            mlflow.log_metrics(
                                {f"{split}/{key}": float(value) for key, value in metrics.items()},
                                step=epoch,
                            )
                        mlflow.log_metrics(
                            {
                                "best/valid_loss": self.best_loss,
                                "phase/distribution_enabled": valid["phase_distribution_enabled"],
                                "phase/adversarial_scale": valid["phase_adversarial_scale"],
                                "phase/variant_enabled": valid["phase_variant_enabled"],
                            },
                            step=epoch,
                        )
                        if improved:
                            mlflow.log_dict(
                                {"epoch": epoch, "valid_loss": self.best_loss},
                                "checkpoints/best.json",
                            )
                    self._log_visualizations(epoch)
            except KeyboardInterrupt:
                interrupted = self.save_dir / "interrupted.pt"
                self._save(max(self.start_epoch, locals().get("epoch", self.start_epoch)), interrupted)
                if mlflow.active_run():
                    mlflow.set_tags({"status": "interrupted", "failure_reason": "SIGINT/SIGTERM"})
                    mlflow.log_text(str(interrupted), "debug/interrupted_checkpoint.txt")
                raise
            except Exception as error:
                if mlflow.active_run():
                    mlflow.set_tags({
                        "status": "failed",
                        "failure_reason": f"{type(error).__name__}: {error}",
                    })
                raise
            else:
                if mlflow.active_run():
                    mlflow.set_tag("status", "completed")
                    if self.log_checkpoint_artifact and self.best.exists():
                        mlflow.log_artifact(str(self.best), artifact_path="checkpoints")

    def smoke_test(self, batches: int = 1):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        metrics = self._run(self.train_loader, True, self.train_members, self.max_epochs,
                            max_batches=max(1, batches))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_gib = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        else:
            peak_gib = 0.0
        elapsed = time.perf_counter() - started
        logging.info("SMOKE_OK variant=%s batches=%d seconds=%.3f peak_memory_gib=%.3f loss=%.6f",
                     self.variant, batches, elapsed, peak_gib, metrics["loss"])
        return {"variant": self.variant, "batches": batches, "seconds": elapsed,
                "peak_memory_gib": peak_gib, **metrics}


def main(args: Optional[argparse.Namespace] = None) -> None:
    def _interrupt(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _interrupt)
    parsed = args or parse_args()
    cfg = load_config(parsed.config)
    if parsed.smoke_test:
        cfg.setdefault("train", {})["auto_resume"] = False
        cfg.setdefault("mlflow", {})["enabled"] = False
    trainer = UncertaintyInterpolationTrainer(cfg)
    if parsed.smoke_test:
        trainer.smoke_test(parsed.smoke_batches)
    else:
        trainer.train()


if __name__ == "__main__":
    cli = parse_args()
    logging.basicConfig(level=getattr(logging, cli.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s")
    main(cli)
