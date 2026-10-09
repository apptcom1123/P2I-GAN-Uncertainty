"""Train a configured P2I uncertainty variant."""

from __future__ import annotations

import argparse
import copy
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
from mlflow.tracking import MlflowClient
from torch.optim import Adam

from p2igan_bench.data.dataloader import P2IDataModule
from p2igan_bench.models import (
    P2IDiscriminator, P2IJointGenerator, UncertaintyOutput, build_generator,
)
from p2igan_bench.modules import (
    ActivationMonitor,
    QUALITY_METRIC_KEYS,
    calibration_figure,
    enable_mc_dropout,
    gan_loss,
    members_figure,
    member_gaussian_nll,
    prediction_figure,
    quality_metrics,
    tensor_finite_summary,
)
from p2igan_bench.modules.losses import (
    compute_forward_difference,
    pm_loss,
    temporal_difference_kl_loss,
)


MODEL_NAMES = {
    "p2igan_joint", "p2igan_bnn_joint", "p2igan_decoder_ensemble", "p2igan_mc_dropout",
    "p2igan_condition_comod",
}
LOSS_METRIC_KEYS = (
    "loss", "nll", "kl", "temporal_kl", "pm", "gen_adv_field",
    "uncertainty_temporal", "disc_field",
)
UNCERTAINTY_METRIC_KEYS = ("member_spread", "aleatoric_std", "predictive_std")
HEALTH_METRIC_KEYS = (
    "output_abs_max", "nonfinite_count", "gradient_norm", "amp_scale",
    "learning_rate", "disc_learning_rate",
)
METRIC_KEYS = (
    LOSS_METRIC_KEYS + UNCERTAINTY_METRIC_KEYS + QUALITY_METRIC_KEYS
    + HEALTH_METRIC_KEYS
)

MLFLOW_LOSS_NAMES = {
    "loss": "backprop_total",
    "nll": "nll",
    "kl": "bayesian_kl",
    "temporal_kl": "temporal_kl",
    "uncertainty_temporal": "sigma_temporal_smoothness",
    "pm": "pm",
    "gen_adv_field": "gan_generator_field",
    "disc_field": "gan_discriminator_field",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
DEFAULT_MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a P2I uncertainty model")
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


def resolve_mlflow_tracking_uri(value: Optional[str]) -> str:
    """Resolve a repository-relative SQLite URI to an absolute database URI."""
    uri = str(value or DEFAULT_MLFLOW_TRACKING_URI)
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri
    database = Path(uri[len(prefix):])
    if not database.is_absolute():
        database = PROJECT_ROOT / database
    database.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database.resolve()}"


def resolve_mlflow_artifact_root(value: Optional[str]) -> str:
    """Resolve a local artifact directory and return its file URI."""
    root = str(value or DEFAULT_MLFLOW_ARTIFACT_ROOT)
    if "://" in root:
        return root
    path = Path(root)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve().as_uri()


def configure_mlflow_storage(tracking: Dict[str, Any]) -> str:
    """Configure the SQL backend and ensure the named experiment exists."""
    tracking_uri = resolve_mlflow_tracking_uri(tracking.get("tracking_uri"))
    artifact_root = resolve_mlflow_artifact_root(tracking.get("artifact_root"))
    experiment_name = str(tracking["experiment_name"])
    if tracking_uri.startswith("sqlite:///"):
        database = Path(tracking_uri[len("sqlite:///"):])
        if not database.exists():
            # MlflowClient does not expose ``default_artifact_root`` for a
            # direct SQL backend. Initialize the store explicitly so even the
            # built-in Default experiment points at mlartifacts/, never at the
            # deprecated file-based mlruns/ store.
            from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
            SqlAlchemyStore(tracking_uri, artifact_root)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_root,
        )
    else:
        experiment_id = experiment.experiment_id
        if experiment.artifact_location.rstrip("/") != artifact_root.rstrip("/"):
            logging.warning(
                "MLflow experiment %s already uses artifact root %s (configured: %s)",
                experiment_name,
                experiment.artifact_location,
                artifact_root,
            )
    mlflow.set_experiment(experiment_id=experiment_id)
    # The fluent client still creates an empty compatibility directory named
    # ``mlruns`` even though every experiment points at mlartifacts/. Remove
    # only that empty placeholder; a non-empty directory is never touched.
    legacy_store = PROJECT_ROOT / "mlruns"
    if legacy_store.is_dir():
        try:
            legacy_store.rmdir()
        except OSError:
            pass
    return tracking_uri


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    # Flatten a nested dictionary into a single-level dictionary with dot-separated keys.
    out = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, name))
        elif value is not None:
            out[name] = json.dumps(value) if isinstance(value, (list, tuple)) else value
    return out


def prepare_batch(batch: Iterable[torch.Tensor], device: torch.device):
    # Permute the batch tensors to (batch, channels, time, height, width) and move to the specified device.
    frames, masked, masks = batch
    return tuple(x.permute(0, 1, 4, 2, 3).to(device, non_blocking=True) for x in (frames, masked, masks))


class UncertaintyInterpolationTrainer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        # Initialize the trainer with the given configuration, set up data loaders, model, optimizer, and other training parameters.
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

        # Set up data module and data loaders
        dm = P2IDataModule(cfg)
        self.train_loader, self.val_loader = dm.train_dataloader(), dm.val_dataloader()
        self.train_size = max(1, len(self.train_loader.dataset))
        self.network = build_generator(cfg)
        pretrained_joint = cfg.get("model", {}).get("pretrained_joint_checkpoint")
        if self.variant == "p2igan_bnn_joint" and pretrained_joint:
            checkpoint_path = Path(pretrained_joint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            source_cfg = copy.deepcopy(checkpoint.get("config", cfg))
            source_cfg.setdefault("model", {})["name"] = "p2igan_joint"
            source = P2IJointGenerator(source_cfg, init_weights=False)
            source.load_state_dict(checkpoint["network"])
            self.network.initialize_from_joint(source)
            del source, checkpoint
            logging.info(
                "Initialized Bayesian posterior means from %s", checkpoint_path
            )
        self.network = self.network.to(self.device)
        loss = cfg.get("loss", {})
        opt = train_cfg["optimizer"]
        fused_adam = bool(performance.get("fused_adam", True)) and self.device.type == "cuda"
        generator_weight_decay = float(opt.get("weight_decay", 0.0))
        if generator_weight_decay < 0.0:
            raise ValueError("optimizer.weight_decay must be non-negative")
        self.generator_weight_decay = generator_weight_decay
        generator_parameters = self.network.parameters()
        if generator_weight_decay > 0.0:
            # Gal-style L2 applies to weight tensors, not scalar/vector biases
            # or normalization affine parameters.  Keeping it inside Adam is
            # equivalent to adding (weight_decay / 2) * ||theta||^2 to the
            # gradient objective, without materializing the very large expert
            # maps in a second autograd graph.
            decay, no_decay = [], []
            for parameter in self.network.parameters():
                (decay if parameter.ndim >= 2 else no_decay).append(parameter)
            generator_parameters = [
                {"params": decay, "weight_decay": generator_weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]
        self.optimizer = Adam(generator_parameters, lr=float(opt.get("lr", 1e-4)),
                              betas=(float(opt.get("beta1", 0.0)), float(opt.get("beta2", 0.99))),
                              fused=fused_adam)

        # Set up discriminators if specified in the configuration
        disc_cfg = cfg.get("discriminator", {})
        length = int((cfg.get("data_loader") or cfg["data"]["train"]).get("sample_length", 16))
        self.discriminators = torch.nn.ModuleDict()
        if bool(disc_cfg.get("field_enabled", disc_cfg.get("enabled", True))):
            self.discriminators["field"] = P2IDiscriminator(length * 3, feature_channels=3).to(self.device)
        disc_opt = train_cfg.get("discriminator_optimizer", opt)
        self.disc_optimizer = None
        if self.discriminators:
            self.disc_optimizer = Adam(self.discriminators.parameters(), lr=float(disc_opt.get("lr", 1e-4)),
                                       betas=(float(disc_opt.get("beta1", 0.0)), float(disc_opt.get("beta2", 0.99))),
                                       weight_decay=float(disc_opt.get("weight_decay", 0.0)),
                                       fused=fused_adam)

        self.weights = {
            "pm": float(loss.get("lambda_pm", 0.0)),
            "adv_field": float(loss.get("lambda_adv_field", loss.get("lambda_adv", 0.0))),
            "temporal_kl": float(loss.get("k1_weight", 0.0005)),
            "uncertainty_temporal": float(
                loss.get("lambda_uncertainty_temporal", 0.0005)
            ),
        }
        self.beta = float(loss.get("beta", 0.0))
        self.lambda_kl = float(loss.get("lambda_kl", 1.0))
        self.kl_warmup_epochs = max(0, int(loss.get("kl_warmup_epochs", 0)))
        self.pm_omega = float(loss.get("pm_omega", 10.0))
        self.gan_type = str(loss.get("gan_loss", "hinge"))
        uncertainty = cfg.get("uncertainty", {})
        self.train_members = int(uncertainty.get("train_samples", 4))
        self.val_members = int(uncertainty.get("val_samples", self.train_members))
        self.max_epochs = int(train_cfg.get("max_epochs", 100))
        self.log_every = max(1, int(train_cfg.get("log_step", 100)))

        # Set up monitoring and checkpointing parameters
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

        # Set up directories for saving model checkpoints and initialize training state
        self.save_dir = Path(cfg.get("save_dir", f"weights/{self.variant}"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.latest, self.best = self.save_dir / "latest.pt", self.save_dir / "best.pt"
        self.best_loss, self.start_epoch, self.global_step = float("inf"), 1, 0
        if bool(train_cfg.get("auto_resume", True)) and self.latest.exists():
            self._load(self.latest)

    def _forward(self, masked, masks, members: int) -> UncertaintyOutput:
        count = members if self.variant in {
            "p2igan_bnn_joint", "p2igan_mc_dropout", "p2igan_condition_comod"
        } else 1
        return self.network(masked, masks, num_members=count)

    @staticmethod
    def _condition(masked, masks, candidate):
        return torch.cat([masked, masks, candidate], dim=2)

    def _discriminator_losses(self, out, target, masked, masks):
        # Compute the field discriminator loss from the first optimizer step.
        zero = target.new_zeros(())
        values = {"disc_field": zero}
        if not self.discriminators:
            return values
        if "field" in self.discriminators:
            # Compute the discriminator loss for the field discriminator
            d = self.discriminators["field"]
            real = gan_loss(d(self._condition(masked, masks, target)), True, loss_type=self.gan_type, is_disc=True)
            fake = torch.stack([gan_loss(d(self._condition(masked, masks, m.detach())), False,
                                         loss_type=self.gan_type, is_disc=True) for m in out.member_mean]).mean()
            values["disc_field"] = 0.5 * (real + fake)
        return values

    def _distribution_losses(self, out, target):
        zero = target.new_zeros(())
        values = {
            "nll": zero,
            "temporal_kl": zero,
            "uncertainty_temporal": zero,
        }
        if self.variant == "p2igan_decoder_ensemble" and self.network.training:
            means, logvars = out.member_mean.float(), out.member_logvar.float()
            variance = logvars.exp().clamp_min(1e-6)
            pixel_nll = 0.5 * ((target.float().unsqueeze(0) - means).square() / variance + logvars)
            per_head_sample = pixel_nll.flatten(2).mean(2)
            # Online Poisson bootstrap gives each decoder a different empirical training set.
            bootstrap = torch.poisson(torch.ones_like(per_head_sample))
            bootstrap[:, 0] += (bootstrap.sum(1) == 0).to(bootstrap.dtype)
            values["nll"] = (per_head_sample * bootstrap).sum() / bootstrap.sum().clamp_min(1.0)
        else:
            values["nll"] = member_gaussian_nll(
                out.member_mean, target, out.member_logvar, beta=self.beta
            )
        member_prediction = out.member_mean.flatten(0, 1)
        member_target = target.unsqueeze(0).expand_as(
            out.member_mean
        ).flatten(0, 1)
        values["temporal_kl"] = (
            self.weights["temporal_kl"]
            * temporal_difference_kl_loss(member_prediction, member_target)
        )
        member_sigma = out.member_logvar.mul(0.5).exp().flatten(0, 1)
        values["uncertainty_temporal"] = self.weights[
            "uncertainty_temporal"
        ] * compute_forward_difference(member_sigma).abs().mean()
        return values

    def _generator_losses(self, out, target, masked, masks, epoch):
        zero = target.new_zeros(())
        values = {key: zero for key in LOSS_METRIC_KEYS}
        values.update(self._distribution_losses(out, target))
        if self.variant == "p2igan_bnn_joint":
            warmup = (
                1.0 if self.kl_warmup_epochs == 0
                else min(1.0, epoch / self.kl_warmup_epochs)
            )
            outputs_per_event = max(1, target[0].numel())
            kl_normalizer = self.train_size * outputs_per_event
            values["kl"] = (
                self.lambda_kl * warmup
                * self.network.kl_divergence()
                / kl_normalizer
            )
        if self.weights["pm"]:
            values["pm"] = self.weights["pm"] * pm_loss(
                out.mean, target, omega=self.pm_omega
            )
        if "field" in self.discriminators and self.weights["adv_field"] > 0.0:
            losses = [gan_loss(self.discriminators["field"](self._condition(masked, masks, member)), True,
                               loss_type=self.gan_type, is_disc=False) for member in out.member_mean]
            values["gen_adv_field"] = self.weights["adv_field"] * torch.stack(losses).mean()
        values["member_spread"] = out.secondary_var.sqrt().mean().detach()
        values["aleatoric_std"] = out.aleatoric_var.sqrt().mean().detach()
        values["predictive_std"] = out.predictive_var.sqrt().mean().detach()
        optimized = (
            "nll", "temporal_kl", "uncertainty_temporal", "kl", "pm",
            "gen_adv_field",
        )
        values["loss"] = sum((values[key] for key in optimized), zero)
        return values

    @staticmethod
    def _requires_grad(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled)

    def _l2_penalty_value(self) -> float:
        """Return the explicit value represented by Adam's coupled L2 term."""
        if self.generator_weight_decay <= 0.0:
            return 0.0
        with torch.no_grad():
            squared_norm = sum(
                parameter.detach().float().square().sum()
                for parameter in self.network.parameters()
                if parameter.requires_grad and parameter.ndim >= 2
            )
        return float(0.5 * self.generator_weight_decay * squared_norm)

    def _loss_manifest(self) -> Dict[str, Dict[str, Any]]:
        has_field_gan = "field" in self.discriminators and self.weights["adv_field"] > 0.0
        return {
            "nll": {
                "enabled": True,
                "weight": 1.0,
                "weighting": "fixed_main_objective",
            },
            "bayesian_kl": {
                "enabled": self.variant == "p2igan_bnn_joint",
                "weight": self.lambda_kl if self.variant == "p2igan_bnn_joint" else 0.0,
                "weighting": "independent_linear_warmup",
                "warmup_epochs": self.kl_warmup_epochs,
                "normalization": "train_events_x_outputs_per_event",
            },
            "temporal_kl": {
                "enabled": True,
                "weight": self.weights["temporal_kl"],
                "weighting": "fixed_auxiliary",
            },
            "sigma_temporal_smoothness": {
                "enabled": True,
                "weight": self.weights["uncertainty_temporal"],
                "weighting": "fixed_auxiliary",
            },
            "pm": {
                "enabled": self.weights["pm"] > 0.0,
                "weight": self.weights["pm"],
                "weighting": "independent_fixed",
            },
            "gan_field": {
                "enabled": has_field_gan,
                "weight": self.weights["adv_field"] if has_field_gan else 0.0,
                "weighting": "independent_fixed",
                "starts_at_step": 1,
            },
            "l2": {
                "enabled": self.generator_weight_decay > 0.0,
                "weight": self.generator_weight_decay,
            },
        }

    @staticmethod
    def _mlflow_loss_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
        logged = {
            f"loss/{prefix}/{mlflow_name}": float(metrics[key])
            for key, mlflow_name in MLFLOW_LOSS_NAMES.items()
            if key in metrics
        }
        if "l2_penalty" in metrics:
            logged[f"loss/{prefix}/l2_penalty"] = float(metrics["l2_penalty"])
        if "full_objective" in metrics:
            logged[f"loss/{prefix}/full_objective"] = float(metrics["full_objective"])
        return logged

    @staticmethod
    def _mlflow_epoch_metrics(split: str, metrics: Dict[str, float]) -> Dict[str, float]:
        logged = UncertaintyInterpolationTrainer._mlflow_loss_metrics(
            metrics, split
        )
        for key in UNCERTAINTY_METRIC_KEYS:
            if key in metrics:
                logged[f"uncertainty/{split}/{key}"] = float(metrics[key])
        for key in QUALITY_METRIC_KEYS:
            if key in metrics:
                logged[f"quality/{split}/{key}"] = float(metrics[key])
        for key in HEALTH_METRIC_KEYS:
            if key in metrics:
                logged[f"health/{split}/{key}"] = float(metrics[key])
        for key in (
            "epoch_seconds", "data_wait_seconds", "data_wait_fraction",
            "samples_per_second", "gpu_memory_gib",
        ):
            if key in metrics:
                logged[f"performance/{split}/{key}"] = float(metrics[key])
        return logged

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
        if not training and self.variant in {"p2igan_mc_dropout", "p2igan_condition_comod"}:
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
            disc_values = {"disc_field": target.new_zeros(())}
            disc_gradient_norm = 0.0
            if (
                training
                and self.disc_optimizer is not None
                and self.weights["adv_field"] > 0.0
            ):
                with self._autocast():
                    with torch.no_grad():
                        disc_out = self._forward(masked, masks, members)
                    disc_values = self._discriminator_losses(disc_out, target, masked, masks)
                disc_total = disc_values["disc_field"]
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
                logging.info("step=%d total=%.5f nll=%.5f kl=%.5f adv_field=%.5f",
                             self.global_step, totals["loss"] / batches, totals["nll"] / batches,
                             totals["kl"] / batches,
                             totals["gen_adv_field"] / batches)
                if mlflow.active_run():
                    scalar_values = {
                        key: float(value.detach())
                        for key, value in values.items()
                        if torch.is_tensor(value) and value.numel() == 1
                    }
                    step_metrics = self._mlflow_loss_metrics(
                        scalar_values, "step/train"
                    )
                    step_metrics.update({
                        "health/step/train/activation_abs_max": activation_stats["activation_abs_max"],
                        "health/step/train/activation_nonfinite_count": activation_stats["activation_nonfinite_count"],
                        "health/step/train/discriminator_gradient_norm": disc_gradient_norm,
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
        })
        result["l2_penalty"] = self._l2_penalty_value()
        result["full_objective"] = result["loss"] + result["l2_penalty"]
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
        if self.variant in {"p2igan_mc_dropout", "p2igan_condition_comod"}:
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
        loss_manifest = self._loss_manifest()
        active_losses = sorted(
            name for name, settings in loss_manifest.items() if settings["enabled"]
        )
        inactive_losses = sorted(
            name for name, settings in loss_manifest.items() if not settings["enabled"]
        )
        mlflow.set_tags({
            "model_variant": self.variant,
            "uncertainty_kind": getattr(self.network, "secondary_kind", "unknown"),
            "precision": precision,
            "dataset": Path(train_root).name if train_root else "unknown",
            "seed": str(self.cfg.get("seed", 42)),
            "run_group": "five-model-comparison",
            "active_losses": ",".join(active_losses),
            "inactive_losses": ",".join(inactive_losses),
            "status": "running",
        })
        mlflow.log_dict(self.cfg, "config/resolved_config.json")
        mlflow.log_dict(loss_manifest, "config/loss_manifest.json")
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
        tracking_uri = None
        if tracking.get("enabled", True) and tracking.get("experiment_name"):
            tracking_uri = configure_mlflow_storage(tracking)
        context = mlflow.start_run(run_name=tracking.get("run_name")) if tracking.get("enabled", True) else nullcontext()
        with context:
            if mlflow.active_run():
                mlflow.set_tag("tracking_backend", "sqlite")
                mlflow.set_tag("tracking_uri", tracking_uri or "")
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
                        "csi@0.1=%.4f predicted_rain=%.4f target_rain=%.4f "
                        "spread=%.6f predictive_std=%.6f",
                        epoch, train["loss"], valid["loss"], valid["rmse_hidden"],
                        valid["coverage_90"], valid["csi_0p1"],
                        valid["prediction_rain_fraction_0p1"],
                        valid["target_rain_fraction_0p1"],
                        valid["member_spread"], valid["predictive_std"],
                    )
                    target_rain_fraction = valid["target_rain_fraction_0p1"]
                    predicted_rain_fraction = valid["prediction_rain_fraction_0p1"]
                    if (
                        target_rain_fraction > 1e-4
                        and predicted_rain_fraction < 0.01 * target_rain_fraction
                    ):
                        logging.warning(
                            "Prediction-collapse warning at epoch=%d: rain fraction at "
                            "threshold 0.1 is %.6f versus target %.6f (CSI=%.6f)",
                            epoch,
                            predicted_rain_fraction,
                            target_rain_fraction,
                            valid["csi_0p1"],
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
                                self._mlflow_epoch_metrics(split, metrics),
                                step=epoch,
                            )
                        mlflow.log_metrics(
                            {
                                "loss/best/valid_backprop_total": self.best_loss,
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
