"""Training diagnostics, calibration metrics, and qualitative figures."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


COVERAGE_Z = {
    "50": (0.50, 0.67448975),
    "80": (0.80, 1.28155157),
    "90": (0.90, 1.64485363),
    "95": (0.95, 1.95996398),
}
PRECIPITATION_THRESHOLDS = (0.10, 0.20, 0.40)

QUALITY_METRIC_KEYS = (
    "mae_hidden",
    "rmse_hidden",
    "target_mean_hidden",
    "prediction_mean_hidden",
    "target_rain_fraction_0p1",
    "prediction_rain_fraction_0p1",
    "error_std_correlation",
    "calibration_error",
    "coverage_90",
    "interval_width_90",
    *(f"csi_{str(value).replace('.', 'p')}" for value in PRECIPITATION_THRESHOLDS),
)


def _masked_mean(value: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    selected = selected.expand_as(value).to(value.dtype)
    count = selected.sum()
    if count.item() == 0:
        return value.mean()
    return (value * selected).sum() / count


def _selected_values(value: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    selected = selected.expand_as(value)
    chosen = value[selected]
    chosen = chosen if chosen.numel() else value.reshape(-1)
    # Quantiles over every 128x128x16 pixel are unnecessarily expensive. A
    # deterministic uniform subset is stable across runs and keeps monitoring
    # overhead bounded.
    maximum = 262_144
    if chosen.numel() > maximum:
        stride = max(1, chosen.numel() // maximum)
        chosen = chosen[::stride][:maximum]
    return chosen


@torch.no_grad()
def quality_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    predictive_var: torch.Tensor,
    observation_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute reconstruction, calibration, and rain-event metrics.

    ``observation_mask == 1`` denotes an observed input. Scientific metrics are
    therefore focused on the complementary interpolation region.
    """

    target = target.float()
    prediction = prediction.float()
    predictive_std = predictive_var.float().clamp_min(1e-12).sqrt()
    hidden = observation_mask < 0.5
    error = (prediction - target).abs()
    squared_error = error.square()

    metrics: Dict[str, torch.Tensor] = {
        "mae_hidden": _masked_mean(error, hidden),
        "rmse_hidden": _masked_mean(squared_error, hidden).sqrt(),
        "target_mean_hidden": _masked_mean(target, hidden),
        "prediction_mean_hidden": _masked_mean(prediction, hidden),
        "target_rain_fraction_0p1": _masked_mean(
            (target >= PRECIPITATION_THRESHOLDS[0]).float(), hidden
        ),
        "prediction_rain_fraction_0p1": _masked_mean(
            (prediction >= PRECIPITATION_THRESHOLDS[0]).float(), hidden
        ),
    }

    selected_error = _selected_values(error, hidden)
    selected_std = _selected_values(predictive_std, hidden)
    error_centered = selected_error - selected_error.mean()
    std_centered = selected_std - selected_std.mean()
    denominator = error_centered.square().sum().sqrt() * std_centered.square().sum().sqrt()
    metrics["error_std_correlation"] = torch.where(
        denominator > 1e-12,
        (error_centered * std_centered).sum() / denominator.clamp_min(1e-12),
        denominator.new_zeros(()),
    )
    nominal, z_value = COVERAGE_Z["90"]
    covered = error <= z_value * predictive_std
    coverage = _masked_mean(covered.float(), hidden)
    metrics["coverage_90"] = coverage
    metrics["interval_width_90"] = _masked_mean(
        2.0 * z_value * predictive_std,
        hidden,
    )
    metrics["calibration_error"] = (coverage - nominal).abs()

    for threshold in PRECIPITATION_THRESHOLDS:
        suffix = str(threshold).replace(".", "p")
        selected = hidden.expand_as(target)
        truth_event = (target >= threshold) & selected
        predicted_event = (prediction >= threshold) & selected
        hits = (truth_event & predicted_event).sum().float()
        misses = (truth_event & ~predicted_event).sum().float()
        false_alarms = (~truth_event & predicted_event & selected).sum().float()
        metrics[f"csi_{suffix}"] = hits / (hits + misses + false_alarms).clamp_min(1.0)

    return metrics


def tensor_finite_summary(tensors: Iterable[torch.Tensor]) -> Dict[str, float]:
    """Return output magnitude and non-finite count without retaining graphs."""

    maximum = 0.0
    nonfinite = 0
    for tensor in tensors:
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        nonfinite += int((~finite).sum().item())
        if finite.any():
            maximum = max(maximum, float(detached[finite].float().abs().max().item()))
    return {"output_abs_max": maximum, "nonfinite_count": float(nonfinite)}


class ActivationMonitor:
    """Sample top-level feature magnitudes only when explicitly enabled."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.enabled = False
        self.abs_max = 0.0
        self.nonfinite_count = 0
        self._handles = []
        for name, child in module.named_children():
            if name == "encoder" or "decoder" in name:
                self._handles.append(child.register_forward_hook(self._hook))

    @staticmethod
    def _tensors(value):
        if torch.is_tensor(value):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from ActivationMonitor._tensors(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from ActivationMonitor._tensors(item)

    def _hook(self, _module, _inputs, output) -> None:
        if not self.enabled:
            return
        for tensor in self._tensors(output):
            finite = torch.isfinite(tensor)
            self.nonfinite_count += int((~finite).sum().item())
            if finite.any():
                self.abs_max = max(
                    self.abs_max,
                    float(tensor.detach()[finite].float().abs().max().item()),
                )

    def start(self) -> None:
        self.abs_max = 0.0
        self.nonfinite_count = 0
        self.enabled = True

    def stop(self) -> Dict[str, float]:
        self.enabled = False
        return {
            "activation_abs_max": self.abs_max,
            "activation_nonfinite_count": float(self.nonfinite_count),
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _image(tensor: torch.Tensor, sample: int, timestep: int) -> torch.Tensor:
    return tensor[sample, timestep, 0].detach().float().cpu()


def prediction_figure(
    target: torch.Tensor,
    masked: torch.Tensor,
    output,
    *,
    sample: int = 0,
    timesteps: Sequence[int] = (0, 4, 8, 12, 15),
):
    """Build a fixed validation panel for mean/error/uncertainty inspection."""

    valid_steps = [min(max(0, int(step)), target.shape[1] - 1) for step in timesteps]
    rows = (
        ("Masked input", masked, "viridis"),
        ("Ground truth", target, "viridis"),
        ("Predicted mean", output.mean, "viridis"),
        ("Absolute error", (output.mean - target).abs(), "magma"),
        ("Aleatoric std", output.aleatoric_var.sqrt(), "magma"),
        (
            f"{output.secondary_kind.replace('_', ' ').title()} std",
            output.secondary_var.sqrt(),
            "magma",
        ),
        ("Predictive std", output.predictive_var.sqrt(), "magma"),
    )
    rain_max = max(1e-6, float(target[sample].detach().float().max().cpu()))
    error_max = max(
        1e-6,
        float((output.mean[sample] - target[sample]).detach().float().abs().max().cpu()),
    )
    uncertainty_max = max(
        1e-6,
        float(output.predictive_var[sample].detach().float().sqrt().max().cpu()),
    )
    figure, axes = plt.subplots(
        len(rows),
        len(valid_steps),
        figsize=(3.0 * len(valid_steps), 2.45 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (label, tensor, cmap) in enumerate(rows):
        image_max = rain_max if row_index < 3 else error_max if row_index == 3 else uncertainty_max
        last = None
        for column, timestep in enumerate(valid_steps):
            last = axes[row_index, column].imshow(
                _image(tensor, sample, timestep),
                cmap=cmap,
                vmin=0.0,
                vmax=image_max,
            )
            # ``set_axis_off()`` also hides the y-label. Keep the axis active
            # while removing ticks and spines so every row remains explicit.
            axes[row_index, column].tick_params(
                left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False,
            )
            for spine in axes[row_index, column].spines.values():
                spine.set_visible(False)
            if row_index == 0:
                axes[row_index, column].set_title(f"t={timestep}")
            if column == 0:
                axes[row_index, column].set_ylabel(
                    label,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=16,
                    fontweight="bold",
                )
        figure.colorbar(last, ax=axes[row_index, :], fraction=0.012, pad=0.01)
    return figure


def members_figure(
    output,
    *,
    sample: int = 0,
    timesteps: Sequence[int] = (0, 4, 8, 12, 15),
    max_members: int = 8,
):
    """Show whether stochastic/ensemble members are meaningfully different."""

    members = output.member_mean
    count = min(int(members.shape[0]), int(max_members))
    if count <= 1:
        return None
    valid_steps = [min(max(0, int(step)), members.shape[2] - 1) for step in timesteps]
    value_max = max(1e-6, float(members[:, sample].detach().float().max().cpu()))
    figure, axes = plt.subplots(
        count,
        len(valid_steps),
        figsize=(3.0 * len(valid_steps), 2.35 * count),
        squeeze=False,
        constrained_layout=True,
    )
    for member in range(count):
        for column, timestep in enumerate(valid_steps):
            axes[member, column].imshow(
                members[member, sample, timestep, 0].detach().float().cpu(),
                cmap="viridis",
                vmin=0.0,
                vmax=value_max,
            )
            axes[member, column].tick_params(
                left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False,
            )
            for spine in axes[member, column].spines.values():
                spine.set_visible(False)
            if member == 0:
                axes[member, column].set_title(f"t={timestep}")
            if column == 0:
                axes[member, column].set_ylabel(
                    f"Member {member + 1}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=16,
                    fontweight="bold",
                )
    return figure


def calibration_figure(
    target: torch.Tensor,
    prediction: torch.Tensor,
    predictive_var: torch.Tensor,
    observation_mask: torch.Tensor,
):
    """Build interval-coverage and predicted-std versus RMSE diagnostics."""

    error = (prediction.detach().float() - target.detach().float()).abs()
    std = predictive_var.detach().float().clamp_min(1e-12).sqrt()
    hidden = observation_mask.detach() < 0.5
    error_values = _selected_values(error, hidden).cpu()
    std_values = _selected_values(std, hidden).cpu()

    nominal = []
    empirical = []
    for _name, (level, z_value) in COVERAGE_Z.items():
        nominal.append(level)
        empirical.append(float((error_values <= z_value * std_values).float().mean()))

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], "--", color="0.5", label="ideal")
    axes[0].plot(nominal, empirical, marker="o", label="model")
    axes[0].set(
        xlabel="nominal interval coverage",
        ylabel="empirical hidden-region coverage",
        xlim=(0.45, 1.0),
        ylim=(0.45, 1.0),
        title="Interval calibration",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    quantiles = torch.quantile(std_values, torch.linspace(0, 1, 11))
    predicted, observed = [], []
    for index in range(10):
        upper_comparison = std_values <= quantiles[index + 1] if index == 9 else std_values < quantiles[index + 1]
        selected = (std_values >= quantiles[index]) & upper_comparison
        if selected.any():
            predicted.append(float(std_values[selected].mean()))
            observed.append(float(error_values[selected].square().mean().sqrt()))
    limit = max([1e-6, *predicted, *observed])
    axes[1].plot([0, limit], [0, limit], "--", color="0.5", label="ideal")
    axes[1].plot(predicted, observed, marker="o", label="model")
    axes[1].set(
        xlabel="mean predicted std",
        ylabel="empirical RMSE",
        xlim=(0, limit),
        ylim=(0, limit),
        title="Uncertainty reliability",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    return figure


__all__ = [
    "ActivationMonitor",
    "QUALITY_METRIC_KEYS",
    "calibration_figure",
    "members_figure",
    "prediction_figure",
    "quality_metrics",
    "tensor_finite_summary",
]
