# P2I-GAN uncertainty benchmark

This branch compares four uncertainty mechanisms on the same spatio-temporal
interpolation task. They share the P2I encoder, data split, conditional field
discriminator, training schedule, inference contract, and evaluation pipeline.
The architectural and statistical meaning of the uncertainty is deliberately
different in each model.

Current status: all four variants have passed 10 consecutive real-HDF5
forward/loss/backward/optimizer steps with the complete late-stage losses on an
RTX 4090. This verifies executable and numerically finite training paths, not
full-run convergence or final uncertainty calibration.

## Four model variants

| ID | Config / model name | Stochastic mechanism | Distribution loss | Secondary uncertainty |
|---|---|---|---|---|
| M1 | `p2igan_joint.json` / `p2igan_joint` | none; deterministic mean/log-variance heads | Gaussian NLL | none (aleatoric only) |
| M2 | `p2igan_decoder_ensemble.json` / `p2igan_decoder_ensemble` | four independently initialized complete decoder tails | member Gaussian NLL + empirical CRPS | between-decoder spread |
| M3 | `p2igan_mc_dropout.json` / `p2igan_mc_dropout` | repeated dropout-enabled passes through one dual-head model | member Gaussian NLL | MC-dropout spread |
| M4 | `p2igan_latent_comod.json` / `p2igan_latent_comod` | latent `z` drives bounded global/spatial co-modulation | Gaussian-mixture NLL + multiscale Energy Score | latent/style spread |

All variants predict a heteroscedastic log variance. The common generator
output is `UncertaintyOutput`:

```text
member_mean       [S,B,T,C,H,W]
member_logvar     [S,B,T,C,H,W]
mean              E_s[member_mean]
aleatoric_var     E_s[exp(member_logvar)]
secondary_var     Var_s(member_mean)
predictive_var    aleatoric_var + secondary_var
secondary_kind    none | between_decoder | mc_dropout | latent_style
```

Observed gauge pixels are copied back to the prediction when
`model.preserve_observations=true`.

### Stable residual stack (`stable_residual_v1`)

All four variants now use the same bounded decoder-stack rule:

```text
ResBlock(x) = x + 0.1 × DOConv₂(ReLU(DOConv₁(GroupNorm(x))))

x8 → residual stack → upsample
                    + x4 → residual stack → upsample
                                           + x2 → residual stack → upsample
                                                                  + x0 → residual stack
                                                                        → GroupNorm → head
```

The previous stack repeatedly added unnormalised residual branches and omitted
the `x2`/`x0` encoder skips. In an affected checkpoint, decoder activations grew
from below 1 to about `1.34e8`; the final sigmoid then saturated to an all-zero
mean. Pre-normalisation, residual scaling, restored multiscale skips, and output
normalisation keep the decoder signal bounded. M4 applies the same rule around
its bounded latent/spatial co-modulation blocks.

`model.stack_version`, `residual_scale`, and `residual_norm_groups` are stored
in every checkpoint. Training and inference reject incompatible legacy
checkpoints. New checkpoints are isolated under
`weights/stable_residual_v1/<model>/`, so existing weights are preserved and
cannot be resumed accidentally.

### Does predicted standard deviation enter the GAN?

For M1, yes. It has two discriminators:

- The field discriminator receives `(masked input, mask, target-or-mean)` and
  enforces realistic space-time fields.
- The joint discriminator receives
  `(masked input, mask, target-or-mean, uncertainty)`. The fake uncertainty is
  predicted standard deviation. The real-side uncertainty proxy is a detached,
  locally smoothed absolute residual. This gives the uncertainty map temporal
  and spatial adversarial supervision without pretending that a true sigma map
  exists in the dataset.

The residual proxy is a practical training target, not uncertainty ground
truth. For publication-grade calibration, replace it with out-of-fold residual
maps produced by a frozen teacher or cross-validation predictions.

M2–M4 feed every generated field member to the field discriminator. Their
uncertainty comes from member spread plus predicted aleatoric variance, so a
separate joint uncertainty discriminator is not used by default.

## Losses and schedule

The common deterministic anchor is precipitation-weighted L1. Distributional
training then adds the model-specific proper score. PM tail matching and GAN
loss start later to avoid destabilizing variance learning.

Default schedule (fractions of total epochs):

1. `0–10%`: weighted L1 warm-up.
2. `10–40%`: Gaussian NLL or Gaussian-mixture NLL.
3. `40–70%`: linearly ramp conditional field GAN.
4. `70–100%`: PM plus M1 joint GAN/temporal sigma, M2 CRPS, or M4 multiscale Energy Score.

The multiscale Energy Score evaluates each generated spatio-temporal field as
one multivariate sample at native, 4× pooled, and 8× pooled scales. Its first
term rewards samples close to the observation; its pairwise term rewards useful
ensemble spread. Unlike pixel-wise NLL, it can train spatial dependence and
coherent scenario diversity.

Training logs each loss separately so ablations remain auditable. The compact
scientific scorecard keeps hidden-region MAE/RMSE, CSI at normalized rain
thresholds 0.1/0.2/0.4, 90% interval coverage/width, calibration error, and
error–uncertainty correlation. It also records gradient/activation magnitudes,
learning rates, numerical-health counts, phase state, throughput, and GPU
memory. Step metrics use the `step/` prefix; complete validation summaries are
recorded once per epoch.

## Configuration

Shared data, optimizer, schedule, and default loss values live in
[`uncertainty_common.json`](p2igan_bench/config/uncertainty_common.json). The
four named configs use the repository's lightweight `extends` merge:

- [`p2igan_joint.json`](p2igan_bench/config/p2igan_joint.json)
- [`p2igan_decoder_ensemble.json`](p2igan_bench/config/p2igan_decoder_ensemble.json)
- [`p2igan_mc_dropout.json`](p2igan_bench/config/p2igan_mc_dropout.json)
- [`p2igan_latent_comod.json`](p2igan_bench/config/p2igan_latent_comod.json)

Legacy DK/STDK, position-ablation, simple fallback, baseline configs, and the
old `p2igan_uncertainty.json` have been removed. The model registry now accepts
only the four named variants and raises an error for unknown names.

The shared config currently maps:

```text
train -> .../NIMROD_Train_datasets_sti/train  (3885 events)
valid -> .../NIMROD_Train_datasets_sti/test   (971 events)
test  -> .../NIMROD_Train_datasets_sti/val    (20 long events)
```

This naming follows the existing dataset layout. Confirm the split policy and
gauge-mask path before publishing results.

## Installation

```bash
pip install -U uv
uv venv .venv
source .venv/bin/activate
uv sync
```

## Recommended workflow

```text
edit uncertainty_common.json
        ↓
smoke-test all four configs
        ↓
train four independent checkpoints
        ↓
infer four model-specific Zarr stores
        ↓
experiment preflight
        ↓
EXP-1 scorecard → EXP-2 calibration → EXP-3 stress/usefulness
```

For a plumbing-only check that needs no checkpoints:

```bash
python -m experiments.generate_fake_outputs --overwrite
python -m experiments.main --config experiments/configs/fake_four_models.json
```

## Training

The default configs are tuned on one RTX 4090 (24 GB):

| Model | Configured batch | Train members |
|---|---:|---:|
| M1 joint | 16 | 1 |
| M2 decoder ensemble | 8 | 4 fixed heads |
| M3 MC dropout | 16 | 4 |
| M4 latent CoMod | 16 | 4 |

The common precision is BF16 AMP: it retains FP16-like activation memory while
avoiding FP16's narrow numeric range. M1 overrides this to FP32 because its
DOConv backward was intermittently non-finite under mixed precision and it is
small enough to fit comfortably (`~7.0 GiB` in the 10-step stress test). M2,
M3, and M4 passed the same BF16 test at approximately 10.7, 17.2, and 13.9 GiB.
Fused Adam, TF32, cuDNN autotuning, non-blocking pinned-memory transfers,
`set_to_none` gradient clearing, and gradient clipping remain enabled. Eight
persistent DataLoader workers with three prefetched batches gave the best
measured throughput on the current host.

The HDF5 loader reads only the requested temporal window and 128×128 center
crop instead of loading every complete 1024×1024 event. The static text gauge
mask is parsed once in the parent process and reused by workers. These two
changes prevent storage and mask parsing from starving the GPU.

M2 uses a lower generator learning rate (`2e-5`): at `1e-4`, its four
independent mean/log-variance tails showed rapidly increasing NLL even though
memory usage was safe. Batch sizes are hardware-specific; lower them first if
resolution, member count, or crop size increases. `train.accumulate_steps`
can recover the desired effective batch size when necessary.

```bash
python scripts/train.py --config p2igan_bench/config/p2igan_joint.json
python scripts/train.py --config p2igan_bench/config/p2igan_decoder_ensemble.json
python scripts/train.py --config p2igan_bench/config/p2igan_mc_dropout.json
python scripts/train.py --config p2igan_bench/config/p2igan_latent_comod.json
```

Before a long run, exercise a complete real-data optimizer step—including the
late-stage GAN and proper-score terms—with:

```bash
python scripts/train.py \
  --config p2igan_bench/config/p2igan_joint.json \
  --smoke-test --smoke-batches 3
```

`SMOKE_OK` reports elapsed time, loss, and peak allocated GPU memory without
writing a checkpoint or MLflow run. Epoch MLflow records also include
`epoch_seconds`, `data_wait_seconds`, `data_wait_fraction`, and
`samples_per_second` for identifying future data-loading regressions.

Each config has a unique versioned `save_dir`, MLflow run name, `best.pt`, and
`latest.pt`, preventing checkpoints from different architectures or residual
stack versions from being mixed accidentally.

### MLflow monitoring

Start the local UI with:

```bash
mlflow ui --backend-store-uri "file:$(pwd)/mlruns" --port 5041
```

Open `http://127.0.0.1:5041`. For remote SSH access, forward the port with
`ssh -L 5041:127.0.0.1:5041 USER@HOST`.

The `monitoring` block in the shared config controls qualitative artifacts and
numerical safeguards. The first epoch of each run and every `visualize_every`
epochs log fixed validation panels containing masked input, target, predicted mean, absolute
error, aleatoric/member/predictive uncertainty, raw stochastic members, and an
interval/reliability calibration figure. The fixed batch makes progress across
epochs visually comparable.

Resolved config, model summary, model/run tags, best-checkpoint metadata, and
the final best checkpoint are stored as artifacts. A NaN/Inf in inputs,
outputs, losses, or gradients immediately marks the MLflow run failed and
writes `debug_nonfinite_*.pt` instead of silently overwriting a healthy
checkpoint. SIGINT/SIGTERM writes `interrupted.pt` and marks the run
interrupted.

To limit monitoring overhead, the full scientific scorecard is computed on
every validation batch but only every `train.log_step` steps and at the final
batch on the training split. `latest.pt` is written on the first epoch, every
`checkpoint_every` epochs, and the final epoch; `best.pt` requires an
improvement larger than `best_min_delta`.

Runs terminated by `kill -9` cannot execute the signal handler and may remain
falsely marked RUNNING. Preview records older than 24 hours, then explicitly
apply the cleanup if the list is correct:

```bash
python scripts/cleanup_mlflow_runs.py
python scripts/cleanup_mlflow_runs.py --apply
```

Run all four smoke tests before launching long jobs:

```bash
for model in joint decoder_ensemble mc_dropout latent_comod; do
  python scripts/train.py \
    --config "p2igan_bench/config/p2igan_${model}.json" \
    --smoke-test --smoke-batches 2
done
```

## Inference and Zarr outputs

One command can infer all four trained configs and writes one Zarr store per
model:

```bash
python scripts/infer.py \
  --config \
    p2igan_bench/config/p2igan_joint.json \
    p2igan_bench/config/p2igan_decoder_ensemble.json \
    p2igan_bench/config/p2igan_mc_dropout.json \
    p2igan_bench/config/p2igan_latent_comod.json \
  --output-dir inference/four_models \
  --overwrite
```

Inference rejects checkpoints whose saved `model.name` or `model.stack_version`
differs from the active config. Set `--ensemble-size 0` when only summary maps
and raw model members are needed; this reduces Zarr size without discarding
model-native samples.

This produces:

```text
inference/four_models/
├── p2igan_joint.zarr
├── p2igan_decoder_ensemble.zarr
├── p2igan_mc_dropout.zarr
└── p2igan_latent_comod.zarr
```

Every event stores `target`, `masked_input`, `mask`, `mean`, `aleatoric_std`,
`secondary_std`, `predictive_std`, raw `member_mean`, raw `member_logvar`, and
sampled `ensemble`. M2/M3 also expose `epistemic_std`; M4 exposes `style_std`;
M1 uses `proxy_secondary_std` (normally zero). The semantic name matters:
latent diversity is not automatically calibrated epistemic uncertainty.

For a single config, explicit checkpoint and output paths are also supported:

```bash
python scripts/infer.py \
  --config p2igan_bench/config/p2igan_mc_dropout.json \
  --checkpoint weights/stable_residual_v1/p2igan_mc_dropout/best.pt \
  --output inference/mc_dropout_ablation.zarr \
  --overwrite
```

## Evaluation

The `experiments/` pipeline compares point accuracy, probabilistic calibration,
proper scores, spatial structure, and uncertainty usefulness. Keep raw members: CRPS,
Energy Score, rank histograms, coverage, spread-skill analysis, and diversity
diagnostics cannot be reconstructed reliably from mean/std maps alone.

Experiment methods and loss/uncertainty semantics are configured in
[`experiments/configs/four_models.json`](experiments/configs/four_models.json).
An end-to-end synthetic contract test is available with:

```bash
python -m experiments.generate_fake_outputs --overwrite
python -m experiments.main --config experiments/configs/fake_four_models.json
```

For real stores, validate contracts before spending time on plots and
bootstrapping:

```bash
python -m experiments.main \
  --config experiments/configs/four_models.json \
  --check-only

python -m experiments.main \
  --config experiments/configs/four_models.json
```

The manifest records model identity, secondary-uncertainty semantics, loss
profile, expected raw-member count, scenario group, and optional independent
calibration store. Never use the evaluated test store as `calibration_path`.

## Interpretation limits

- M1 estimates aleatoric uncertainty; its residual map is a discriminator-side
  training proxy, not uncertainty ground truth.
- M2/M3 member spread is treated as an epistemic proxy, but still requires
  coverage and spread-skill validation.
- M4 style spread measures conditional scenario diversity. It should not be
  labelled calibrated epistemic uncertainty without supporting evidence.
- Compare all models on identical event keys, masks, target arrays, output
  scale, and ensemble evaluation protocol. Experiment preflight enforces these
  invariants.

Detailed design rationale is in
[`discuss/four_model_best_architecture_and_loss.html`](discuss/four_model_best_architecture_and_loss.html).
The implementation, performance, Zarr, and experiment integration report is in
[`discuss/four_model_implementation_report.html`](discuss/four_model_implementation_report.html).

## License

MIT. See `LICENSE`.
