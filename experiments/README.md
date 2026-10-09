# Uncertainty evaluation

The experiments evaluate the Zarr format written by `scripts/infer.py`. Each
event must contain `target`, `mask`, `mean`, `predictive_std`, and may also
contain `aleatoric_std`, `secondary_std`, `masked_input`, raw `member_mean`,
raw `member_logvar`, and `ensemble`. `epistemic_std` is retained as a semantic
alias for decoder-ensemble, MC-dropout, and condition-routed MC-dropout stores.

## Design

- **EXP-1 — scorecard:** MAE/RMSE/bias/SSIM, rainfall-threshold scores, FSS,
  predictive/aleatoric Gaussian NLL, ensemble CRPS, sampled multivariate Energy
  Score, spread-skill ratio, interval coverage/width/Winkler score, uncertainty
  decomposition, and event-bootstrap 95% confidence intervals.
- **EXP-2 — calibration:** coverage reliability, rank histogram,
  uncertainty–error alignment, and spatial case studies. Optional variance
  scaling is fitted only when a separate `calibration_path` is configured.
- **EXP-3 — stress and usefulness:** heavy-rain strata, bad-error AUROC/AUPRC,
  risk–coverage/AURC, aleatoric/secondary-spread alignment, and comparisons across any
  ablation or stress stores listed in `config.py`.

`mask=1` means observed. The default evaluation region is therefore `missing`,
which measures interpolation pixels rather than copying performance at gauges.
Ensembles are transformed member-by-member to mm/h before CRPS, coverage, and
spread are computed; this is necessary because the rain encoding is nonlinear.

## Run

First export inference results (the configured default is shown):

```bash
python scripts/infer.py \
  --config \
    p2igan_bench/config/p2igan_joint.json \
    p2igan_bench/config/p2igan_decoder_ensemble.json \
    p2igan_bench/config/p2igan_mc_dropout.json \
    p2igan_bench/config/p2igan_condition_comod.json \
  --output-dir inference/four_models \
  --overwrite
```

The evaluation manifest records model identity, uncertainty semantics, training
loss profile, expected raw-member count, test store, independent calibration
store, and scenario group. Edit
`experiments/configs/four_models.json` for real runs, then run:

```bash
python -m experiments.main --config experiments/configs/four_models.json
```

Before metrics are calculated, preflight checks model variant metadata, event
alignment, target/mask equality, and raw-member counts. Use `--check-only` when
validating a newly exported set of stores.

## Fake end-to-end integration

Generate four small Zarr stores with the exact inference contract—including
raw means/log-variances and architecture-specific secondary uncertainty—and
run all experiments:

```bash
python -m experiments.generate_fake_outputs --overwrite
python -m experiments.main --config experiments/configs/fake_four_models.json
```

Outputs are written under
`experiments/results/fake_four_model_integration/`. This exercises EXP-1/2/3,
calibration fitting, JSON/TSV reports, and every plot without requiring trained
checkpoints. Synthetic scores only validate plumbing and must never be reported
as model performance.

Never use the test output as `calibration_path`. For station-density or sensor
noise experiments, export a separate inference store for every condition and
give it a non-`in-domain` `group` in the configuration.
