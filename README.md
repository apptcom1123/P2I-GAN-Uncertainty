

## P2IGAN-MU : Mixture-of-Experts-Modulated Uncertainty-Aware GAN for Diverse and Reliable Spatio-Temporal Interpolation


This branch is now centered on uncertainty-aware spatio-temporal interpolation rather than the original GAN-vs-baseline benchmark workflow.
The current implementation keeps the P2IGAN interpolation backbone, adds a heteroscedastic uncertainty head, uses Monte Carlo dropout sampling to estimate epistemic uncertainty, and restores adversarial training only on the reconstructed interpolation result.
Inference now exports:

- predictive mean
- aleatoric standard deviation
- epistemic standard deviation
- total predictive standard deviation
- correlated ensemble members

This is conceptually related with the DEUCE paper, but adapted for interpolation. The running method is not a full variational-inference Bayesian neural network. It is an engineering-stable approximation based on:

- heteroscedastic mean / log-variance prediction
- MC-dropout stochastic forward sampling
- discriminator supervision only on reconstructed mean sequences
- optional expert-modulated decoder upsampling via `UPPosMod`

⭐ If P2IGAN is helpful to your projects, please help star this repo. Thanks! 🤗

---

</div>

## Update
- **2026.04.22**: switched the uncertainty branch to a stable interpolation-oriented pipeline with heteroscedastic uncertainty, MC dropout, multi-sample discriminator updates, MLflow logging, and optional `UPPosMod` expert modulation.


### TODO
- [x] make model output with ensemble.
- [x] make model output uncertainty decomposition.
- [x] restore discriminator training on reconstruction only.
- [x] support modulation-based decoder upsampling with configurable experts.
- [ ] improve calibration and stabilize predicted aleatoric variance.
- [ ] revisit full variational inference once the custom DOConv backbone is made compatible.


## Dependencies and Installation

1. Clone Repo

   ```bash
   git clone https://github.com/NTU-CompHydroMet-Lab/P2I-GAN-benchmark
   ```

2. Create Conda Environment and Install Dependencies

   ```bash
   # install uv if it is not already available
   pip install -U uv
   
   # direct to project folder
   cd P2I-GAN-benchmark

   # create and activate a local virtual environment
   uv venv .venv
   source .venv/bin/activate

   # install project dependencies from pyproject.toml / uv.lock
   uv pip install -e .
   uv sync
   ```
 
## Get Started


### Prepare pretrained models
Download our pretrained models from [Releases V0.1.0](https://drive.google.com/file/d/1oPkmll4_5NlkVTdDr57Bhr4UbWi-Fruw/view?usp=drive_link)

The directory structure will be arranged as:
```
weights
   |- test
      |- P2IGANv0.1.0.pt
```

### Inference with Fake Data

Since we cannot share the original Nimrod (radar) or MIDAS (gauge) datasets with you, we provide a small fake dataset instead. You can just use code here to see the result.

```bash
python scripts/infer.py \
  --config p2igan_bench/config/p2igan_baseline_eval.json \
  --experiment-name p2igan-eval-fakedata
```

## Training
The uncertainty-oriented config is provided in [`p2igan_uncertainty.json`](./p2igan_bench/config/p2igan_uncertainty.json).

### Current training behavior
The current training loop is centered on [`scripts/train.py`](./scripts/train.py) and behaves as follows:

- The generator outputs `mean` and `logvar`.
- `mean` is the reconstructed interpolation sequence.
- `logvar` is used by the heteroscedastic Gaussian negative log-likelihood to learn aleatoric uncertainty.
- MC dropout remains active during stochastic sampling, so repeated forward passes produce multiple predictions for epistemic estimation.
- The discriminator only sees `mean prediction` versus `ground-truth sequence`. It does not see aleatoric maps, epistemic maps, or predictive std maps.
- The discriminator is updated with multi-sample fake sequences. For one batch, `sample_count` stochastic predictions are generated, each fake-vs-real discriminator loss is computed independently, and the discriminator loss is averaged before the optimizer step.
- The generator is trained with `reconstruction loss + adversarial loss`, where adversarial supervision is applied only to the reconstructed mean sequence.

### Decoder modulation
If `model.mod = true`, the mean decoder switches from `UPPos` to `UPPosMod`.

- `x8` is flattened and passed through a 4-layer MLP to produce `(B, 3 * number_of_experts)`.
- A uniform random noise tensor in `[-1, 1]` is added with strength `model.mod_noise_rate`.
- The resulting gate vector is split into three parts and passed to the three decoder upsampling blocks.
- Each `UPPosMod` maintains an expert bank of learnable positional tensors and mixes them with the gate weights to produce the modulation used in `x = x + x * pos`.
- At the moment, this modulation path is applied to the `mean_decoder` only. The `logvar_decoder` remains on the original `UPPos` path for stability.

### Important config flags
The most important knobs in [`p2igan_uncertainty.json`](./p2igan_bench/config/p2igan_uncertainty.json) are:

- `uncertainty.train_samples`: number of stochastic forward passes per training batch
- `uncertainty.val_samples`: number of stochastic forward passes per validation batch
- `uncertainty.predict_samples`: number of MC samples used during inference
- `uncertainty.ensemble_size`: number of exported ensemble members
- `loss.adversarial_weight`: generator adversarial loss weight
- `discriminator.enabled`: whether the discriminator branch is used
- `model.mod`: whether to enable `UPPosMod`
- `model.mod_noise_rate`: noise strength for the modulation gate
- `model.number_of_experts`: number of positional experts per modulation block

### Logging and visualization
Training now records MLflow metrics and artifacts:

- step losses for total / reconstruction / generator adversarial / discriminator
- epoch losses for train and validation
- generator and discriminator learning rates
- latest and best checkpoints
- visualization artifacts containing masked input, target, predictive mean, aleatoric std, and epistemic std

### Commands
Run one of the following commands for training:
```shell
# Train uncertainty-aware interpolation model
python scripts/train.py --config p2igan_bench/config/p2igan_uncertainty.json

# Export predictive mean / std maps / ensemble members
python scripts/infer.py \
  --config p2igan_bench/config/p2igan_uncertainty.json \
  --checkpoint weights/p2igan_uncertainty/best.pt \
  --overwrite

```

## License
This project is licensed under the MIT License — see the LICENSE file for details.


## Contact
If you have any questions about the technical issues, please feel free to reach me out at r13521608@caece.net. 
