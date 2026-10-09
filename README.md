# P2I-GAN Uncertainty

以同一套 P2I 時空插值骨幹比較五種不確定性建模方法。所有模型保留相同輸入輸出格式，並預測降雨均值與逐像素 log-variance。

## 模型

| ID | Config | 不確定性來源 | 推理 raw members | 訓練目標 |
|---|---|---|---:|---|
| M1 | `p2igan_joint.json` | deterministic mean/logvar heads | 1 | NLL + Temporal KL + σ smoothness + GAN |
| M2 | `p2igan_decoder_ensemble.json` | 8 個 mean decoders，共用 logvar decoder | 8 | Poisson member NLL + Temporal KL + σ smoothness + GAN |
| M3 | `p2igan_mc_dropout.json` | MC Dropout | 16 | NLL + Temporal KL + σ smoothness + L2 + GAN |
| M4 | `p2igan_condition_comod.json` | MC Dropout + condition-routed 16 experts | 16 | NLL + Temporal KL + σ smoothness + L2 + GAN |
| M5 | `p2igan_bnn_joint.json` | Bayesian mean/logvar decoders | 16 | NLL + Temporal KL + σ smoothness + Bayesian KL warmup |

統一設定：

- NLL 為權重 `1.0` 的主要目標；Temporal KL 與 σ temporal smoothness 各使用固定權重 `0.0005`
- M3/M4 L2 weight decay：`1e-5`
- PM loss：全部關閉
- M1–M4 field GAN：`0.01`
- M5 GAN：關閉
- M5 Bayesian KL：自動除以訓練事件數與每事件輸出元素數，並線性 warmup
- 各模型依獨立顯存與計算特性訓練：M1/M2/M3/M4/M5 的實體 batch 分別為 `32/12/8/6/16`，皆不使用 gradient accumulation

模型輸出為：

```text
member_mean       [S, B, T, C, H, W]
member_logvar     [S, B, T, C, H, W]
mean              E[member_mean]
aleatoric_var     E[exp(member_logvar)]
secondary_var     Var(member_mean)
predictive_var    aleatoric_var + secondary_var
```

M1 的 `secondary_var` 為零；M2–M5 的 member spread 分別代表 decoder、dropout、condition/dropout 與 Bayesian weight 的差異。

## 安裝

需求為 Python 3.8–3.10。

```bash
pip install -U uv
uv sync
source .venv/bin/activate
```

資料位置、mask 與共用訓練參數設定在 `p2igan_bench/config/uncertainty_common.json`。開始訓練前請確認 `data.train`、`data.valid` 與 `data.test` 路徑。

## 訓練

```bash
nohup python scripts/train.py --config p2igan_bench/config/p2igan_joint.json \
  > logs/p2igan_joint.log 2>&1 &

nohup python scripts/train.py --config p2igan_bench/config/p2igan_decoder_ensemble.json \
  > logs/p2igan_decoder_ensemble.log 2>&1 &

nohup python scripts/train.py --config p2igan_bench/config/p2igan_mc_dropout.json \
  > logs/p2igan_mc_dropout.log 2>&1 &

nohup python scripts/train.py --config p2igan_bench/config/p2igan_condition_comod.json \
  > logs/p2igan_condition_comod.log 2>&1 &

nohup python scripts/train.py --config p2igan_bench/config/p2igan_bnn_joint.json \
  > logs/p2igan_bnn_joint.log 2>&1 &
```

每次啟動後可用 `echo $!` 查看 PID，並用 `tail -f logs/<model>.log` 追蹤進度。同一張 GPU 建議一次只執行一個模型；同時貼上五條命令會讓五個訓練程序競爭顯存與算力。

短程檢查：

```bash
python scripts/train.py \
  --config p2igan_bench/config/p2igan_joint.json \
  --smoke-test --smoke-batches 2
```

每個模型分別儲存 `best.pt` 與 `latest.pt` 至 config 指定的 `save_dir`。checkpoint 會檢查 `model.name` 與 `stack_version`，避免載入不相容架構。

## MLflow

MLflow 僅使用以下兩個儲存位置：

- `mlflow.db`：experiment、run、metric、parameter 與 tag。
- `mlartifacts/`：圖表、config 與 checkpoint 等 artifact。

```bash
.venv/bin/mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

請從專案根目錄執行這條指令。再以外部瀏覽器開啟 forwarded address（通常是 `http://127.0.0.1:5000`）

```bash
python scripts/cleanup_mlflow_runs.py
python scripts/cleanup_mlflow_runs.py --apply
```

## 推理

一次推理五個模型：

```bash
python scripts/infer.py \
  --config \
    p2igan_bench/config/p2igan_joint.json \
    p2igan_bench/config/p2igan_decoder_ensemble.json \
    p2igan_bench/config/p2igan_mc_dropout.json \
    p2igan_bench/config/p2igan_condition_comod.json \
    p2igan_bench/config/p2igan_bnn_joint.json \
  --output-dir inference/five_models \
  --overwrite
```

每個 Zarr event 包含：

```text
target, masked_input, mask
mean, aleatoric_std, secondary_std, predictive_std
member_mean, member_logvar, ensemble
```

預設由模型 raw members 與 aleatoric variance 產生 48 個具空間相關 noise 的 ensemble。使用 `--ensemble-size 0` 可只保留統計量與 raw members。

目前 test loader 固定 `batch_size=1`。`--predict-samples` 只會改變 M3、M4、M5 的抽樣數；M1 固定 1 個輸出，M2 固定使用 config 的 `num_heads`。

## License

MIT，見 `LICENSE`。
