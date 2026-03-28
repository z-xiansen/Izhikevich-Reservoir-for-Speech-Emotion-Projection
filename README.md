# Izhikevich Reservoir for Speech Emotion Projection (TESS)

本项目实现了一个基于 Izhikevich 神经元模型的随机储备池系统，用于语音情绪识别（SER）的状态空间可视化。
重点是观察不同情绪样本在高维神经动态状态上的自发分离能力，而不是训练分类器。

## 功能覆盖

- 自动下载 TESS 数据集（Kaggle 优先，HuggingFace 镜像兜底）
- `librosa` 提取 40 维 MFCC
- Direct Coding：将归一化 MFCC 直接作为 `I_app` 注入输入神经元
- 20x20 储备池（400 神经元），80% RS + 20% FS 参数异质性
- 距离衰减随机连接 + 局部抑制拓扑
- Izhikevich 动力学与阈值重置机制
- STDP（双指数痕迹）与 STP（TM：`x/u`）
- 内在塑性（IP）调节偏置电流，维持边际混沌附近活性
- 基于状态向量的 t-SNE 降维可视化

## 项目结构

```text
.
├─ run_ser_experiment.py
├─ src/ser_reservoir
│  ├─ config.py
│  ├─ data.py
│  ├─ reservoir.py
│  ├─ analysis.py
│  └─ pipeline.py
├─ data/
│  ├─ raw/
│  └─ processed/
└─ results/
```

## 环境与安装（虚拟环境）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Kaggle 凭证（可选但推荐）

若你有 Kaggle API key，可配置后优先走 Kaggle 下载：

```powershell
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_api_key"
```

若 Kaggle 不可用，代码会自动回退到 HuggingFace 的 parquet 镜像并解包为 wav。

## 运行实验

快速运行（默认 420 条样本，均衡采样）：

```powershell
.\.venv\Scripts\python .\run_ser_experiment.py
```

全量样本运行：

```powershell
.\.venv\Scripts\python .\run_ser_experiment.py --no-limit
```

调参示例：

```powershell
.\.venv\Scripts\python .\run_ser_experiment.py --max-samples 560 --max-per-emotion 80 --frame-repeat 2 --input-gain 14
```

保存外部模型文件：

```powershell
.\.venv\Scripts\python .\run_ser_experiment.py --save-model
```

加载已有模型继续运行，并另存一份新的模型文件：

```powershell
.\.venv\Scripts\python .\run_ser_experiment.py --load-model results/reservoir_model.npz --save-model --model-path results/reservoir_model_next.npz
```

## 输出

- `results/tsne_emotion_clusters.png`：2D 情绪状态投影图
- `data/processed/reservoir_states.npz`：高维状态、嵌入、标签、路径
- `results/run_summary.yaml`：运行摘要与网络统计
- `results/reservoir_model.npz`：可选，保存网络结构、突触权重、输入映射与内在偏置（通过 `--save-model` 生成）

## 说明

- 当前实现默认保留跨样本的长期塑性（STDP + IP），并在每个样本前重置短期动力学状态。
- 若传入 `--load-model`，程序会直接加载外部模型文件，不再重新随机生成网络结构与突触连接；若不传，则从随机初始化的新模型开始运行。
- 模型文件会保存关键动力学/可塑性配置的元信息，避免把不兼容的配置静默加载到旧模型上。
- 若你后续需要接入岭回归或线性探针，可直接使用 `reservoir_states.npz` 的 `states` 与 `labels`。
