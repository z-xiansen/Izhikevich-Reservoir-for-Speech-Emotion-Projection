# Izhikevich Reservoir for Speech Emotion Projection (TESS)

本项目实现了一个基于 Izhikevich 神经元模型的随机储备池系统，用于语音情绪识别（SER）任务的特征提取与分类。
通过模拟生物类脑的神经动态，验证储备池学习在情绪识别中的可行性与有效性。

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
- 基于 800 维状态向量的情绪分类器训练与验证（按情绪分层 80/20 切分）

## 分类器说明

当前项目默认使用一个轻量级基线分类器，对储备池输出的 `800` 维状态向量进行情绪分类：

- 先使用 `StandardScaler` 对特征做标准化
- 再使用 `LogisticRegression` 进行多分类训练

这样设计的目的不是引入一个复杂分类头，而是先验证储备池提取出的状态特征本身是否具有良好的情绪可分性。
对于当前这种固定长度、中高维的状态向量，逻辑回归训练速度快、结果稳定、解释也比较直接，适合作为 baseline。

分类器的训练数据来自 `reservoir_states.npz` 中保存的：

- `states`：每条语音对应的 `800` 维特征
- `labels`：情绪标签

默认按情绪分层做 `80%` 训练、`20%` 验证。

## 项目结构

```text
.
├─ run_ser_experiment.py
├─ train_classifier_from_states.py
├─ src/ser_reservoir
│  ├─ config.py
│  ├─ data.py
│  ├─ reservoir.py
│  ├─ analysis.py
│  ├─ classifier.py
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

### 这些 `--xxx` 命令是怎么来的

本项目中的 `--no-limit`、`--save-model`、`--load-model` 这类写法，都是 Python 脚本的命令行参数。
它们由 `argparse` 定义，来源分别是：

- `run_ser_experiment.py`：主实验入口，负责数据采样、MFCC 提取、储备池仿真、t-SNE 可视化、分类器训练、结果保存
- `train_classifier_from_states.py`：独立分类器入口，只读取已经保存好的 `reservoir_states.npz` 来训练分类器

如果你想先查看脚本支持哪些参数，可以直接运行：

```powershell
.\.venv\Scripts\python .\run_ser_experiment.py --help
.\.venv\Scripts\python .\train_classifier_from_states.py --help
```

说明：

- 形如 `--no-limit`、`--save-model`、`--quiet` 这种参数是开关，写上就表示启用，不需要再跟 `true/false`
- 形如 `--seed 7`、`--model-path results/x.npz` 这种参数后面需要跟值

### 主实验脚本：`run_ser_experiment.py`

这个脚本会完成一整套流程：

- 采样音频
- 提取 MFCC
- 生成每条语音的 `800` 维储备池状态向量
- 生成 t-SNE 投影图
- 基于状态向量训练分类器
- 保存 summary、分类器指标、混淆矩阵等结果

最常用命令如下。

快速运行，使用默认采样限制：

```powershell
.\.venv\Scripts\python .\run_ser_experiment.py
```

默认配置含义：

- 总样本数最多 `420`
- 每种情绪最多 `70`
- 会训练分类器
- 不会保存储备池模型文件

全量样本运行：

```powershell
.\.venv\Scripts\python .\run_ser_experiment.py --no-limit
```

全量 TESS 下当前数据规模是：

- 每种情绪 `400` 条
- 共 `2800` 条
- 分类器自动按每类 `320` 条训练、`80` 条验证

运行结束后，终端会打印：

- 一份更易读的分类报告
- 一份完整的 JSON 运行摘要

常见命令组合：

```powershell
.\.venv\Scripts\python .\run_ser_experiment.py --save-model
.\.venv\Scripts\python .\run_ser_experiment.py --no-limit --save-model
.\.venv\Scripts\python .\run_ser_experiment.py --no-limit --load-model results/reservoir_model.npz
.\.venv\Scripts\python .\run_ser_experiment.py --load-model results/reservoir_model.npz --save-model --model-path results/reservoir_model_next.npz
.\.venv\Scripts\python .\run_ser_experiment.py --max-samples 560 --max-per-emotion 80 --frame-repeat 2 --input-gain 14
.\.venv\Scripts\python .\run_ser_experiment.py --no-limit --no-classifier
.\.venv\Scripts\python .\run_ser_experiment.py --no-limit --quiet
```

各参数说明如下。

`--seed`

- 随机种子
- 控制样本抽样、网络初始化、分类器切分等随机过程
- 默认值为 `7`
- 示例：`--seed 7`

`--max-samples`

- 限制总样本数
- 默认值为 `420`
- 只有在未使用 `--no-limit` 时才生效
- 示例：`--max-samples 560`

`--max-per-emotion`

- 限制每种情绪最多抽取多少条样本
- 默认值为 `70`
- 只有在未使用 `--no-limit` 时才生效
- 示例：`--max-per-emotion 80`

`--no-limit`

- 取消 `--max-samples` 和 `--max-per-emotion` 的限制
- 启用后直接使用当前数据集中全部可用样本
- 示例：`--no-limit`

`--frame-repeat`

- 每一帧 MFCC 在储备池中重复注入的次数
- 会影响时间展开长度和神经动态表现
- 默认值为 `2`
- 示例：`--frame-repeat 2`

`--input-gain`

- 输入电流增益，控制 MFCC 注入神经元时的强度
- 默认值为 `14.0`
- 示例：`--input-gain 14`

`--save-model` 或 `--save-checkpoint`

- 运行结束后保存当前储备池模型
- 两个写法是同一个参数的别名，效果相同
- 示例：`--save-model`

`--model-path` 或 `--checkpoint-path`

- 指定储备池模型保存路径
- 默认值为 `results/reservoir_model.npz`
- 示例：`--model-path results/reservoir_model_next.npz`

`--load-model` 或 `--load-checkpoint`

- 加载已有储备池模型，而不是重新随机生成网络结构和权重
- 两个写法是同一个参数的别名，效果相同
- 示例：`--load-model results/reservoir_model.npz`

`--no-classifier`

- 跳过分类器训练
- 仍然会保留状态提取、t-SNE、summary 等原有流程
- 示例：`--no-classifier`

`--quiet`

- 减少过程输出
- 适合批量运行或不想看进度信息时使用
- 示例：`--quiet`

### 独立分类器脚本：`train_classifier_from_states.py`

这个脚本不会重新跑储备池仿真。
它只读取已经保存的 `data/processed/reservoir_states.npz`，然后基于其中的：

- `states`
- `labels`
- `paths`

重新训练分类器并输出结果。

最常用命令：

```powershell
.\.venv\Scripts\python .\train_classifier_from_states.py --state-file data/processed/reservoir_states.npz
```

这个脚本适合下面这些场景：

- 你已经跑完储备池，不想再次重复仿真
- 你只想重新切分训练集/验证集并训练分类器
- 你想把分类器输出到新的文件名

常见命令组合：

```powershell
.\.venv\Scripts\python .\train_classifier_from_states.py --state-file data/processed/reservoir_states.npz
.\.venv\Scripts\python .\train_classifier_from_states.py --state-file data/processed/reservoir_states.npz --validation-ratio 0.25
.\.venv\Scripts\python .\train_classifier_from_states.py --state-file data/processed/reservoir_states.npz --model-path results/my_classifier.joblib --metrics-path results/my_metrics.yaml
```

各参数说明如下。

`--state-file`

- 指定输入的 `reservoir_states.npz` 文件
- 默认值为 `data/processed/reservoir_states.npz`
- 示例：`--state-file data/processed/reservoir_states.npz`

`--seed`

- 控制训练集/验证集切分的随机性
- 默认值为 `7`
- 示例：`--seed 7`

`--validation-ratio`

- 指定每种情绪中拿出多少比例作为验证集
- 默认值为 `0.2`
- 例如 `0.2` 表示每类 `80%` 用于训练，`20%` 用于验证
- 示例：`--validation-ratio 0.25`

`--model-path`

- 指定分类器模型保存路径
- 默认值为 `results/reservoir_classifier.joblib`

`--metrics-path`

- 指定分类器指标保存路径
- 默认值为 `results/classifier_metrics.yaml`

`--confusion-matrix-path`

- 指定混淆矩阵图片保存路径
- 默认值为 `results/classifier_confusion_matrix.png`

`--predictions-path`

- 指定验证集逐样本预测 CSV 的保存路径
- 默认值为 `results/classifier_validation_predictions.csv`

### 推荐使用顺序

如果你是第一次跑项目，建议按下面顺序：

1. 先运行主实验脚本，生成状态向量、t-SNE 图和基础分类结果
2. 再根据需要，用独立分类器脚本重复训练分类器，而不重复跑储备池

对应命令：

```powershell
.\.venv\Scripts\python .\run_ser_experiment.py --no-limit
.\.venv\Scripts\python .\train_classifier_from_states.py --state-file data/processed/reservoir_states.npz
```

## 输出

- `results/tsne_emotion_clusters.png`：2D 情绪状态投影图
- `data/processed/reservoir_states.npz`：高维状态、嵌入、标签、路径
- `results/run_summary.yaml`：运行摘要与网络统计
- `results/reservoir_classifier.joblib`：基于 800 维状态向量训练得到的分类器
- `results/classifier_metrics.yaml`：训练/验证划分、准确率、Macro-F1、逐类指标、混淆矩阵
- `results/classifier_confusion_matrix.png`：验证集混淆矩阵可视化
- `results/classifier_validation_predictions.csv`：验证集逐样本预测结果
- `results/reservoir_model.npz`：可选，保存网络结构、突触权重、输入映射与内在偏置（通过 `--save-model` 生成）

## 说明

- 当前实现默认保留跨样本的长期塑性（STDP + IP），并在每个样本前重置短期动力学状态。
- 分类器默认使用当前运行得到的 `reservoir_states.npz` 同源状态向量进行训练，先做标准化，再训练多分类逻辑回归。
- 若传入 `--load-model`，程序会直接加载外部模型文件，不再重新随机生成网络结构与突触连接；若不传，则从随机初始化的新模型开始运行。
- 模型文件会保存关键动力学/可塑性配置的元信息，避免把不兼容的配置静默加载到旧模型上。
- 若你后续需要接入岭回归或线性探针，可直接使用 `reservoir_states.npz` 的 `states` 与 `labels`。

## 运行结果报告

下面给出一次全量运行的示例结果。
该次运行使用了完整 TESS 数据集，共 `2800` 条语音样本，每种情绪 `400` 条；分类器按情绪分层切分为每类 `320` 条训练、`80` 条验证。
从结果看，储备池生成的 `800` 维状态向量具有很强的情绪区分能力，验证集分类准确率达到 `0.9964`，主要混淆仅出现在 `fear` 与 `happy` 之间，各有 `1` 条样本被错分。

关键结果摘要：

- 数据集规模：`2800` 条样本，`7` 类情绪，每类 `400` 条
- 储备池规模：`400` 个神经元，`3227` 条连接
- 状态特征维度：`800`
- 平均全局发放率：`0.010204`
- t-SNE 最终 KL divergence：`0.921806`
- 分类器：`StandardScaler + LogisticRegression`
- 验证集准确率：`0.9964`
- 验证集 Macro-F1：`0.9964`
- 主要混淆：`fear -> happy: 1`，`happy -> fear: 1`

终端输出节选如下：

```text
Reservoir simulation: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2800/2800 [03:11<00:00, 14.64sample/s]
[t-SNE] Computing 91 nearest neighbors...
[t-SNE] Indexed 2800 samples in 0.002s...
[t-SNE] Computed neighbors for 2800 samples in 1.661s...
[t-SNE] Computed conditional probabilities for sample 1000 / 2800
[t-SNE] Computed conditional probabilities for sample 2000 / 2800
[t-SNE] Computed conditional probabilities for sample 2800 / 2800
[t-SNE] Mean sigma: 5.577257
[t-SNE] KL divergence after 250 iterations with early exaggeration: 62.907555
[t-SNE] KL divergence after 1200 iterations: 0.921806
Classifier Report
Model: standardized_logistic_regression
Split: train=2240  validation=560
Validation: accuracy=0.9964  balanced_accuracy=0.9964  macro_f1=0.9964

Per-class Metrics
label              precision  recall  f1      support
-----------------  ---------  ------  ------  -------
angry              1.0000     1.0000  1.0000  80
disgust            1.0000     1.0000  1.0000  80
fear               0.9875     0.9875  0.9875  80
happy              0.9875     0.9875  0.9875  80
neutral            1.0000     1.0000  1.0000  80
pleasant_surprise  1.0000     1.0000  1.0000  80
sad                1.0000     1.0000  1.0000  80

Main Confusions:
- fear -> happy: 1
- happy -> fear: 1
```

该次运行对应的结构化结果摘要如下：

```yaml
dataset_root: G:\Izhikevich Reservoir for Speech Emotion Projection\data\raw\TESS
num_samples: 2800
label_distribution:
  angry: 400
  disgust: 400
  fear: 400
  happy: 400
  neutral: 400
  pleasant_surprise: 400
  sad: 400
num_state_features: 800
mean_global_firing_rate: 0.010204106108618102
std_global_firing_rate: 0.000282512674132369
reservoir:
  n_neurons: 400
  n_edges: 3227
  excitatory_neurons: 320
  inhibitory_neurons: 80
  excitatory_edges: 2290
  inhibitory_edges: 937
outputs:
  state_file: G:\Izhikevich Reservoir for Speech Emotion Projection\data\processed\reservoir_states.npz
  tsne_plot: G:\Izhikevich Reservoir for Speech Emotion Projection\results\tsne_emotion_clusters.png
  summary_yaml: G:\Izhikevich Reservoir for Speech Emotion Projection\results\run_summary.yaml
  loaded_model: G:\Izhikevich Reservoir for Speech Emotion Projection\results\reservoir_model.npz
  saved_model: G:\Izhikevich Reservoir for Speech Emotion Projection\results\reservoir_model.npz
classifier:
  enabled: true
  model_type: standardized_logistic_regression
  train_samples: 2240
  validation_samples: 560
  train_distribution:
    angry: 320
    disgust: 320
    fear: 320
    happy: 320
    neutral: 320
    pleasant_surprise: 320
    sad: 320
  validation_distribution:
    angry: 80
    disgust: 80
    fear: 80
    happy: 80
    neutral: 80
    pleasant_surprise: 80
    sad: 80
  validation_accuracy: 0.9964285714285714
  validation_balanced_accuracy: 0.9964285714285713
  validation_macro_f1: 0.9964285714285713
  outputs:
    classifier_model: G:\Izhikevich Reservoir for Speech Emotion Projection\results\reservoir_classifier.joblib
    classifier_metrics: G:\Izhikevich Reservoir for Speech Emotion Projection\results\classifier_metrics.yaml
    confusion_matrix_plot: G:\Izhikevich Reservoir for Speech Emotion Projection\results\classifier_confusion_matrix.png
    validation_predictions: G:\Izhikevich Reservoir for Speech Emotion Projection\results\classifier_validation_predictions.csv
```
