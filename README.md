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
- 支持 `STP / IP` 开关与独立消融实验脚本
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
├─ experiments/
│  └─ stp_ip_ablation/
│     ├─ README.md
│     ├─ run_ablation.py
│     └─ outputs/
│  └─ mfcc_direct_classification/
│     ├─ README.md
│     ├─ run_experiment.py
│     └─ outputs/
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

## 环境与安装

本项目提供两种安装方式：

- `conda` / `mamba`：使用仓库中的 `environment.yml` 创建完整环境
- `pip`：使用仓库中的 `requirements.txt` 安装 Python 依赖

推荐优先使用 `conda` 或 `mamba`，因为仓库已经提供了环境定义文件，复现更直接。

### 方式一：使用 `environment.yml`（推荐）

```powershell
conda env create -f environment.yml
conda activate torch
```

如果环境已经存在，需要按文件内容更新，可以使用：

```powershell
conda env update -f environment.yml --prune
conda activate torch
```

如果你使用的是 `mamba`，对应命令为：

```powershell
mamba env create -f environment.yml
mamba activate torch
```

### 方式二：使用 `requirements.txt`

如果你已经有一个可用的 Python 环境，也可以直接安装 `requirements.txt` 中列出的依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

建议使用 Python `3.11`。

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
- `experiments/stp_ip_ablation/run_ablation.py`：消融实验入口，自动对比 `baseline`、`no_stp`、`no_ip`、`no_stp_no_ip`
- `experiments/stp_ip_ablation/run_readout_comparison.py`：直接读取各条件已保存的 `reservoir_states.npz`，比较多个分类读出器
- `experiments/mfcc_direct_classification/run_experiment.py`：不经过储备池，直接用 MFCC 特征比较多个分类器，并与 reservoir baseline 对照
- `experiments/mfcc_direct_classification/plot_tsne.py`：直接基于已缓存的 MFCC 特征生成 `t-SNE` 情绪聚类图

如果你想先查看脚本支持哪些参数，可以直接运行：

```powershell
python .\run_ser_experiment.py --help
python .\train_classifier_from_states.py --help
python .\experiments\stp_ip_ablation\run_ablation.py --help
python .\experiments\stp_ip_ablation\run_readout_comparison.py --help
python .\experiments\mfcc_direct_classification\run_experiment.py --help
python .\experiments\mfcc_direct_classification\plot_tsne.py --help
```

说明：

- 形如 `--no-limit`、`--save-model`、`--quiet` 这种参数是开关，写上就表示启用，不需要再跟 `true/false`
- 形如 `--seed 7`、`--model-path results/models` 这种参数后面需要跟值

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
python .\run_ser_experiment.py
```

默认配置含义：

- 总样本数最多 `420`
- 每种情绪最多 `70`
- 会训练分类器
- 不会保存储备池模型文件

全量样本运行：

```powershell
python .\run_ser_experiment.py --no-limit
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
python .\run_ser_experiment.py --save-model
python .\run_ser_experiment.py --no-limit --save-model
python .\run_ser_experiment.py --no-limit --load-model
python .\run_ser_experiment.py --load-model --save-model
python .\run_ser_experiment.py --load-model results/models/reservoir_model_20260329_120000.npz
python .\run_ser_experiment.py --save-model --model-path results/models
python .\run_ser_experiment.py --max-samples 560 --max-per-emotion 80 --frame-repeat 2 --input-gain 14
python .\run_ser_experiment.py --no-limit --no-classifier
python .\run_ser_experiment.py --no-limit --quiet
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
- 若未额外指定路径，模型会默认保存到 `results/models/`，并自动在文件名后追加时间戳
- 示例：`--save-model`

`--model-path` 或 `--checkpoint-path`

- 指定储备池模型保存目录或保存文件名前缀
- 若传目录，例如 `results/models`，程序会自动生成带时间戳的文件名
- 若传文件名，例如 `results/models/my_model.npz`，程序会保存为 `my_model_时间戳.npz`
- 未显式传入时，默认保存目录为 `results/models/`
- 示例：`--model-path results/models`
- 示例：`--model-path results/models/my_model.npz`

`--load-model` 或 `--load-checkpoint`

- 加载已有储备池模型，而不是重新随机生成网络结构和权重
- 如果只写 `--load-model` 而不跟路径，程序会默认加载 `results/models/` 中最新的模型文件
- 也可以显式指定某一个模型文件，或传入一个目录并自动选择其中最新的模型
- 两个写法是同一个参数的别名，效果相同
- 示例：`--load-model`
- 示例：`--load-model results/models/reservoir_model_20260329_120000.npz`

`--no-classifier`

- 跳过分类器训练
- 仍然会保留状态提取、t-SNE、summary 等原有流程
- 示例：`--no-classifier`

`--quiet`

- 减少过程输出
- 适合批量运行或不想看进度信息时使用
- 示例：`--quiet`

### 消融实验脚本：`experiments/stp_ip_ablation/run_ablation.py`

这个脚本用于做 `STP / IP` 消融实验，会自动跑下面 4 组条件：

- `baseline`：保留 `STP + IP`
- `no_stp`：去除 `STP`
- `no_ip`：去除 `IP`
- `no_stp_no_ip`：同时去除 `STP` 和 `IP`

每组条件都会独立完成：

- 数据采样
- MFCC 提取
- 储备池状态提取
- t-SNE 可视化
- 分类器训练与验证

常用命令如下：

```powershell
python .\experiments\stp_ip_ablation\run_ablation.py
python .\experiments\stp_ip_ablation\run_ablation.py --no-limit
python .\experiments\stp_ip_ablation\run_ablation.py --max-samples 560 --max-per-emotion 80 --quiet
```

参数含义与主实验脚本保持一致：

- `--seed`：控制网络初始化、样本切分和实验随机性
- `--max-samples`：限制总样本数
- `--max-per-emotion`：限制每类样本数
- `--no-limit`：使用全部样本
- `--frame-repeat`：控制每帧 MFCC 注入次数
- `--input-gain`：控制输入电流强度
- `--quiet`：减少进度输出

输出会统一保存到：

```text
experiments/stp_ip_ablation/outputs/
```

其中每个条件都有自己的独立目录，例如：

- `experiments/stp_ip_ablation/outputs/baseline/results/`
- `experiments/stp_ip_ablation/outputs/no_stp/results/`
- `experiments/stp_ip_ablation/outputs/no_ip/results/`
- `experiments/stp_ip_ablation/outputs/no_stp_no_ip/results/`

同时还会额外汇总生成：

- `experiments/stp_ip_ablation/outputs/comparison_metrics.csv`
- `experiments/stp_ip_ablation/outputs/comparison_summary.yaml`
- `experiments/stp_ip_ablation/outputs/comparison_report.md`
- `experiments/stp_ip_ablation/outputs/comparison_metrics.png`

如果你已经有各条件的状态文件，不想重新经过储备池，还可以直接比较不同读出器：

```powershell
python .\experiments\stp_ip_ablation\run_readout_comparison.py
```

默认会比较：

- `logistic_regression`
- `nearest_centroid`
- `gaussian_nb`
- `ridge_classifier`

结果会保存到：

- `experiments/stp_ip_ablation/outputs/readout_comparison/comparison_metrics.csv`
- `experiments/stp_ip_ablation/outputs/readout_comparison/comparison_summary.yaml`
- `experiments/stp_ip_ablation/outputs/readout_comparison/comparison_report.md`

### 直接 MFCC 对比实验：`experiments/mfcc_direct_classification/run_experiment.py`

这个脚本不经过储备池，而是直接对每条语音提取 MFCC，然后：

- 将变长 MFCC 统一补齐到相同帧长
- 展平成定长特征向量
- 训练并比较多个分类读出器
- 读取 reservoir baseline 结果并做差值对照
- 默认额外生成直接 MFCC 特征的 `t-SNE` 图

常用命令如下：

```powershell
python .\experiments\mfcc_direct_classification\run_experiment.py
python .\experiments\mfcc_direct_classification\run_experiment.py --refresh-features
python .\experiments\mfcc_direct_classification\run_experiment.py --classifiers logistic_regression ridge_classifier
python .\experiments\mfcc_direct_classification\plot_tsne.py
```

结果会保存到：

- `experiments/mfcc_direct_classification/outputs/processed/mfcc_flat_features.npz`
- `experiments/mfcc_direct_classification/outputs/processed/mfcc_tsne_embedding.npz`
- `experiments/mfcc_direct_classification/outputs/results/tsne_emotion_clusters.png`
- `experiments/mfcc_direct_classification/outputs/comparison_metrics.csv`
- `experiments/mfcc_direct_classification/outputs/comparison_summary.yaml`
- `experiments/mfcc_direct_classification/outputs/comparison_report.md`

### 独立分类器脚本：`train_classifier_from_states.py`

这个脚本不会重新跑储备池仿真。
它只读取已经保存的 `data/processed/reservoir_states.npz`，然后基于其中的：

- `states`
- `labels`
- `paths`

重新训练分类器并输出结果。

最常用命令：

```powershell
python .\train_classifier_from_states.py --state-file data/processed/reservoir_states.npz
```

这个脚本适合下面这些场景：

- 你已经跑完储备池，不想再次重复仿真
- 你只想重新切分训练集/验证集并训练分类器
- 你想把分类器输出到新的文件名

常见命令组合：

```powershell
python .\train_classifier_from_states.py --state-file data/processed/reservoir_states.npz
python .\train_classifier_from_states.py --state-file data/processed/reservoir_states.npz --classifier nearest_centroid
python .\train_classifier_from_states.py --state-file data/processed/reservoir_states.npz --validation-ratio 0.25
python .\train_classifier_from_states.py --state-file data/processed/reservoir_states.npz --model-path results/my_classifier.joblib --metrics-path results/my_metrics.yaml
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

`--classifier`

- 指定读出器类型
- 默认值为 `logistic_regression`
- 支持：`logistic_regression`、`nearest_centroid`、`gaussian_nb`、`ridge_classifier`
- 示例：`--classifier ridge_classifier`

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
python .\run_ser_experiment.py --no-limit
python .\train_classifier_from_states.py --state-file data/processed/reservoir_states.npz
```

## 输出

- `results/tsne_emotion_clusters.png`：2D 情绪状态投影图
- `data/processed/reservoir_states.npz`：高维状态、嵌入、标签、路径
- `results/run_summary.yaml`：运行摘要与网络统计
- `results/reservoir_classifier.joblib`：基于 800 维状态向量训练得到的分类器
- `results/classifier_metrics.yaml`：训练/验证划分、准确率、Macro-F1、逐类指标、混淆矩阵
- `results/classifier_confusion_matrix.png`：验证集混淆矩阵可视化
- `results/classifier_validation_predictions.csv`：验证集逐样本预测结果
- `results/models/*.npz`：可选，保存网络结构、突触权重、输入映射与内在偏置（通过 `--save-model` 生成，文件名自动追加时间戳）
- `experiments/stp_ip_ablation/outputs/*/processed/reservoir_states.npz`：各消融条件的状态文件
- `experiments/stp_ip_ablation/outputs/*/results/run_summary.yaml`：各消融条件的运行摘要
- `experiments/stp_ip_ablation/outputs/comparison_metrics.csv`：消融实验关键指标汇总表
- `experiments/stp_ip_ablation/outputs/comparison_report.md`：消融实验结果汇总报告
- `experiments/stp_ip_ablation/outputs/readout_comparison/*`：基于已有 reservoir states 的多读出器对比结果
- `experiments/mfcc_direct_classification/outputs/processed/mfcc_flat_features.npz`：直接 MFCC 展平特征缓存
- `experiments/mfcc_direct_classification/outputs/results/tsne_emotion_clusters.png`：直接 MFCC 特征的 `t-SNE` 图
- `experiments/mfcc_direct_classification/outputs/comparison_report.md`：直接 MFCC 与 reservoir baseline 的对比报告
- `EXPERIMENT_RESULTS_ANALYSIS.md`：当前阶段实验结论、局限性与后续研究方向整理

## 说明

- 当前实现默认保留跨样本的长期塑性（STDP + IP），并在每个样本前重置短期动力学状态。
- `ExperimentConfig` 中新增了 `enable_stp` 与 `enable_intrinsic_plasticity` 开关，默认都为 `True`。
- 若你需要做 `STP / IP` 消融，推荐直接使用 `experiments/stp_ip_ablation/run_ablation.py`，避免手工修改主流程代码。
- 分类器默认使用当前运行得到的 `reservoir_states.npz` 同源状态向量进行训练，先做标准化，再训练多分类逻辑回归。
- 若传入 `--load-model`，程序会加载已有模型；若只写 `--load-model` 不跟路径，则默认读取 `results/models/` 下最新模型；若不传，则从随机初始化的新模型开始运行。
- 模型文件会保存关键动力学/可塑性配置的元信息，避免把不兼容的配置静默加载到旧模型上。
- 若你后续需要接入岭回归或线性探针，可直接使用 `reservoir_states.npz` 的 `states` 与 `labels`。

## 当前结论

基于当前已经完成的 3 组实验，可以先得到一个阶段性结论：

- 在当前 `TESS + 整句分类 + 全量样本 + 线性读出` 设定下，直接 MFCC 已经足够强。
- `mfcc_direct + logistic_regression = 0.9929`，与 reservoir baseline `0.9946` 几乎相同。
- `mfcc_direct + ridge_classifier = 0.9964`，甚至略高于 reservoir baseline。
- `STP / IP` 消融里，`IP` 的影响明显大于 `STP`；去掉 `STP` 并没有带来稳定退化。
- 因此，当前实验还不能证明储备池在这个任务设定下具有明确优势。

但这不等价于“储备池没有研究价值”。更准确的说法是：

- 当前任务对直接高信息量 MFCC 特征过于友好。
- 当前评估方式还没有充分激发储备池在动态建模、时序记忆和鲁棒性方面的潜在优势。

如果目标是继续研究储备池，下一步更建议做：

- speaker-independent 或 cross-corpus 泛化
- 加噪声、混响、通道扰动的鲁棒性评估
- 小样本训练曲线
- 压缩后的直接 MFCC 统计特征 vs reservoir 状态
- 严格在线 / 流式识别

更完整的分析见：

- `EXPERIMENT_RESULTS_ANALYSIS.md`

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
dataset_root: data/raw/TESS
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
  state_file: data/processed/reservoir_states.npz
  tsne_plot: results/tsne_emotion_clusters.png
  summary_yaml: results/run_summary.yaml
  loaded_model: results/models/reservoir_model_20260329_120000.npz
  saved_model: results/models/reservoir_model_20260329_120000.npz
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
    classifier_model: results/reservoir_classifier.joblib
    classifier_metrics: results/classifier_metrics.yaml
    confusion_matrix_plot: results/classifier_confusion_matrix.png
    validation_predictions: results/classifier_validation_predictions.csv
```
