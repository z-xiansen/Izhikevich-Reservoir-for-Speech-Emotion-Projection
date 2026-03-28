# ExperimentConfig 参数完整指南

## 概述

本文档总结了 `src/ser_reservoir/config.py` 中所有配置参数的含义和作用。该配置类用于**脉冲神经网络水库**进行语音情绪识别（SER）任务。

---

## 目录
1. [导入模块](#导入模块)
2. [数据相关参数](#数据相关参数)
3. [音频特征提取（MFCC）](#音频特征提取mfcc)
4. [水库拓扑参数](#水库拓扑参数)
5. [Izhikevich 神经元模型](#izhikevich-神经元模型)
6. [STDP 可塑性](#stdp-可塑性)
7. [STP 短期可塑性](#stp-短期可塑性)
8. [内在可塑性](#内在可塑性)
9. [t-SNE 可视化](#tsne-可视化)
10. [类方法和属性](#类方法和属性)

---

## 导入模块

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
```

| 模块 | 作用 |
|------|------|
| `__future__.annotations` | 启用 Python 3.10+ 的延迟注解评估，允许使用类名作为类型提示 |
| `dataclass` | 装饰器，自动生成 `__init__`、`__repr__` 等方法 |
| `field` | 用于定义数据类字段的高级配置 |
| `Path` | 跨平台路径操作，避免硬编码路径分隔符 |

---

## 数据相关参数

### 基本配置

```python
seed: int = 7
workspace: Path = Path(".")
```

| 参数 | 默认值 | 含义 |
|------|-------|------|
| `seed` | 7 | 随机数种子，保证实验可复现性 |
| `workspace` | `"."` | 项目根目录（当前工作目录） |

### 数据集配置

```python
dataset_id: str = "ejlok1/toronto-emotional-speech-set-tess"
hf_dataset_id: str = "TwinkStart/TESS"
raw_data_dir: Path = Path("data/raw")
processed_dir: Path = Path("data/processed")
results_dir: Path = Path("results")
```

| 参数 | 默认值 | 含义 |
|------|-------|------|
| `dataset_id` | Kaggle ID | Hugging Face TESS 数据集位置 |
| `hf_dataset_id` | Hugging Face ID | 备用数据集镜像 |
| `raw_data_dir` | `data/raw` | 原始 WAV 文件存放目录 |
| `processed_dir` | `data/processed` | 处理后的 MFCC 和特征存放目录 |
| `results_dir` | `results` | 实验结果、可视化、指标输出目录 |

### 样本限制

```python
max_samples: int | None = 420
max_samples_per_emotion: int | None = 70
```

| 参数 | 默认值 | 含义 |
|------|-------|------|
| `max_samples` | 420 | 最多加载 420 个样本（`None` = 无限制） |
| `max_samples_per_emotion` | 70 | 每种情绪最多 70 个样本，确保数据平衡 |

**设计理由**：$420 = 6 \text{ 种情绪} \times 70 \text{ 个样本}$，平衡数据集

---

## 音频特征提取（MFCC）

```python
sample_rate: int = 22050
n_mfcc: int = 40
n_fft: int = 1024
hop_length: int = 256
win_length: int = 1024
```

### 参数详解

| 参数 | 值 | 含义 | 单位 |
|------|-----|------|------|
| `sample_rate` | 22050 | 音频重采样频率 | Hz |
| `n_mfcc` | 40 | 每一帧提取的 MFCC 系数维度 | 维 |
| `n_fft` | 1024 | 短时傅里叶变换的窗长 | 采样点 |
| `hop_length` | 256 | 相邻两帧的步长 | 采样点 |
| `win_length` | 1024 | 每帧的加窗长度 | 采样点 |

### 时间对应

- 每帧时长：$\frac{1024}{22050} \approx 46.4 \text{ ms}$
- 帧间步长：$\frac{256}{22050} \approx 11.6 \text{ ms}$
- 帧间重叠：$1024 - 256 = 768$ 个采样点，约 $34.8 \text{ ms}$

### 帧数计算

对于 2 秒音频（$N = 44100$ 采样点）：

$$F = 1 + \left\lfloor \frac{N - \text{win\_length}}{\text{hop\_length}} \right\rfloor = 1 + \left\lfloor \frac{44100 - 1024}{256} \right\rfloor = 169 \text{ 帧}$$

**最终 MFCC 矩阵形状**：$[40, 169]$

### 处理流程

```
1024 点采样 → STFT → 频谱 → 梅尔变换 → Log → DCT → 40 维 MFCC 向量
```

---

## 水库拓扑参数

```python
grid_side: int = 20
exc_ratio: float = 0.8
base_conn_prob: float = 0.18
distance_lambda: float = 3.2
local_inh_radius: float = 2.0
local_inh_prob: float = 0.45
init_w_exc_mean: float = 0.85
init_w_exc_std: float = 0.18
init_w_inh_mean: float = 1.10
init_w_inh_std: float = 0.20
```

### 网络规模

| 参数 | 值 | 含义 |
|------|-----|------|
| `grid_side` | 20 | 神经元排成 20×20 网格 |
| **总神经元数** | **400** | $20 \times 20$ |

### 神经元分类

| 类型 | 数量 | 比例 |
|------|------|------|
| `excitatory_neurons` | 320 | 80% |
| `inhibitory_neurons` | 80 | 20% |

**生物学对应**：脑皮层约 75-85% 兴奋性，15-25% 抑制性

### 连接规则

| 参数 | 值 | 含义 |
|------|-----|------|
| `base_conn_prob` | 0.18 | 基础连接概率 18% |
| `distance_lambda` | 3.2 | 距离衰减参数（越小衰减越快） |
| `local_inh_radius` | 2.0 | 抑制神经元局部作用范围（网格距离） |
| `local_inh_prob` | 0.45 | 范围内连接到抑制性神经元的概率 |

**距离衰减公式**：
$$P(\text{连接}) = \text{base\_conn\_prob} \times e^{-\text{distance} / \text{distance\_lambda}}$$

### 权重初始化

| 参数 | 值 | 含义 |
|------|-----|------|
| `init_w_exc_mean` | 0.85 | 兴奋权重分布的均值 |
| `init_w_exc_std` | 0.18 | 兴奋权重分布的标准差 |
| `init_w_inh_mean` | 1.10 | 抑制权重分布的均值 |
| `init_w_inh_std` | 0.20 | 抑制权重分布的标准差 |

权重采样：
- 兴奋：$w_{\text{exc}} \sim |\mathcal{N}(0.85, 0.18^2)|$（绝对值，保证非负）
- 抑制：$w_{\text{inh}} \sim -|\mathcal{N}(1.10, 0.20^2)|$（负值）

### 实际网络统计

```yaml
n_neurons: 400
n_edges: 3227
excitatory_neurons: 320
inhibitory_neurons: 80
excitatory_edges: 2290
inhibitory_edges: 937
```

**连接密度**：$\frac{3227}{400 \times 399} \approx 2\%$

---

## Izhikevich 神经元模型

```python
dt_ms: float = 1.0
frame_repeat: int = 2
input_gain: float = 14.0
v_thresh: float = 30.0
v_init: float = -65.0
u_init_factor: float = 1.0
```

### 参数含义

| 参数 | 值 | 含义 |
|------|-----|------|
| `dt_ms` | 1.0 | 时间步长（毫秒） |
| `frame_repeat` | 2 | 每帧 MFCC 重复输入 2 次 |
| `input_gain` | 14.0 | 输入 MFCC 的增益系数 |
| `v_thresh` | 30.0 | 膜电位放电阈值（mV） |
| `v_init` | -65.0 | 初始膜电位（mV） |
| `u_init_factor` | 1.0 | 恢复变量初始化因子 |

### 模型动力学

Izhikevich 模型微分方程：

$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I$$

$$\frac{du}{dt} = a(bv - u)$$

放电条件：$v \geq v_{\text{thresh}}$ 时，$v \leftarrow c$，$u \leftarrow u + d$

---

## STDP 可塑性

```python
stdp_a_plus: float = 0.0040
stdp_a_minus: float = 0.0043
stdp_tau_pre: float = 20.0
stdp_tau_post: float = 20.0
w_exc_max: float = 2.0
w_inh_min: float = -2.5
```

### STDP 原理

**Spike-Timing Dependent Plasticity**（尖峰时间依赖可塑性）

- 如果**前神经元比后神经元提前**放电 → 权重**增强**（强化因果）
- 如果**前神经元比后神经元延后**放电 → 权重**减弱**（削弱反向因果）

### 参数详解

| 参数 | 值 | 含义 |
|------|-----|------|
| `stdp_a_plus` | 0.004 | 学习率（前导增强）|
| `stdp_a_minus` | 0.0043 | 学习率（后导削弱）|
| `stdp_tau_pre` | 20.0 | 前神经元放电迹痕衰减时间常数（ms） |
| `stdp_tau_post` | 20.0 | 后神经元放电迹痕衰减时间常数（ms） |
| `w_exc_max` | 2.0 | 兴奋权重的最大值 |
| `w_inh_min` | -2.5 | 抑制权重的最小值 |

### 效果时间窗口

- **20ms** 内的脉冲对会产生 STDP 效应
- 时间差 $\Delta t = t_{\text{post}} - t_{\text{pre}}$
- 权重范围：兴奋 $[0, 2.0]$，抑制 $[-2.5, 0]$

---

## STP 短期可塑性

```python
stp_u_exc: float = 0.22
stp_tau_d_exc: float = 700.0
stp_tau_f_exc: float = 50.0
stp_u_inh: float = 0.06
stp_tau_d_inh: float = 120.0
stp_tau_f_inh: float = 760.0
```

### Tsodyks-Markram 模型

突触效能：$\text{效果} = w \times u(t) \times x(t)$

其中：
- $u(t)$：利用率（促进过程）
- $x(t)$：可用资源（抑制过程）

### 兴奋性突触

| 参数 | 值 | 含义 |
|------|-----|------|
| `stp_u_exc` | 0.22 | 初始利用率（22%） |
| `stp_tau_d_exc` | 700.0 | 抑制恢复时间常数（ms） |
| `stp_tau_f_exc` | 50.0 | 促进衰退时间常数（ms） |

**特点**：频繁刺激时容易**疲劳**（资源耗尽）

### 抑制性突触

| 参数 | 值 | 含义 |
|------|-----|------|
| `stp_u_inh` | 0.06 | 初始利用率（6%） |
| `stp_tau_d_inh` | 120.0 | 抑制恢复时间常数（ms） |
| `stp_tau_f_inh` | 760.0 | 促进衰退时间常数（ms） |

**特点**：频繁刺激时会**积累**（促进效应持久）

### 对比

| 特性 | 兴奋 | 抑制 |
|------|------|------|
| 初始利用率 | 0.22 | 0.06 |
| 容易疲劳 | ✓ | ✗ |
| 促进持久 | ✗ | ✓ |
| 功能 | 制动器 | 加速器 |

---

## 内在可塑性

```python
ip_lr: float = 0.0012
ip_target_rate: float = 0.045
ip_bias_min: float = -4.0
ip_bias_max: float = 4.0
```

### 目的

自动调整每个神经元的**内在偏置**，使网络工作在"**边界混沌**"状态（最优计算能力）。

### 参数详解

| 参数 | 值 | 含义 |
|------|-----|------|
| `ip_lr` | 0.0012 | 学习率（0.12%，很慢） |
| `ip_target_rate` | 0.045 | 目标放电率（4.5%） |
| `ip_bias_min` | -4.0 | 偏置的最小值 |
| `ip_bias_max` | 4.0 | 偏置的最大值 |

### 调整机制

```python
error = ip_target_rate - actual_firing_rate
delta = ip_lr * error
bias += delta
bias = clip(bias, ip_bias_min, ip_bias_max)
```

- 如果神经元**放电太频繁** → 降低偏置
- 如果神经元**放电太稀疏** → 提高偏置
- 最终稳定在 **4.5% 的目标放电率**

### 边界混沌状态

```
有序态          边界混沌         混沌态
(Dead zone)   (Critical)    (Chaos)
───────────────────────────────────
快速消退  →  缓慢传播 ←  发散
信息损失      最优工作点
              (ip_target_rate = 0.045)
```

---

## t-SNE 可视化

```python
tsne_perplexity: float = 30.0
tsne_learning_rate: float = 200.0
tsne_n_iter: int = 1200
```

### 参数含义

| 参数 | 值 | 含义 |
|------|-----|------|
| `tsne_perplexity` | 30.0 | 困惑度（考虑邻点数） |
| `tsne_learning_rate` | 200.0 | 学习率 |
| `tsne_n_iter` | 1200 | 最大迭代次数 |

### 过程

```
高维特征 [N, 800]
    ↓
标准化 (StandardScaler)
    ↓
t-SNE 降维
(perplexity=30, lr=200, iter=1200)
    ↓
2D 嵌入 [N, 2]
    ↓
绘制散点图（不同情绪不同颜色）
```

### 参数选择理由

- **perplexity < 50**：平衡局部和全局结构
- **learning_rate = 200**：标准值，收敛快
- **n_iter = 1200**：足够充分收敛

---

## 类方法和属性

### 1. resolve() 方法

**作用**：将相对路径转换为绝对路径

```python
def resolve(self) -> "ExperimentConfig":
    self.workspace = self.workspace.resolve()
    self.raw_data_dir = (self.workspace / self.raw_data_dir).resolve()
    self.processed_dir = (self.workspace / self.processed_dir).resolve()
    self.results_dir = (self.workspace / self.results_dir).resolve()
    return self
```

**使用**：
```python
cfg = ExperimentConfig().resolve()  # 返回自己，支持链式调用
```

### 2. @property 计算属性

#### n_neurons
```python
@property
def n_neurons(self) -> int:
    return self.grid_side * self.grid_side  # 20 × 20 = 400
```

#### tess_target_dir
```python
@property
def tess_target_dir(self) -> Path:
    return self.raw_data_dir / "TESS"
```

#### state_feature_path
```python
@property
def state_feature_path(self) -> Path:
    return self.processed_dir / "reservoir_states.npz"
```
**内容**：状态矩阵 $[N, 800]$，embedding $[N, 2]$，labels，paths

#### tsne_plot_path
```python
@property
def tsne_plot_path(self) -> Path:
    return self.results_dir / "tsne_emotion_clusters.png"
```

#### metrics_path
```python
@property
def metrics_path(self) -> Path:
    return self.results_dir / "run_summary.yaml"
```

#### label_order
```python
@property
def label_order(self) -> list[str]:
    return ["angry", "disgust", "fear", "happy", 
            "neutral", "pleasant_surprise", "sad"]
```
**作用**：确保情绪标签顺序一致（6 类情感）

### 3. as_dict() 方法

**作用**：将配置对象转换为可序列化的字典

```python
def as_dict(self) -> dict:
    return {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in self.__dict__.items()
    }
```

**功能**：
- 遍历所有属性
- Path 对象转换为字符串
- 其他类型保持原样

**用途**：保存配置到 YAML/JSON

---

## 数据流总结

```
音频输入 (WAV)
    ↓
提取 MFCC [40×F] (sample_rate, n_mfcc, n_fft...)
    ↓
归一化 [0, 1]
    ↓
输入水库网络
    ├─ 400 个 Izhikevich 神经元 (dt_ms, v_thresh...)
    ├─ 3227 条调制突触 (base_conn_prob, distance_lambda...)
    ├─ STDP 学习 (stdp_a_plus, stdp_tau_pre...)
    ├─ STP 动态 (stp_u_exc, stp_tau_d_exc...)
    └─ 内在可塑性调谐 (ip_lr, ip_target_rate...)
    ↓
提取特征 [N, 800]
    ├─ 前 400 维：平均膜电位
    └─ 后 400 维：放电率
    ↓
t-SNE 降维 [N, 2] (tsne_perplexity, tsne_learning_rate...)
    ↓
可视化 + 情绪聚类
```

---

## 快速参考表

### 网络规模
- **神经元**：400 个（20×20 网格）
- **兴奋/抑制**：320/80
- **连接**：3227 条（~2% 密度）
- **输出特征**：800 维

### 时间尺度
- **采样**：22050 Hz
- **MFCC 帧**：~11.6 ms 间隔
- **模拟时间步**：1.0 ms
- **STDP 窗口**：20 ms
- **STP 恢复**：50-760 ms

### 学习参数
- **STDP**：0.004-0.0043（很小）
- **内在可塑性**：0.0012（极小）
- **目标放电率**：4.5%（稀疏）

### 约束范围
- **权重**：兴奋 [0, 2.0]，抑制 [-2.5, 0]
- **偏置**：[-4.0, 4.0]
- **样本**：最多 420 个（70 个/情绪）

---

## 修改建议

| 目标 | 修改参数 |
|------|---------|
| 增加网络复杂性 | ↑ `grid_side` (20 → 30) |
| 提高学习速率 | ↑ `stdp_a_plus`, `stdp_a_minus` |
| 加强稀疏性 | ↓ `ip_target_rate` (0.045 → 0.03) |
| 提高时间分辨率 | ↓ `hop_length` (256 → 128) |
| 增加频谱细节 | ↑ `n_mfcc` (40 → 80) |
| 加快 t-SNE | ↑ `tsne_learning_rate` (200 → 300) |

