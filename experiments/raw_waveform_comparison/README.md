# Raw Waveform Comparison Experiment

这个实验专门比较两种设定：

- `raw_direct`：不加 `MFCC`、不加储备池，直接用原始波形训练分类器
- `raw_reservoir`：不加 `MFCC`，但把原始波形改写成 `40 x 258` 输入矩阵后送入储备池，再训练分类器

## 原始波形是怎么输入的

每条语音都按同一套规则处理：

1. `soundfile.read(path)` 读取音频
2. 若是多通道，先做 mono 平均
3. 用 polyphase 重采样统一到 `22050 Hz`
4. 对整条波形做峰值归一化，压到 `[-1, 1]`
5. 用线性插值把整条语音统一重采样到固定长度 `10320` 点
6. 再把这 `10320` 点平移到 `[0, 1]`

之后分成两条实验线：

- `raw_direct`
  - 直接把 `10320` 维向量送入分类器
- `raw_reservoir`
  - 把同一条 `10320` 维向量按时间顺序 reshape 成 `flat.reshape(258, 40).T`
  - 得到 `40 x 258` 输入矩阵
  - 每一列代表连续的 `40` 个采样点
  - 再直接送入现有 reservoir，得到 `800` 维状态向量后训练分类器

两个条件强制共享同一批样本和同一份 `80/20` 分层切分，切分索引会单独保存到 `split_indices.npz`。

现在脚本默认还会额外为这两条实验线各生成一张 `t-SNE` 图：

- `raw_direct`：直接基于 `10320` 维原始波形向量做 `t-SNE`
- `raw_reservoir`：基于 `800` 维 reservoir 状态向量做 `t-SNE`

对应的 `t-SNE` 嵌入文件里现在会一起保存：

- `embedding`
- `labels`
- `kl_divergence`
- `perplexity_used`
- `n_iter_completed`
- 以及本次运行使用的其他关键参数

## 运行方式

默认使用全量 TESS：

```powershell
python .\experiments\raw_waveform_comparison\run_experiment.py
```

如果你想先做一个很小的 smoke run：

```powershell
python .\experiments\raw_waveform_comparison\run_experiment.py --max-samples 28 --max-per-emotion 4 --quiet
```

如果你想强制重新构建原始波形缓存：

```powershell
python .\experiments\raw_waveform_comparison\run_experiment.py --refresh-features
```

如果你只想先跑分类结果，不生成 `t-SNE`：

```powershell
python .\experiments\raw_waveform_comparison\run_experiment.py --skip-tsne
```

## 输出位置

统一保存在：

```text
experiments/raw_waveform_comparison/outputs/
```

主要文件包括：

- `processed/raw_waveform_flat_features.npz`
- `processed/split_indices.npz`
- `processed/raw_direct_tsne_embedding.npz`
- `processed/raw_reservoir_tsne_embedding.npz`
- `raw_direct/results/`
- `raw_reservoir/processed/reservoir_states.npz`
- `raw_reservoir/results/`
- `comparison_metrics.csv`
- `comparison_summary.yaml`
- `comparison_report.md`

其中两张 `t-SNE` 图分别会落在：

- `raw_direct/results/tsne_emotion_clusters.png`
- `raw_reservoir/results/tsne_emotion_clusters.png`
