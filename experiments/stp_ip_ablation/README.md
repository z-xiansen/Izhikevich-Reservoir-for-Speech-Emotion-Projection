# STP / IP Ablation Experiment

这个目录用于做储备池可塑性消融实验，对比以下 4 种设置：

- `baseline`: 保留 `STP + IP`
- `no_stp`: 去除 `STP`
- `no_ip`: 去除 `IP`
- `no_stp_no_ip`: 同时去除 `STP` 和 `IP`

## 运行方式

默认使用与主项目一致的采样限制：

```powershell
python .\experiments\stp_ip_ablation\run_ablation.py
```

若要使用全部 TESS 样本：

```powershell
python .\experiments\stp_ip_ablation\run_ablation.py --no-limit
```

如果你已经跑完消融实验并生成了各条件的 `reservoir_states.npz`，可以直接比较不同分类读出头，而不重新经过储备池：

```powershell
python .\experiments\stp_ip_ablation\run_readout_comparison.py
```

这个脚本默认会对每个条件直接读取已有状态文件，并比较 4 种读出器：

- `logistic_regression`
- `nearest_centroid`
- `gaussian_nb`
- `ridge_classifier`

## 当前观察

基于当前已经完成的消融结果：

- `IP` 的作用明显强于 `STP`
- 去掉 `IP` 后性能会从约 `0.9946` 降到 `0.9375`
- 去掉 `STP` 后并没有出现稳定退化，`no_stp` 甚至略高于 `baseline`

这更接近于说明：

- 当前实现中的 `IP` 确实在显著影响网络可分性
- 当前实验设定下，`STP` 的边际贡献较弱，或者被其他机制与读出方式稀释了

因此，这组实验更适合被理解为“当前任务下哪种可塑性更重要”，而不是“储备池整体是否必要”的最终判断。

## 输出位置

实验输出统一保存在：

```text
experiments/stp_ip_ablation/outputs/
```

其中每个条件都会有独立目录，包含：

- `processed/reservoir_states.npz`
- `results/run_summary.yaml`
- `results/classifier_metrics.yaml`
- `results/classifier_confusion_matrix.png`
- `results/tsne_emotion_clusters.png`

汇总比较结果会额外保存为：

- `outputs/comparison_metrics.csv`
- `outputs/comparison_summary.yaml`
- `outputs/comparison_report.md`
- `outputs/comparison_metrics.png`
- `outputs/readout_comparison/comparison_metrics.csv`
- `outputs/readout_comparison/comparison_summary.yaml`
- `outputs/readout_comparison/comparison_report.md`
