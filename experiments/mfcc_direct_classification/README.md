# MFCC Direct Classification Experiment

这个实验不再经过储备池，而是直接对每条语音提取 MFCC，并将其展平成定长特征后送入分类器。
脚本默认还会基于这些直接 MFCC 特征生成一个 `t-SNE` 情绪聚类图。

默认会比较 4 种读出器：

- `logistic_regression`
- `nearest_centroid`
- `gaussian_nb`
- `ridge_classifier`

同时会读取已有的 reservoir `baseline` 指标文件，并在汇总结果里给出相对 baseline 的差值。

## 当前观察

基于当前已经生成的结果：

- `mfcc_direct + logistic_regression` 与 reservoir baseline 几乎持平
- `mfcc_direct + ridge_classifier` 甚至略高于 reservoir baseline
- `nearest_centroid` 与 `gaussian_nb` 明显更弱，更适合作为“低能力读出器”对照
- 直接 MFCC 的 `t-SNE` 图已经表现出较好的情绪分离

这说明在当前 `TESS + 整句分类 + 全量样本` 设定下，直接 MFCC 特征本身已经非常强，储备池的优势尚未被这一任务充分激发。

如果你的目标是继续研究储备池，更适合把该实验当作一个强 baseline，而不是最终结论。

## 运行方式

默认使用全部可用样本：

```powershell
python .\experiments\mfcc_direct_classification\run_experiment.py
```

如果你想强制重新提取 MFCC 特征缓存：

```powershell
python .\experiments\mfcc_direct_classification\run_experiment.py --refresh-features
```

如果你只想跑部分类别器：

```powershell
python .\experiments\mfcc_direct_classification\run_experiment.py --classifiers logistic_regression ridge_classifier
```

如果你只想看 MFCC 直接特征的 `t-SNE` 图，不想重跑分类器：

```powershell
python .\experiments\mfcc_direct_classification\plot_tsne.py
```

## 输出位置

输出统一保存到：

```text
experiments/mfcc_direct_classification/outputs/
```

其中包括：

- `processed/mfcc_flat_features.npz`
- `processed/mfcc_tsne_embedding.npz`
- `results/logistic_regression/`
- `results/nearest_centroid/`
- `results/gaussian_nb/`
- `results/ridge_classifier/`
- `results/tsne_emotion_clusters.png`
- `comparison_metrics.csv`
- `comparison_summary.yaml`
- `comparison_report.md`
