# MFCC Direct Classification Report

## Feature Setup

- samples: 2800
- feature_dim: 10320
- n_mfcc: 40
- target_frames: 258
- frame_lengths: min=109  max=258  mean=177.52
- baseline_metrics: G:\Izhikevich Reservoir for Speech Emotion Projection\experiments\stp_ip_ablation\outputs\baseline\results\classifier_metrics.yaml

## Comparison

| Experiment | Classifier | Accuracy | Balanced Acc | Macro F1 | Delta Acc vs Reservoir Baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| reservoir_baseline | logistic_regression | 0.9946 | 0.9946 | 0.9946 | +0.0000 |
| mfcc_direct | gaussian_nb | 0.3946 | 0.3946 | 0.3175 | -0.6000 |
| mfcc_direct | logistic_regression | 0.9929 | 0.9929 | 0.9928 | -0.0018 |
| mfcc_direct | nearest_centroid | 0.8696 | 0.8696 | 0.8694 | -0.1250 |
| mfcc_direct | ridge_classifier | 0.9964 | 0.9964 | 0.9964 | +0.0018 |
