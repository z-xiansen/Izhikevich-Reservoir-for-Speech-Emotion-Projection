# Raw Waveform Comparison Report

## Raw Input Definition

- `soundfile.read()` -> mono downmix if needed
- polyphase resample to `22050 Hz`
- peak-normalize waveform to `[-1, 1]`
- linearly resample each utterance to `10320` samples
- shift the fixed-length vector to `[0, 1]`
- `raw_direct`: feed the `10320`-D vector directly into the classifier
- `raw_reservoir`: reshape with `flat.reshape(258, 40).T` to `[40, 258]`, then feed the reservoir

## Feature Setup

- samples: 2800
- sample_rate: 22050
- target_length: 10320
- input_channels: 40
- target_frames: 258
- feature_file: G:\Izhikevich Reservoir for Speech Emotion Projection\experiments\raw_waveform_comparison\outputs\processed\raw_waveform_flat_features.npz
- split_file: G:\Izhikevich Reservoir for Speech Emotion Projection\experiments\raw_waveform_comparison\outputs\processed\split_indices.npz
- cache_source: loaded

## Reservoir Summary

- state_file: G:\Izhikevich Reservoir for Speech Emotion Projection\experiments\raw_waveform_comparison\outputs\raw_reservoir\processed\reservoir_states.npz
- num_state_features: 800
- mean_global_firing_rate: 0.009343
- std_global_firing_rate: 0.001517

## t-SNE Outputs

- raw_direct_embedding: G:\Izhikevich Reservoir for Speech Emotion Projection\experiments\raw_waveform_comparison\outputs\processed\raw_direct_tsne_embedding.npz
- raw_direct_plot: G:\Izhikevich Reservoir for Speech Emotion Projection\experiments\raw_waveform_comparison\outputs\raw_direct\results\tsne_emotion_clusters.png
- raw_direct_kl_divergence: 3.9707140922546387
- raw_direct_perplexity_used: 30.0
- raw_direct_n_iter_completed: 1199
- raw_reservoir_embedding: G:\Izhikevich Reservoir for Speech Emotion Projection\experiments\raw_waveform_comparison\outputs\processed\raw_reservoir_tsne_embedding.npz
- raw_reservoir_plot: G:\Izhikevich Reservoir for Speech Emotion Projection\experiments\raw_waveform_comparison\outputs\raw_reservoir\results\tsne_emotion_clusters.png
- raw_reservoir_kl_divergence: 1.9566401243209839
- raw_reservoir_perplexity_used: 30.0
- raw_reservoir_n_iter_completed: 1199

## Comparison

| Experiment | Accuracy | Balanced Acc | Macro F1 | Delta Acc vs Raw Direct |
| --- | ---: | ---: | ---: | ---: |
| raw_direct | 0.2411 | 0.2411 | 0.2412 | +0.0000 |
| raw_reservoir | 0.8679 | 0.8679 | 0.8680 | +0.6268 |
