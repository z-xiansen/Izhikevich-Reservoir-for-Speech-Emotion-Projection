from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from .analysis import plot_embedding, run_tsne, save_states
from .classifier import train_and_evaluate_classifier
from .config import ExperimentConfig
from .data import (
    collect_audio_samples,
    ensure_tess_dataset,
    extract_mfcc,
)
from .reservoir import IzhikevichReservoir


def run_pipeline(cfg: ExperimentConfig) -> dict:
    cfg.resolve()
    np.random.seed(cfg.seed)

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = ensure_tess_dataset(cfg)
    samples = collect_audio_samples(dataset_root, cfg)
    if len(samples) < 3:
        raise RuntimeError("At least 3 samples are required.")

    if cfg.load_model_path is not None:
        reservoir = IzhikevichReservoir.from_model(cfg, cfg.load_model_path)
    else:
        reservoir = IzhikevichReservoir(cfg)

    states: list[np.ndarray] = []
    labels: list[str] = []
    path_list: list[str] = []
    global_rates: list[float] = []

    iterator = tqdm(samples, desc="Reservoir simulation", unit="sample", disable=not cfg.verbose)
    for sample in iterator:
        mfcc = extract_mfcc(sample.path, cfg)
        feature, global_rate = reservoir.present(mfcc)
        states.append(feature)
        labels.append(sample.label)
        path_list.append(str(sample.path))
        global_rates.append(global_rate)

    state_matrix = np.vstack(states).astype(np.float32)
    embedding = run_tsne(state_matrix, cfg)
    plot_embedding(embedding, labels, cfg.tsne_plot_path)
    save_states(cfg.state_feature_path, state_matrix, embedding, labels, path_list)

    classifier_summary: dict | None = None
    if cfg.train_classifier:
        classifier_summary = train_and_evaluate_classifier(state_matrix, labels, path_list, cfg)

    saved_model_path: Path | None = None
    if cfg.save_model:
        reservoir.save_model(cfg.model_path)
        saved_model_path = cfg.model_path

    summary = _build_summary(
        cfg=cfg,
        reservoir=reservoir,
        dataset_root=dataset_root,
        labels=labels,
        global_rates=global_rates,
        n_features=state_matrix.shape[1],
        saved_model_path=saved_model_path,
        classifier_summary=classifier_summary,
    )
    _save_summary(cfg.metrics_path, summary)
    return summary


def _build_summary(
    cfg: ExperimentConfig,
    reservoir: IzhikevichReservoir,
    dataset_root: Path,
    labels: list[str],
    global_rates: list[float],
    n_features: int,
    saved_model_path: Path | None,
    classifier_summary: dict | None,
) -> dict:
    label_count = dict(sorted(Counter(labels).items(), key=lambda x: x[0]))
    res_stats = reservoir.stats()
    return {
        "dataset_root": str(dataset_root),
        "num_samples": len(labels),
        "label_distribution": label_count,
        "num_state_features": int(n_features),
        "mean_global_firing_rate": float(np.mean(global_rates)),
        "std_global_firing_rate": float(np.std(global_rates)),
        "reservoir": {
            "n_neurons": res_stats.n_neurons,
            "n_edges": res_stats.n_edges,
            "excitatory_neurons": res_stats.excitatory_neurons,
            "inhibitory_neurons": res_stats.inhibitory_neurons,
            "excitatory_edges": res_stats.excitatory_edges,
            "inhibitory_edges": res_stats.inhibitory_edges,
        },
        "outputs": {
            "state_file": str(cfg.state_feature_path),
            "tsne_plot": str(cfg.tsne_plot_path),
            "summary_yaml": str(cfg.metrics_path),
            "loaded_model": (
                str(cfg.load_model_path) if cfg.load_model_path is not None else None
            ),
            "saved_model": (
                str(saved_model_path) if saved_model_path is not None else None
            ),
        },
        "classifier": (
            classifier_summary
            if classifier_summary is not None
            else {
                "enabled": False,
                "outputs": {
                    "classifier_model": None,
                    "classifier_metrics": None,
                    "confusion_matrix_plot": None,
                    "validation_predictions": None,
                },
            }
        ),
    }


def _save_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)
