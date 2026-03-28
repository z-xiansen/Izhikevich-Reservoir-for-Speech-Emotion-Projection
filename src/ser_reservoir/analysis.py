from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig


def run_tsne(states: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    if states.ndim != 2:
        raise ValueError(f"State matrix must be 2D, got {states.shape}")
    n_samples = states.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 samples for t-SNE projection.")

    # t-SNE works better after scaling because state features contain mixed units.
    z = StandardScaler().fit_transform(states)
    max_perplexity = max(5.0, min(cfg.tsne_perplexity, (n_samples - 1) / 3.0))

    tsne = TSNE(
        n_components=2,
        perplexity=max_perplexity,
        learning_rate=cfg.tsne_learning_rate,
        max_iter=cfg.tsne_n_iter,
        init="pca",
        random_state=cfg.seed,
        metric="euclidean",
        verbose=1 if cfg.verbose else 0,
    )
    embedding = tsne.fit_transform(z)
    return embedding.astype(np.float32)


def plot_embedding(
    embedding: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str = "Reservoir Emotional State Clusters (t-SNE)",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if embedding.shape[1] != 2:
        raise ValueError(f"Embedding must be [N,2], got {embedding.shape}")
    if len(labels) != embedding.shape[0]:
        raise ValueError("Number of labels must match embedding rows.")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
    palette = sns.color_palette("tab10", n_colors=len(set(labels)))
    unique_labels = sorted(set(labels))
    palette_map = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}

    for label in unique_labels:
        idx = [i for i, lbl in enumerate(labels) if lbl == label]
        points = embedding[idx]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=24,
            alpha=0.82,
            c=[palette_map[label]],
            label=label,
            edgecolors="none",
        )

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(title="Emotion", frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_states(
    path: Path,
    states: np.ndarray,
    embedding: np.ndarray,
    labels: list[str],
    paths: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=states.astype(np.float32),
        embedding=embedding.astype(np.float32),
        labels=np.array(labels, dtype=object),
        paths=np.array(paths, dtype=object),
    )

