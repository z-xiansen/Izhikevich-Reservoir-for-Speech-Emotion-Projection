from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig


_CLASSIFIER_ALIASES = {
    "logistic_regression": "logistic_regression",
    "logistic": "logistic_regression",
    "lr": "logistic_regression",
    "nearest_centroid": "nearest_centroid",
    "centroid": "nearest_centroid",
    "nc": "nearest_centroid",
    "gaussian_nb": "gaussian_nb",
    "gnb": "gaussian_nb",
    "naive_bayes": "gaussian_nb",
    "ridge_classifier": "ridge_classifier",
    "ridge": "ridge_classifier",
    "rc": "ridge_classifier",
}

_CLASSIFIER_LABELS = {
    "logistic_regression": "standardized_logistic_regression",
    "nearest_centroid": "standardized_nearest_centroid",
    "gaussian_nb": "standardized_gaussian_nb",
    "ridge_classifier": "standardized_ridge_classifier",
}


def load_saved_states(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    state_path = Path(path)
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")

    with np.load(state_path, allow_pickle=True) as data:
        required = {"states", "labels", "paths"}
        missing = required.difference(data.files)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"State file is missing required arrays: {joined}")

        states = data["states"].astype(np.float32, copy=True)
        labels = data["labels"].tolist()
        paths = data["paths"].tolist()

    return states, [str(label) for label in labels], [str(p) for p in paths]


def train_and_evaluate_classifier(
    states: np.ndarray,
    labels: list[str],
    paths: list[str],
    cfg: ExperimentConfig,
) -> dict:
    if states.ndim != 2:
        raise ValueError(f"Classifier expects a 2D state matrix, got {states.shape}")
    if states.shape[0] != len(labels) or len(labels) != len(paths):
        raise ValueError("States, labels, and paths must describe the same number of samples.")

    y = np.asarray(labels, dtype=object)
    split = stratified_holdout_split(y, validation_ratio=cfg.classifier_validation_ratio, seed=cfg.seed)

    return train_and_evaluate_classifier_with_split(states, labels, paths, split, cfg)


def train_classifier_from_saved_states(
    state_path: Path,
    cfg: ExperimentConfig,
) -> dict:
    states, labels, paths = load_saved_states(state_path)
    return train_and_evaluate_classifier(states, labels, paths, cfg)


def train_and_evaluate_classifier_with_split(
    states: np.ndarray,
    labels: list[str],
    paths: list[str],
    split: dict[str, np.ndarray],
    cfg: ExperimentConfig,
) -> dict:
    if states.ndim != 2:
        raise ValueError(f"Classifier expects a 2D state matrix, got {states.shape}")
    if states.shape[0] != len(labels) or len(labels) != len(paths):
        raise ValueError("States, labels, and paths must describe the same number of samples.")

    y = np.asarray(labels, dtype=object)
    x_train = states[split["train_idx"]]
    y_train = y[split["train_idx"]]
    x_val = states[split["val_idx"]]
    y_val = y[split["val_idx"]]
    val_paths = [paths[i] for i in split["val_idx"].tolist()]

    classifier_kind = normalize_classifier_kind(cfg.classifier_kind)
    clf = build_classifier_pipeline(classifier_kind, cfg.seed)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)

    metrics = build_classifier_metrics(
        y=y,
        y_train=y_train,
        y_val=y_val,
        y_pred=y_pred,
        x_train=x_train,
        x_val=x_val,
        cfg=cfg,
    )

    save_classifier_model(cfg.classifier_model_path, clf)
    save_confusion_matrix_plot(
        cfg.classifier_confusion_matrix_path,
        np.asarray(metrics["confusion_matrix"], dtype=np.int32),
        metrics["labels"],
    )
    save_validation_predictions(cfg.classifier_predictions_path, val_paths, y_val, y_pred)
    save_classifier_metrics(cfg.classifier_metrics_path, metrics)
    return metrics


def normalize_classifier_kind(kind: str) -> str:
    key = str(kind).strip().lower()
    if key not in _CLASSIFIER_ALIASES:
        supported = ", ".join(sorted(_CLASSIFIER_LABELS))
        raise ValueError(f"Unsupported classifier kind {kind!r}. Choose from: {supported}.")
    return _CLASSIFIER_ALIASES[key]


def supported_classifier_kinds() -> list[str]:
    return sorted(_CLASSIFIER_LABELS)


def build_classifier_pipeline(kind: str, seed: int) -> Pipeline:
    normalized = normalize_classifier_kind(kind)
    estimator: object

    if normalized == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=4000,
            solver="lbfgs",
            random_state=seed,
        )
    elif normalized == "nearest_centroid":
        estimator = NearestCentroid()
    elif normalized == "gaussian_nb":
        estimator = GaussianNB()
    elif normalized == "ridge_classifier":
        estimator = RidgeClassifier(random_state=seed)
    else:
        raise AssertionError(f"Unhandled classifier kind: {normalized}")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def build_classifier_metrics(
    y: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_pred: np.ndarray,
    x_train: np.ndarray,
    x_val: np.ndarray,
    cfg: ExperimentConfig,
) -> dict:
    label_set = set(y.tolist())
    labels_present = [label for label in cfg.label_order if label in label_set]
    cm = confusion_matrix(y_val, y_pred, labels=labels_present)
    report = classification_report(
        y_val,
        y_pred,
        labels=labels_present,
        output_dict=True,
        zero_division=0,
    )

    classifier_kind = normalize_classifier_kind(cfg.classifier_kind)
    return {
        "enabled": True,
        "classifier_kind": classifier_kind,
        "model_type": _CLASSIFIER_LABELS[classifier_kind],
        "train_samples": int(x_train.shape[0]),
        "validation_samples": int(x_val.shape[0]),
        "train_distribution": dict(sorted(Counter(y_train.tolist()).items())),
        "validation_distribution": dict(sorted(Counter(y_val.tolist()).items())),
        "validation_accuracy": float(accuracy_score(y_val, y_pred)),
        "validation_balanced_accuracy": float(balanced_accuracy_score(y_val, y_pred)),
        "validation_macro_f1": float(f1_score(y_val, y_pred, average="macro")),
        "per_class_metrics": _sanitize_metric_tree(report),
        "labels": labels_present,
        "confusion_matrix": cm.astype(int).tolist(),
        "outputs": {
            "classifier_model": str(cfg.classifier_model_path),
            "classifier_metrics": str(cfg.classifier_metrics_path),
            "confusion_matrix_plot": str(cfg.classifier_confusion_matrix_path),
            "validation_predictions": str(cfg.classifier_predictions_path),
        },
    }


def stratified_holdout_split(
    labels: np.ndarray,
    validation_ratio: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if labels.ndim != 1:
        raise ValueError(f"Labels must be a 1D array, got {labels.shape}")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be between 0 and 1, got {validation_ratio}")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for label in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == label)
        if idx.size < 2:
            raise ValueError(
                f"Need at least 2 samples for label {label!r} to build train/validation splits."
            )
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        n_val = int(round(idx.size * validation_ratio))
        n_val = min(max(n_val, 1), idx.size - 1)
        val_parts.append(np.sort(shuffled[:n_val]))
        train_parts.append(np.sort(shuffled[n_val:]))

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    return {
        "train_idx": np.sort(train_idx.astype(np.int32)),
        "val_idx": np.sort(val_idx.astype(np.int32)),
    }


def save_classifier_model(path: Path, model: Pipeline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_classifier_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=False, allow_unicode=True)


def save_validation_predictions(
    path: Path,
    sample_paths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "predicted_label", "correct"])
        for sample_path, true_label, pred_label in zip(sample_paths, y_true.tolist(), y_pred.tolist()):
            writer.writerow([sample_path, true_label, pred_label, str(true_label == pred_label).lower()])


def save_confusion_matrix_plot(path: Path, cm: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Validation Confusion Matrix", fontsize=13, weight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def format_classifier_report(metrics: dict) -> str:
    lines = [
        "Classifier Report",
        f"Model: {metrics['model_type']}",
        (
            "Split: "
            f"train={metrics['train_samples']}  "
            f"validation={metrics['validation_samples']}"
        ),
        (
            "Validation: "
            f"accuracy={metrics['validation_accuracy']:.4f}  "
            f"balanced_accuracy={metrics['validation_balanced_accuracy']:.4f}  "
            f"macro_f1={metrics['validation_macro_f1']:.4f}"
        ),
        "",
        _format_class_table(metrics),
    ]

    confusion_notes = _summarize_confusion_errors(metrics)
    if confusion_notes:
        lines.extend(["", "Main Confusions:"])
        lines.extend(confusion_notes)

    outputs = metrics.get("outputs", {})
    if outputs:
        lines.extend(
            [
                "",
                "Outputs:",
                f"- model: {outputs.get('classifier_model')}",
                f"- metrics: {outputs.get('classifier_metrics')}",
                f"- confusion_matrix: {outputs.get('confusion_matrix_plot')}",
                f"- validation_predictions: {outputs.get('validation_predictions')}",
            ]
        )

    return "\n".join(lines)


def _format_class_table(metrics: dict) -> str:
    class_metrics = metrics.get("per_class_metrics", {})
    labels = metrics.get("labels", [])
    rows: list[tuple[str, str, str, str, str]] = []

    for label in labels:
        item = class_metrics.get(label, {})
        rows.append(
            (
                str(label),
                f"{float(item.get('precision', 0.0)):.4f}",
                f"{float(item.get('recall', 0.0)):.4f}",
                f"{float(item.get('f1-score', 0.0)):.4f}",
                str(int(round(float(item.get('support', 0.0))))),
            )
        )

    headers = ("label", "precision", "recall", "f1", "support")
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(widths[i], len(row[i])) for i in range(len(headers))]

    header_line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    divider_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    body_lines = [
        "  ".join(row[i].ljust(widths[i]) for i in range(len(headers)))
        for row in rows
    ]
    return "\n".join(["Per-class Metrics", header_line, divider_line, *body_lines])


def _summarize_confusion_errors(metrics: dict) -> list[str]:
    labels = metrics.get("labels", [])
    cm = np.asarray(metrics.get("confusion_matrix", []), dtype=np.int32)
    if cm.size == 0 or cm.shape != (len(labels), len(labels)):
        return []

    notes: list[tuple[int, str]] = []
    for true_idx, true_label in enumerate(labels):
        for pred_idx, pred_label in enumerate(labels):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count > 0:
                notes.append((count, f"- {true_label} -> {pred_label}: {count}"))

    notes.sort(key=lambda item: (-item[0], item[1]))
    return [text for _, text in notes[:5]]


def _sanitize_metric_tree(tree: dict) -> dict:
    clean: dict = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            clean[str(key)] = _sanitize_metric_tree(value)
        elif isinstance(value, (np.floating, float)):
            clean[str(key)] = float(value)
        elif isinstance(value, (np.integer, int)):
            clean[str(key)] = int(value)
        else:
            clean[str(key)] = value
    return clean
