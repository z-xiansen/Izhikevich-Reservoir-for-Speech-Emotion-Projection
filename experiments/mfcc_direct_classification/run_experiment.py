from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ser_reservoir import ExperimentConfig
from ser_reservoir.analysis import plot_embedding, run_tsne
from ser_reservoir.classifier import (
    stratified_holdout_split,
    supported_classifier_kinds,
    train_and_evaluate_classifier_with_split,
)
from ser_reservoir.data import collect_audio_samples, ensure_tess_dataset, extract_mfcc


EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
PROCESSED_ROOT = OUTPUT_ROOT / "processed"
RESULTS_ROOT = OUTPUT_ROOT / "results"
FEATURE_FILE = PROCESSED_ROOT / "mfcc_flat_features.npz"
TSNE_EMBEDDING_FILE = PROCESSED_ROOT / "mfcc_tsne_embedding.npz"
TSNE_PLOT_PATH = RESULTS_ROOT / "tsne_emotion_clusters.png"
BASELINE_METRICS_PATH = (
    ROOT
    / "experiments"
    / "stp_ip_ablation"
    / "outputs"
    / "baseline"
    / "results"
    / "classifier_metrics.yaml"
)
CLASSIFIERS = (
    "logistic_regression",
    "nearest_centroid",
    "gaussian_nb",
    "ridge_classifier",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct-MFCC classifier comparisons without the reservoir."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Per-emotion holdout ratio for validation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional total sample cap. Default uses all available samples.",
    )
    parser.add_argument(
        "--max-per-emotion",
        type=int,
        default=None,
        help="Optional per-emotion sample cap. Default uses all available samples.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="*",
        default=list(CLASSIFIERS),
        help="Subset of classifier readouts to evaluate.",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=str,
        default=str(BASELINE_METRICS_PATH),
        help="Reservoir baseline classifier_metrics.yaml used for comparison.",
    )
    parser.add_argument(
        "--refresh-features",
        action="store_true",
        help="Re-extract MFCC features even if the cached feature file already exists.",
    )
    parser.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Skip t-SNE projection and plotting for the direct MFCC features.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_data_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(
        workspace=ROOT,
        seed=args.seed,
        max_samples=args.max_samples,
        max_samples_per_emotion=args.max_per_emotion,
        verbose=not args.quiet,
    )
    return cfg.resolve()


def build_eval_config(classifier_kind: str, args: argparse.Namespace) -> ExperimentConfig:
    result_dir = RESULTS_ROOT / classifier_kind
    cfg = ExperimentConfig(
        workspace=ROOT,
        seed=args.seed,
        classifier_kind=classifier_kind,
        classifier_validation_ratio=args.validation_ratio,
        classifier_model_path=result_dir / "classifier_model.joblib",
        classifier_metrics_path=result_dir / "classifier_metrics.yaml",
        classifier_confusion_matrix_path=result_dir / "classifier_confusion_matrix.png",
        classifier_predictions_path=result_dir / "classifier_validation_predictions.csv",
    )
    return cfg.resolve()


def resolve_classifier_kinds(names: list[str]) -> list[str]:
    supported = set(supported_classifier_kinds())
    resolved: list[str] = []
    for name in names:
        key = str(name).strip().lower()
        if key not in supported:
            valid = ", ".join(sorted(supported))
            raise ValueError(f"Unknown classifier {name!r}. Choose from: {valid}.")
        resolved.append(key)
    return resolved


def load_or_build_features(
    cfg: ExperimentConfig,
    feature_file: Path,
    refresh: bool,
) -> tuple[np.ndarray, list[str], list[str], dict]:
    if feature_file.exists() and not refresh:
        with np.load(feature_file, allow_pickle=True) as data:
            features = data["features"].astype(np.float32, copy=True)
            labels = [str(x) for x in data["labels"].tolist()]
            paths = [str(x) for x in data["paths"].tolist()]
            metadata = {
                "n_samples": int(features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "n_mfcc": int(np.asarray(data["n_mfcc"]).item()),
                "target_frames": int(np.asarray(data["target_frames"]).item()),
                "frame_lengths_min": int(np.asarray(data["frame_lengths_min"]).item()),
                "frame_lengths_max": int(np.asarray(data["frame_lengths_max"]).item()),
                "frame_lengths_mean": float(np.asarray(data["frame_lengths_mean"]).item()),
                "cache_source": "loaded",
            }
        return features, labels, paths, metadata

    dataset_root = ensure_tess_dataset(cfg)
    samples = collect_audio_samples(dataset_root, cfg)
    if len(samples) < 3:
        raise RuntimeError("At least 3 samples are required.")

    mfcc_list: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    frame_lengths: list[int] = []

    iterator = tqdm(samples, desc="MFCC extraction", unit="sample", disable=not cfg.verbose)
    for sample in iterator:
        mfcc = extract_mfcc(sample.path, cfg)
        mfcc_list.append(mfcc)
        labels.append(sample.label)
        paths.append(str(sample.path))
        frame_lengths.append(int(mfcc.shape[1]))

    target_frames = max(frame_lengths)
    features = np.vstack(
        [
            _flatten_mfcc_with_padding(mfcc, target_frames=target_frames)
            for mfcc in mfcc_list
        ]
    ).astype(np.float32)

    feature_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_file,
        features=features,
        labels=np.array(labels, dtype=object),
        paths=np.array(paths, dtype=object),
        n_mfcc=np.int32(cfg.n_mfcc),
        target_frames=np.int32(target_frames),
        frame_lengths_min=np.int32(min(frame_lengths)),
        frame_lengths_max=np.int32(max(frame_lengths)),
        frame_lengths_mean=np.float32(np.mean(frame_lengths)),
    )

    metadata = {
        "dataset_root": str(dataset_root),
        "n_samples": len(labels),
        "feature_dim": int(features.shape[1]),
        "n_mfcc": int(cfg.n_mfcc),
        "target_frames": int(target_frames),
        "frame_lengths_min": int(min(frame_lengths)),
        "frame_lengths_max": int(max(frame_lengths)),
        "frame_lengths_mean": float(np.mean(frame_lengths)),
        "cache_source": "rebuilt",
    }
    return features, labels, paths, metadata


def _flatten_mfcc_with_padding(mfcc: np.ndarray, target_frames: int) -> np.ndarray:
    if mfcc.ndim != 2:
        raise ValueError(f"Expected MFCC shape [n_mfcc, frames], got {mfcc.shape}")
    n_mfcc, frames = mfcc.shape
    if frames > target_frames:
        mfcc = mfcc[:, :target_frames]
    elif frames < target_frames:
        pad = np.zeros((n_mfcc, target_frames - frames), dtype=np.float32)
        mfcc = np.concatenate([mfcc, pad], axis=1)
    return mfcc.reshape(1, -1)


def load_baseline_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Reservoir baseline metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        baseline = yaml.safe_load(f)
    required = ("model_type", "validation_accuracy", "validation_balanced_accuracy", "validation_macro_f1")
    missing = [key for key in required if key not in baseline]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Baseline metrics file is missing fields: {joined}")
    return baseline


def save_tsne_outputs(features: np.ndarray, labels: list[str], cfg: ExperimentConfig) -> dict:
    embedding = run_tsne(features, cfg)
    plot_embedding(
        embedding,
        labels,
        TSNE_PLOT_PATH,
        title="MFCC Direct Emotional Clusters (t-SNE)",
    )

    TSNE_EMBEDDING_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TSNE_EMBEDDING_FILE,
        embedding=embedding.astype(np.float32),
        labels=np.array(labels, dtype=object),
    )
    return {
        "tsne_embedding_file": str(TSNE_EMBEDDING_FILE),
        "tsne_plot": str(TSNE_PLOT_PATH),
    }


def build_result_row(classifier_kind: str, metrics: dict, baseline: dict) -> dict:
    return {
        "experiment": "mfcc_direct",
        "classifier_kind": classifier_kind,
        "model_type": metrics["model_type"],
        "train_samples": int(metrics["train_samples"]),
        "validation_samples": int(metrics["validation_samples"]),
        "validation_accuracy": float(metrics["validation_accuracy"]),
        "validation_balanced_accuracy": float(metrics["validation_balanced_accuracy"]),
        "validation_macro_f1": float(metrics["validation_macro_f1"]),
        "delta_accuracy_vs_reservoir_baseline": float(
            metrics["validation_accuracy"] - baseline["validation_accuracy"]
        ),
        "delta_balanced_accuracy_vs_reservoir_baseline": float(
            metrics["validation_balanced_accuracy"] - baseline["validation_balanced_accuracy"]
        ),
        "delta_macro_f1_vs_reservoir_baseline": float(
            metrics["validation_macro_f1"] - baseline["validation_macro_f1"]
        ),
        "classifier_metrics": metrics["outputs"]["classifier_metrics"],
        "classifier_model": metrics["outputs"]["classifier_model"],
        "confusion_matrix_plot": metrics["outputs"]["confusion_matrix_plot"],
        "validation_predictions": metrics["outputs"]["validation_predictions"],
    }


def build_baseline_row(baseline: dict) -> dict:
    outputs = baseline.get("outputs", {})
    return {
        "experiment": "reservoir_baseline",
        "classifier_kind": "logistic_regression",
        "model_type": str(baseline["model_type"]),
        "train_samples": int(baseline["train_samples"]),
        "validation_samples": int(baseline["validation_samples"]),
        "validation_accuracy": float(baseline["validation_accuracy"]),
        "validation_balanced_accuracy": float(baseline["validation_balanced_accuracy"]),
        "validation_macro_f1": float(baseline["validation_macro_f1"]),
        "delta_accuracy_vs_reservoir_baseline": 0.0,
        "delta_balanced_accuracy_vs_reservoir_baseline": 0.0,
        "delta_macro_f1_vs_reservoir_baseline": 0.0,
        "classifier_metrics": outputs.get("classifier_metrics"),
        "classifier_model": outputs.get("classifier_model"),
        "confusion_matrix_plot": outputs.get("confusion_matrix_plot"),
        "validation_predictions": outputs.get("validation_predictions"),
    }


def save_comparison_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "classifier_kind",
        "model_type",
        "train_samples",
        "validation_samples",
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_f1",
        "delta_accuracy_vs_reservoir_baseline",
        "delta_balanced_accuracy_vs_reservoir_baseline",
        "delta_macro_f1_vs_reservoir_baseline",
        "classifier_metrics",
        "classifier_model",
        "confusion_matrix_plot",
        "validation_predictions",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def save_comparison_report(rows: list[dict], metadata: dict, baseline_path: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MFCC Direct Classification Report",
        "",
        "## Feature Setup",
        "",
        f"- samples: {metadata['n_samples']}",
        f"- feature_dim: {metadata['feature_dim']}",
        f"- n_mfcc: {metadata['n_mfcc']}",
        f"- target_frames: {metadata['target_frames']}",
        (
            "- frame_lengths: "
            f"min={metadata['frame_lengths_min']}  "
            f"max={metadata['frame_lengths_max']}  "
            f"mean={metadata['frame_lengths_mean']:.2f}"
        ),
        f"- baseline_metrics: {baseline_path}",
        "",
        "## Comparison",
        "",
        "| Experiment | Classifier | Accuracy | Balanced Acc | Macro F1 | Delta Acc vs Reservoir Baseline |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['classifier_kind']} | "
            f"{row['validation_accuracy']:.4f} | "
            f"{row['validation_balanced_accuracy']:.4f} | "
            f"{row['validation_macro_f1']:.4f} | "
            f"{row['delta_accuracy_vs_reservoir_baseline']:+.4f} |"
        )
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join([*lines, ""]))


def print_table(rows: list[dict]) -> None:
    print("MFCC Direct vs Reservoir Baseline")
    print("experiment            classifier            accuracy   balanced_acc   macro_f1   delta_acc")
    print("--------------------  --------------------  --------   ------------   --------   ---------")
    for row in rows:
        print(
            f"{row['experiment']:<20}  "
            f"{row['classifier_kind']:<20}  "
            f"{row['validation_accuracy']:.4f}     "
            f"{row['validation_balanced_accuracy']:.4f}        "
            f"{row['validation_macro_f1']:.4f}     "
            f"{row['delta_accuracy_vs_reservoir_baseline']:+.4f}"
        )


def main() -> None:
    args = parse_args()
    classifiers = resolve_classifier_kinds(args.classifiers)
    data_cfg = build_data_config(args)
    features, labels, paths, metadata = load_or_build_features(
        cfg=data_cfg,
        feature_file=FEATURE_FILE,
        refresh=args.refresh_features,
    )
    tsne_outputs = (
        {
            "tsne_embedding_file": None,
            "tsne_plot": None,
        }
        if args.skip_tsne
        else save_tsne_outputs(features, labels, data_cfg)
    )
    split = stratified_holdout_split(
        np.asarray(labels, dtype=object),
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    baseline_path = Path(args.baseline_metrics)
    if not baseline_path.is_absolute():
        baseline_path = (ROOT / baseline_path).resolve()
    baseline_metrics = load_baseline_metrics(baseline_path)

    rows = [build_baseline_row(baseline_metrics)]
    classifier_runs: list[dict] = []

    for classifier_kind in classifiers:
        if not args.quiet:
            print(f"[MFCC] Training {classifier_kind}")
        cfg = build_eval_config(classifier_kind, args)
        metrics = train_and_evaluate_classifier_with_split(features, labels, paths, split, cfg)
        rows.append(build_result_row(classifier_kind, metrics, baseline_metrics))
        classifier_runs.append(
            {
                "classifier_kind": classifier_kind,
                "metrics": metrics,
            }
        )

    rows.sort(key=lambda item: (item["experiment"] != "reservoir_baseline", item["classifier_kind"]))

    csv_path = OUTPUT_ROOT / "comparison_metrics.csv"
    yaml_path = OUTPUT_ROOT / "comparison_summary.yaml"
    report_path = OUTPUT_ROOT / "comparison_report.md"

    save_comparison_csv(rows, csv_path)
    save_comparison_yaml(
        {
            "settings": {
                "seed": args.seed,
                "validation_ratio": args.validation_ratio,
                "max_samples": args.max_samples,
                "max_samples_per_emotion": args.max_per_emotion,
                "classifiers": classifiers,
                "feature_file": str(FEATURE_FILE),
                "skip_tsne": bool(args.skip_tsne),
            },
            "feature_metadata": metadata,
            "outputs": tsne_outputs,
            "reservoir_baseline_metrics_path": str(baseline_path),
            "results": rows,
            "classifier_runs": classifier_runs,
        },
        yaml_path,
    )
    save_comparison_report(rows, metadata, baseline_path, report_path)

    print()
    print_table(rows)
    print()
    print(
        json.dumps(
            {
                "feature_file": str(FEATURE_FILE),
                "tsne_plot": tsne_outputs["tsne_plot"],
                "tsne_embedding_file": tsne_outputs["tsne_embedding_file"],
                "comparison_csv": str(csv_path),
                "comparison_yaml": str(yaml_path),
                "comparison_report": str(report_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
