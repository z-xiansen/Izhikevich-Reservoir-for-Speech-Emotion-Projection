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
from ser_reservoir.analysis import plot_embedding, run_tsne_with_metadata, save_tsne_embedding
from ser_reservoir.classifier import (
    stratified_holdout_split,
    supported_classifier_kinds,
    train_and_evaluate_classifier_with_split,
)
from ser_reservoir.data import (
    RAW_WAVEFORM_INPUT_CHANNELS,
    RAW_WAVEFORM_TARGET_LENGTH,
    collect_audio_samples,
    ensure_tess_dataset,
    extract_raw_waveform_feature,
    raw_waveform_feature_metadata,
    raw_waveform_feature_to_reservoir_input,
)
from ser_reservoir.reservoir import IzhikevichReservoir


EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
PROCESSED_ROOT = OUTPUT_ROOT / "processed"
RAW_FEATURE_FILE = PROCESSED_ROOT / "raw_waveform_flat_features.npz"
SPLIT_FILE = PROCESSED_ROOT / "split_indices.npz"
RAW_DIRECT_TSNE_EMBEDDING_FILE = PROCESSED_ROOT / "raw_direct_tsne_embedding.npz"
RAW_RESERVOIR_TSNE_EMBEDDING_FILE = PROCESSED_ROOT / "raw_reservoir_tsne_embedding.npz"

RAW_DIRECT_NAME = "raw_direct"
RAW_RESERVOIR_NAME = "raw_reservoir"
RAW_DIRECT_TSNE_PLOT = OUTPUT_ROOT / RAW_DIRECT_NAME / "results" / "tsne_emotion_clusters.png"
RAW_RESERVOIR_TSNE_PLOT = OUTPUT_ROOT / RAW_RESERVOIR_NAME / "results" / "tsne_emotion_clusters.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare direct raw-waveform classification against raw-waveform + reservoir."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Per-emotion holdout ratio for validation.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="logistic_regression",
        choices=supported_classifier_kinds(),
        help="Classifier head shared by raw_direct and raw_reservoir.",
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
        "--no-limit",
        action="store_true",
        help="Ignore sample caps and use all available TESS samples.",
    )
    parser.add_argument("--frame-repeat", type=int, default=2)
    parser.add_argument("--input-gain", type=float, default=14.0)
    parser.add_argument(
        "--refresh-features",
        action="store_true",
        help="Rebuild the raw-waveform feature cache even if it already exists.",
    )
    parser.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Skip t-SNE projection and plotting for raw_direct and raw_reservoir.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_data_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(
        workspace=ROOT,
        seed=args.seed,
        verbose=not args.quiet,
    )
    if args.no_limit:
        cfg.max_samples = None
        cfg.max_samples_per_emotion = None
    else:
        cfg.max_samples = args.max_samples
        cfg.max_samples_per_emotion = args.max_per_emotion
    return cfg.resolve()


def build_eval_config(experiment_name: str, args: argparse.Namespace) -> ExperimentConfig:
    result_dir = OUTPUT_ROOT / experiment_name / "results"
    cfg = ExperimentConfig(
        workspace=ROOT,
        seed=args.seed,
        classifier_kind=args.classifier,
        classifier_validation_ratio=args.validation_ratio,
        classifier_model_path=result_dir / "classifier_model.joblib",
        classifier_metrics_path=result_dir / "classifier_metrics.yaml",
        classifier_confusion_matrix_path=result_dir / "classifier_confusion_matrix.png",
        classifier_predictions_path=result_dir / "classifier_validation_predictions.csv",
    )
    return cfg.resolve()


def build_reservoir_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(
        workspace=ROOT,
        seed=args.seed,
        frame_repeat=args.frame_repeat,
        input_gain=args.input_gain,
        verbose=not args.quiet,
    )
    return cfg.resolve()


def build_tsne_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(
        workspace=ROOT,
        seed=args.seed,
        verbose=not args.quiet,
    )
    return cfg.resolve()


def load_or_build_raw_features(
    cfg: ExperimentConfig,
    feature_file: Path,
    refresh: bool,
) -> tuple[np.ndarray, list[str], list[str], dict]:
    expected_meta = raw_waveform_feature_metadata(
        target_length=RAW_WAVEFORM_TARGET_LENGTH,
        input_channels=RAW_WAVEFORM_INPUT_CHANNELS,
        sample_rate=cfg.sample_rate,
    )

    if feature_file.exists() and not refresh:
        with np.load(feature_file, allow_pickle=True) as data:
            if _feature_cache_matches(data, cfg, expected_meta):
                features = data["features"].astype(np.float32, copy=True)
                labels = [str(x) for x in data["labels"].tolist()]
                paths = [str(x) for x in data["paths"].tolist()]
                metadata = {
                    "dataset_root": _optional_string_item(data, "dataset_root"),
                    "n_samples": int(features.shape[0]),
                    "feature_dim": int(features.shape[1]),
                    "sample_rate": int(np.asarray(data["sample_rate"]).item()),
                    "target_length": int(np.asarray(data["target_length"]).item()),
                    "input_channels": int(np.asarray(data["input_channels"]).item()),
                    "target_frames": int(np.asarray(data["target_frames"]).item()),
                    "selection_seed": int(np.asarray(data["selection_seed"]).item()),
                    "max_samples": _restore_optional_limit(
                        int(np.asarray(data["max_samples"]).item())
                    ),
                    "max_samples_per_emotion": _restore_optional_limit(
                        int(np.asarray(data["max_samples_per_emotion"]).item())
                    ),
                    "cache_source": "loaded",
                }
                return features, labels, paths, metadata

    dataset_root = ensure_tess_dataset(cfg)
    samples = collect_audio_samples(dataset_root, cfg)
    if len(samples) < 3:
        raise RuntimeError("At least 3 samples are required.")

    features: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []

    iterator = tqdm(samples, desc="Raw waveform extraction", unit="sample", disable=not cfg.verbose)
    for sample in iterator:
        feature = extract_raw_waveform_feature(
            sample.path,
            cfg,
            target_length=RAW_WAVEFORM_TARGET_LENGTH,
        )
        features.append(feature.reshape(1, -1))
        labels.append(sample.label)
        paths.append(str(sample.path))

    feature_matrix = np.vstack(features).astype(np.float32)
    feature_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_file,
        features=feature_matrix,
        labels=np.array(labels, dtype=object),
        paths=np.array(paths, dtype=object),
        dataset_root=np.array(str(dataset_root)),
        sample_rate=np.int32(expected_meta["sample_rate"]),
        target_length=np.int32(expected_meta["target_length"]),
        input_channels=np.int32(expected_meta["input_channels"]),
        target_frames=np.int32(expected_meta["target_frames"]),
        selection_seed=np.int32(cfg.seed),
        max_samples=np.int32(_serialize_optional_limit(cfg.max_samples)),
        max_samples_per_emotion=np.int32(_serialize_optional_limit(cfg.max_samples_per_emotion)),
    )

    metadata = {
        "dataset_root": str(dataset_root),
        "n_samples": int(feature_matrix.shape[0]),
        "feature_dim": int(feature_matrix.shape[1]),
        "sample_rate": int(expected_meta["sample_rate"]),
        "target_length": int(expected_meta["target_length"]),
        "input_channels": int(expected_meta["input_channels"]),
        "target_frames": int(expected_meta["target_frames"]),
        "selection_seed": int(cfg.seed),
        "max_samples": cfg.max_samples,
        "max_samples_per_emotion": cfg.max_samples_per_emotion,
        "cache_source": "rebuilt",
    }
    return feature_matrix, labels, paths, metadata


def build_raw_reservoir_states(
    features: np.ndarray,
    labels: list[str],
    paths: list[str],
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, dict]:
    if features.ndim != 2:
        raise ValueError(f"Expected features to be 2D, got {features.shape}")

    reservoir = IzhikevichReservoir(cfg)
    states: list[np.ndarray] = []
    global_rates: list[float] = []

    iterator = tqdm(features, desc="Raw waveform reservoir", unit="sample", disable=not cfg.verbose)
    for feature in iterator:
        reservoir_input = raw_waveform_feature_to_reservoir_input(
            feature,
            input_channels=RAW_WAVEFORM_INPUT_CHANNELS,
        )
        state, global_rate = reservoir.present(reservoir_input)
        states.append(state)
        global_rates.append(global_rate)

    state_matrix = np.vstack(states).astype(np.float32)
    state_file = OUTPUT_ROOT / RAW_RESERVOIR_NAME / "processed" / "reservoir_states.npz"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        state_file,
        states=state_matrix,
        labels=np.array(labels, dtype=object),
        paths=np.array(paths, dtype=object),
    )

    summary = {
        "state_file": str(state_file),
        "num_state_features": int(state_matrix.shape[1]),
        "mean_global_firing_rate": float(np.mean(global_rates)),
        "std_global_firing_rate": float(np.std(global_rates)),
    }
    return state_matrix, summary


def save_split_indices(split: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_idx=split["train_idx"].astype(np.int32),
        val_idx=split["val_idx"].astype(np.int32),
    )


def save_tsne_outputs(
    features: np.ndarray,
    labels: list[str],
    cfg: ExperimentConfig,
    embedding_path: Path,
    plot_path: Path,
    title: str,
) -> dict[str, object]:
    tsne_result = run_tsne_with_metadata(features, cfg)
    plot_embedding(tsne_result.embedding, labels, plot_path, title=title)
    save_tsne_embedding(
        embedding_path,
        tsne_result.embedding,
        labels,
        metadata=tsne_result.metadata,
    )
    return {
        "tsne_embedding_file": str(embedding_path),
        "tsne_plot": str(plot_path),
        "tsne_metadata": tsne_result.metadata,
    }


def build_result_row(
    experiment_name: str,
    metrics: dict,
    baseline_metrics: dict,
    input_representation: str,
    extra_outputs: dict[str, str | None] | None = None,
) -> dict:
    outputs = metrics.get("outputs", {})
    row = {
        "experiment": experiment_name,
        "classifier_kind": metrics["classifier_kind"],
        "model_type": metrics["model_type"],
        "input_representation": input_representation,
        "train_samples": int(metrics["train_samples"]),
        "validation_samples": int(metrics["validation_samples"]),
        "validation_accuracy": float(metrics["validation_accuracy"]),
        "validation_balanced_accuracy": float(metrics["validation_balanced_accuracy"]),
        "validation_macro_f1": float(metrics["validation_macro_f1"]),
        "delta_accuracy_vs_raw_direct": float(
            metrics["validation_accuracy"] - baseline_metrics["validation_accuracy"]
        ),
        "delta_balanced_accuracy_vs_raw_direct": float(
            metrics["validation_balanced_accuracy"]
            - baseline_metrics["validation_balanced_accuracy"]
        ),
        "delta_macro_f1_vs_raw_direct": float(
            metrics["validation_macro_f1"] - baseline_metrics["validation_macro_f1"]
        ),
        "classifier_metrics": outputs.get("classifier_metrics"),
        "classifier_model": outputs.get("classifier_model"),
        "confusion_matrix_plot": outputs.get("confusion_matrix_plot"),
        "validation_predictions": outputs.get("validation_predictions"),
    }
    if extra_outputs:
        row.update(extra_outputs)
    return row


def save_comparison_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "classifier_kind",
        "model_type",
        "input_representation",
        "train_samples",
        "validation_samples",
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_f1",
        "delta_accuracy_vs_raw_direct",
        "delta_balanced_accuracy_vs_raw_direct",
        "delta_macro_f1_vs_raw_direct",
        "classifier_metrics",
        "classifier_model",
        "confusion_matrix_plot",
        "validation_predictions",
        "feature_file",
        "state_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def save_comparison_report(
    rows: list[dict],
    feature_metadata: dict,
    split_path: Path,
    reservoir_summary: dict,
    tsne_outputs: dict[str, dict[str, object] | None],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Raw Waveform Comparison Report",
        "",
        "## Raw Input Definition",
        "",
        "- `soundfile.read()` -> mono downmix if needed",
        "- polyphase resample to `22050 Hz`",
        "- peak-normalize waveform to `[-1, 1]`",
        "- linearly resample each utterance to `10320` samples",
        "- shift the fixed-length vector to `[0, 1]`",
        "- `raw_direct`: feed the `10320`-D vector directly into the classifier",
        "- `raw_reservoir`: reshape with `flat.reshape(258, 40).T` to `[40, 258]`, then feed the reservoir",
        "",
        "## Feature Setup",
        "",
        f"- samples: {feature_metadata['n_samples']}",
        f"- sample_rate: {feature_metadata['sample_rate']}",
        f"- target_length: {feature_metadata['target_length']}",
        f"- input_channels: {feature_metadata['input_channels']}",
        f"- target_frames: {feature_metadata['target_frames']}",
        f"- feature_file: {RAW_FEATURE_FILE}",
        f"- split_file: {split_path}",
        f"- cache_source: {feature_metadata['cache_source']}",
        "",
        "## Reservoir Summary",
        "",
        f"- state_file: {reservoir_summary['state_file']}",
        f"- num_state_features: {reservoir_summary['num_state_features']}",
        f"- mean_global_firing_rate: {reservoir_summary['mean_global_firing_rate']:.6f}",
        f"- std_global_firing_rate: {reservoir_summary['std_global_firing_rate']:.6f}",
        "",
        "## t-SNE Outputs",
        "",
        f"- raw_direct_embedding: {_tsne_output_value(tsne_outputs, RAW_DIRECT_NAME, 'tsne_embedding_file')}",
        f"- raw_direct_plot: {_tsne_output_value(tsne_outputs, RAW_DIRECT_NAME, 'tsne_plot')}",
        f"- raw_direct_kl_divergence: {_tsne_metric_value(tsne_outputs, RAW_DIRECT_NAME, 'kl_divergence')}",
        f"- raw_direct_perplexity_used: {_tsne_metric_value(tsne_outputs, RAW_DIRECT_NAME, 'perplexity_used')}",
        f"- raw_direct_n_iter_completed: {_tsne_metric_value(tsne_outputs, RAW_DIRECT_NAME, 'n_iter_completed')}",
        f"- raw_reservoir_embedding: {_tsne_output_value(tsne_outputs, RAW_RESERVOIR_NAME, 'tsne_embedding_file')}",
        f"- raw_reservoir_plot: {_tsne_output_value(tsne_outputs, RAW_RESERVOIR_NAME, 'tsne_plot')}",
        f"- raw_reservoir_kl_divergence: {_tsne_metric_value(tsne_outputs, RAW_RESERVOIR_NAME, 'kl_divergence')}",
        f"- raw_reservoir_perplexity_used: {_tsne_metric_value(tsne_outputs, RAW_RESERVOIR_NAME, 'perplexity_used')}",
        f"- raw_reservoir_n_iter_completed: {_tsne_metric_value(tsne_outputs, RAW_RESERVOIR_NAME, 'n_iter_completed')}",
        "",
        "## Comparison",
        "",
        "| Experiment | Accuracy | Balanced Acc | Macro F1 | Delta Acc vs Raw Direct |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | "
            f"{row['validation_accuracy']:.4f} | "
            f"{row['validation_balanced_accuracy']:.4f} | "
            f"{row['validation_macro_f1']:.4f} | "
            f"{row['delta_accuracy_vs_raw_direct']:+.4f} |"
        )

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join([*lines, ""]))


def print_table(rows: list[dict]) -> None:
    print("Raw Waveform Comparison")
    print("experiment        accuracy   balanced_acc   macro_f1   delta_acc")
    print("---------------   --------   ------------   --------   ---------")
    for row in rows:
        print(
            f"{row['experiment']:<15}   "
            f"{row['validation_accuracy']:.4f}     "
            f"{row['validation_balanced_accuracy']:.4f}        "
            f"{row['validation_macro_f1']:.4f}     "
            f"{row['delta_accuracy_vs_raw_direct']:+.4f}"
        )


def _feature_cache_matches(
    data: np.lib.npyio.NpzFile,
    cfg: ExperimentConfig,
    expected_meta: dict[str, int],
) -> bool:
    required = {
        "features",
        "labels",
        "paths",
        "sample_rate",
        "target_length",
        "input_channels",
        "target_frames",
        "selection_seed",
        "max_samples",
        "max_samples_per_emotion",
    }
    if not required.issubset(set(data.files)):
        return False

    checks = {
        "sample_rate": expected_meta["sample_rate"],
        "target_length": expected_meta["target_length"],
        "input_channels": expected_meta["input_channels"],
        "target_frames": expected_meta["target_frames"],
        "selection_seed": int(cfg.seed),
        "max_samples": _serialize_optional_limit(cfg.max_samples),
        "max_samples_per_emotion": _serialize_optional_limit(cfg.max_samples_per_emotion),
    }
    for key, expected in checks.items():
        actual = int(np.asarray(data[key]).item())
        if actual != int(expected):
            return False
    return True


def _serialize_optional_limit(value: int | None) -> int:
    return -1 if value is None else int(value)


def _restore_optional_limit(value: int) -> int | None:
    return None if value < 0 else int(value)


def _optional_string_item(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data.files:
        return None
    return str(np.asarray(data[key]).item())


def _tsne_output_value(
    tsne_outputs: dict[str, dict[str, object] | None],
    experiment_name: str,
    key: str,
) -> str | None:
    experiment_outputs = tsne_outputs.get(experiment_name)
    if experiment_outputs is None:
        return None
    value = experiment_outputs.get(key)
    return None if value is None else str(value)


def _tsne_metric_value(
    tsne_outputs: dict[str, dict[str, object] | None],
    experiment_name: str,
    key: str,
) -> float | int | None:
    experiment_outputs = tsne_outputs.get(experiment_name)
    if experiment_outputs is None:
        return None
    metadata = experiment_outputs.get("tsne_metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(key)
    if isinstance(value, (float, int)):
        return value
    return None


def main() -> None:
    args = parse_args()
    data_cfg = build_data_config(args)
    features, labels, paths, feature_metadata = load_or_build_raw_features(
        cfg=data_cfg,
        feature_file=RAW_FEATURE_FILE,
        refresh=args.refresh_features,
    )
    tsne_cfg = build_tsne_config(args)

    split = stratified_holdout_split(
        np.asarray(labels, dtype=object),
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    save_split_indices(split, SPLIT_FILE)

    direct_cfg = build_eval_config(RAW_DIRECT_NAME, args)
    direct_metrics = train_and_evaluate_classifier_with_split(
        features,
        labels,
        paths,
        split,
        direct_cfg,
    )

    reservoir_cfg = build_reservoir_config(args)
    reservoir_states, reservoir_summary = build_raw_reservoir_states(
        features,
        labels,
        paths,
        reservoir_cfg,
    )
    reservoir_eval_cfg = build_eval_config(RAW_RESERVOIR_NAME, args)
    reservoir_metrics = train_and_evaluate_classifier_with_split(
        reservoir_states,
        labels,
        paths,
        split,
        reservoir_eval_cfg,
    )

    tsne_outputs: dict[str, dict[str, object] | None]
    if args.skip_tsne:
        tsne_outputs = {
            RAW_DIRECT_NAME: None,
            RAW_RESERVOIR_NAME: None,
        }
    else:
        tsne_outputs = {
            RAW_DIRECT_NAME: save_tsne_outputs(
                features=features,
                labels=labels,
                cfg=tsne_cfg,
                embedding_path=RAW_DIRECT_TSNE_EMBEDDING_FILE,
                plot_path=RAW_DIRECT_TSNE_PLOT,
                title="Raw Waveform Direct Emotional Clusters (t-SNE)",
            ),
            RAW_RESERVOIR_NAME: save_tsne_outputs(
                features=reservoir_states,
                labels=labels,
                cfg=tsne_cfg,
                embedding_path=RAW_RESERVOIR_TSNE_EMBEDDING_FILE,
                plot_path=RAW_RESERVOIR_TSNE_PLOT,
                title="Raw Waveform Reservoir Emotional Clusters (t-SNE)",
            ),
        }

    rows = [
        build_result_row(
            experiment_name=RAW_DIRECT_NAME,
            metrics=direct_metrics,
            baseline_metrics=direct_metrics,
            input_representation="raw_waveform_flat_10320",
            extra_outputs={
                "feature_file": str(RAW_FEATURE_FILE),
                "state_file": None,
            },
        ),
        build_result_row(
            experiment_name=RAW_RESERVOIR_NAME,
            metrics=reservoir_metrics,
            baseline_metrics=direct_metrics,
            input_representation="raw_waveform_40x258_to_reservoir_800d_state",
            extra_outputs={
                "feature_file": str(RAW_FEATURE_FILE),
                "state_file": reservoir_summary["state_file"],
            },
        ),
    ]

    csv_path = OUTPUT_ROOT / "comparison_metrics.csv"
    yaml_path = OUTPUT_ROOT / "comparison_summary.yaml"
    report_path = OUTPUT_ROOT / "comparison_report.md"

    save_comparison_csv(rows, csv_path)
    save_comparison_yaml(
        {
            "settings": {
                "seed": args.seed,
                "validation_ratio": args.validation_ratio,
                "classifier": args.classifier,
                "frame_repeat": args.frame_repeat,
                "input_gain": args.input_gain,
                "max_samples": None if args.no_limit else args.max_samples,
                "max_samples_per_emotion": None if args.no_limit else args.max_per_emotion,
                "use_full_dataset": bool(
                    args.no_limit
                    or (args.max_samples is None and args.max_per_emotion is None)
                ),
            },
            "feature_metadata": feature_metadata,
            "outputs": {
                "feature_file": str(RAW_FEATURE_FILE),
                "split_file": str(SPLIT_FILE),
                "comparison_csv": str(csv_path),
                "comparison_yaml": str(yaml_path),
                "comparison_report": str(report_path),
                "tsne_outputs": tsne_outputs,
            },
            "split": {
                "train_size": int(split["train_idx"].size),
                "validation_size": int(split["val_idx"].size),
            },
            "results": rows,
            "raw_reservoir_summary": reservoir_summary,
        },
        yaml_path,
    )
    save_comparison_report(
        rows,
        feature_metadata,
        SPLIT_FILE,
        reservoir_summary,
        tsne_outputs,
        report_path,
    )

    print()
    print_table(rows)
    print()
    print(
        json.dumps(
            {
                "feature_file": str(RAW_FEATURE_FILE),
                "split_file": str(SPLIT_FILE),
                "raw_reservoir_state_file": reservoir_summary["state_file"],
                "raw_direct_tsne_plot": _tsne_output_value(
                    tsne_outputs, RAW_DIRECT_NAME, "tsne_plot"
                ),
                "raw_reservoir_tsne_plot": _tsne_output_value(
                    tsne_outputs, RAW_RESERVOIR_NAME, "tsne_plot"
                ),
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
