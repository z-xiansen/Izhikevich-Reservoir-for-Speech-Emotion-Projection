from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ser_reservoir import ExperimentConfig
from ser_reservoir.classifier import (
    format_classifier_report,
    supported_classifier_kinds,
    train_classifier_from_saved_states,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the reservoir feature classifier from a saved reservoir_states.npz file."
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="data/processed/reservoir_states.npz",
        help="Path to a saved reservoir_states.npz file.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--classifier",
        type=str,
        default="logistic_regression",
        choices=supported_classifier_kinds(),
        help="Which classifier head to train on top of the saved state vectors.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Per-emotion holdout ratio for validation.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="results/reservoir_classifier.joblib",
        help="Where to save the trained classifier.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="results/classifier_metrics.yaml",
        help="Where to save classifier metrics.",
    )
    parser.add_argument(
        "--confusion-matrix-path",
        type=str,
        default="results/classifier_confusion_matrix.png",
        help="Where to save the confusion matrix image.",
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        default="results/classifier_validation_predictions.csv",
        help="Where to save validation predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        workspace=ROOT,
        seed=args.seed,
        classifier_kind=args.classifier,
        classifier_validation_ratio=args.validation_ratio,
        classifier_model_path=Path(args.model_path),
        classifier_metrics_path=Path(args.metrics_path),
        classifier_confusion_matrix_path=Path(args.confusion_matrix_path),
        classifier_predictions_path=Path(args.predictions_path),
    )
    cfg.resolve()

    state_path = Path(args.state_file)
    if not state_path.is_absolute():
        state_path = (ROOT / state_path).resolve()

    metrics = train_classifier_from_saved_states(state_path, cfg)
    print(format_classifier_report(metrics))
    print()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
