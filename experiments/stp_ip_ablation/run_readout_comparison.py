from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ser_reservoir import ExperimentConfig
from ser_reservoir.classifier import (
    load_saved_states,
    stratified_holdout_split,
    supported_classifier_kinds,
    train_and_evaluate_classifier_with_split,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
READOUT_OUTPUT_ROOT = OUTPUT_ROOT / "readout_comparison"

CONDITIONS = (
    {
        "name": "baseline",
        "description": "STP + IP",
    },
    {
        "name": "no_stp",
        "description": "No STP",
    },
    {
        "name": "no_ip",
        "description": "No IP",
    },
    {
        "name": "no_stp_no_ip",
        "description": "No STP + No IP",
    },
)

CLASSIFIERS = (
    "logistic_regression",
    "nearest_centroid",
    "gaussian_nb",
    "ridge_classifier",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple classifier readouts using saved reservoir_states.npz files."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Per-emotion holdout ratio for validation.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=[item["name"] for item in CONDITIONS],
        help="Subset of ablation conditions to evaluate.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="*",
        default=list(CLASSIFIERS),
        help="Subset of classifier readouts to evaluate.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_eval_config(condition_name: str, classifier_kind: str, args: argparse.Namespace) -> ExperimentConfig:
    result_dir = READOUT_OUTPUT_ROOT / condition_name / classifier_kind
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


def resolve_condition_specs(names: list[str]) -> list[dict]:
    known = {item["name"]: item for item in CONDITIONS}
    resolved: list[dict] = []
    for name in names:
        key = str(name).strip()
        if key not in known:
            valid = ", ".join(sorted(known))
            raise ValueError(f"Unknown condition {name!r}. Choose from: {valid}.")
        resolved.append(known[key])
    return resolved


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


def get_state_file(condition_name: str) -> Path:
    state_file = OUTPUT_ROOT / condition_name / "processed" / "reservoir_states.npz"
    if not state_file.exists():
        raise FileNotFoundError(f"Saved state file not found: {state_file}")
    return state_file


def build_result_row(condition: dict, classifier_kind: str, metrics: dict, state_file: Path) -> dict:
    return {
        "condition": condition["name"],
        "description": condition["description"],
        "classifier_kind": classifier_kind,
        "model_type": metrics["model_type"],
        "num_samples": int(metrics["train_samples"] + metrics["validation_samples"]),
        "train_samples": int(metrics["train_samples"]),
        "validation_samples": int(metrics["validation_samples"]),
        "validation_accuracy": float(metrics["validation_accuracy"]),
        "validation_balanced_accuracy": float(metrics["validation_balanced_accuracy"]),
        "validation_macro_f1": float(metrics["validation_macro_f1"]),
        "state_file": str(state_file),
        "classifier_metrics": metrics["outputs"]["classifier_metrics"],
        "classifier_model": metrics["outputs"]["classifier_model"],
        "confusion_matrix_plot": metrics["outputs"]["confusion_matrix_plot"],
        "validation_predictions": metrics["outputs"]["validation_predictions"],
    }


def save_comparison_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "description",
        "classifier_kind",
        "model_type",
        "num_samples",
        "train_samples",
        "validation_samples",
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_f1",
        "state_file",
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


def save_comparison_report(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Readout Comparison Report",
        "",
        "| Condition | Classifier | Accuracy | Balanced Acc | Macro F1 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['classifier_kind']} | "
            f"{row['validation_accuracy']:.4f} | "
            f"{row['validation_balanced_accuracy']:.4f} | "
            f"{row['validation_macro_f1']:.4f} |"
        )
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join([*lines, ""]))


def print_table(rows: list[dict]) -> None:
    print("Readout Comparison")
    print("condition         classifier            accuracy   balanced_acc   macro_f1")
    print("---------------   --------------------  --------   ------------   --------")
    for row in rows:
        print(
            f"{row['condition']:<15}   "
            f"{row['classifier_kind']:<20}  "
            f"{row['validation_accuracy']:.4f}     "
            f"{row['validation_balanced_accuracy']:.4f}        "
            f"{row['validation_macro_f1']:.4f}"
        )


def main() -> None:
    args = parse_args()
    conditions = resolve_condition_specs(args.conditions)
    classifiers = resolve_classifier_kinds(args.classifiers)

    rows: list[dict] = []
    run_payload: list[dict] = []

    for condition in conditions:
        state_file = get_state_file(condition["name"])
        states, labels, paths = load_saved_states(state_file)
        split = stratified_holdout_split(
            np.asarray(labels, dtype=object),
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
        run_entry = {
            "condition": condition["name"],
            "description": condition["description"],
            "state_file": str(state_file),
            "classifiers": [],
        }

        for classifier_kind in classifiers:
            if not args.quiet:
                print(f"[Readout] {condition['name']} with {classifier_kind}")
            cfg = build_eval_config(condition["name"], classifier_kind, args)
            metrics = train_and_evaluate_classifier_with_split(states, labels, paths, split, cfg)
            rows.append(build_result_row(condition, classifier_kind, metrics, state_file))
            run_entry["classifiers"].append(
                {
                    "classifier_kind": classifier_kind,
                    "metrics": metrics,
                }
            )

        run_payload.append(run_entry)

    rows.sort(key=lambda item: (item["condition"], item["classifier_kind"]))

    csv_path = READOUT_OUTPUT_ROOT / "comparison_metrics.csv"
    yaml_path = READOUT_OUTPUT_ROOT / "comparison_summary.yaml"
    report_path = READOUT_OUTPUT_ROOT / "comparison_report.md"

    save_comparison_csv(rows, csv_path)
    save_comparison_yaml(
        {
            "settings": {
                "seed": args.seed,
                "validation_ratio": args.validation_ratio,
                "conditions": [item["name"] for item in conditions],
                "classifiers": classifiers,
                "source": "saved_reservoir_states_only",
            },
            "results": rows,
            "runs": run_payload,
        },
        yaml_path,
    )
    save_comparison_report(rows, report_path)

    print()
    print_table(rows)
    print()
    print(
        json.dumps(
            {
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
