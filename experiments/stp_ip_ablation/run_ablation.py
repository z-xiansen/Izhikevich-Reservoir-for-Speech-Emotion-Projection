from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ser_reservoir import ExperimentConfig, run_pipeline


EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"

CONDITIONS = (
    {
        "name": "baseline",
        "enable_stp": True,
        "enable_intrinsic_plasticity": True,
        "description": "STP + IP",
    },
    {
        "name": "no_stp",
        "enable_stp": False,
        "enable_intrinsic_plasticity": True,
        "description": "No STP",
    },
    {
        "name": "no_ip",
        "enable_stp": True,
        "enable_intrinsic_plasticity": False,
        "description": "No IP",
    },
    {
        "name": "no_stp_no_ip",
        "enable_stp": False,
        "enable_intrinsic_plasticity": False,
        "description": "No STP + No IP",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run STP/IP ablation experiments for the SER reservoir."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples", type=int, default=420)
    parser.add_argument("--max-per-emotion", type=int, default=70)
    parser.add_argument("--no-limit", action="store_true", help="Use all available samples.")
    parser.add_argument("--frame-repeat", type=int, default=2)
    parser.add_argument("--input-gain", type=float, default=14.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_config(condition: dict, args: argparse.Namespace) -> ExperimentConfig:
    condition_root = OUTPUT_ROOT / condition["name"]
    results_dir = condition_root / "results"
    processed_dir = condition_root / "processed"

    cfg = ExperimentConfig(
        workspace=ROOT,
        seed=args.seed,
        frame_repeat=args.frame_repeat,
        input_gain=args.input_gain,
        verbose=not args.quiet,
        train_classifier=True,
        processed_dir=processed_dir,
        results_dir=results_dir,
        classifier_model_path=results_dir / "reservoir_classifier.joblib",
        classifier_metrics_path=results_dir / "classifier_metrics.yaml",
        classifier_confusion_matrix_path=results_dir / "classifier_confusion_matrix.png",
        classifier_predictions_path=results_dir / "classifier_validation_predictions.csv",
        enable_stp=condition["enable_stp"],
        enable_intrinsic_plasticity=condition["enable_intrinsic_plasticity"],
    )

    if args.no_limit:
        cfg.max_samples = None
        cfg.max_samples_per_emotion = None
    else:
        cfg.max_samples = args.max_samples
        cfg.max_samples_per_emotion = args.max_per_emotion

    return cfg


def summarize_condition(condition: dict, summary: dict) -> dict:
    classifier = summary.get("classifier", {})
    outputs = summary.get("outputs", {})
    return {
        "condition": condition["name"],
        "description": condition["description"],
        "enable_stp": condition["enable_stp"],
        "enable_intrinsic_plasticity": condition["enable_intrinsic_plasticity"],
        "num_samples": int(summary["num_samples"]),
        "mean_global_firing_rate": float(summary["mean_global_firing_rate"]),
        "std_global_firing_rate": float(summary["std_global_firing_rate"]),
        "validation_accuracy": float(classifier.get("validation_accuracy", 0.0)),
        "validation_balanced_accuracy": float(
            classifier.get("validation_balanced_accuracy", 0.0)
        ),
        "validation_macro_f1": float(classifier.get("validation_macro_f1", 0.0)),
        "state_file": outputs.get("state_file"),
        "summary_yaml": outputs.get("summary_yaml"),
        "classifier_metrics": classifier.get("outputs", {}).get("classifier_metrics"),
        "confusion_matrix_plot": classifier.get("outputs", {}).get("confusion_matrix_plot"),
    }


def save_comparison_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "description",
        "enable_stp",
        "enable_intrinsic_plasticity",
        "num_samples",
        "mean_global_firing_rate",
        "std_global_firing_rate",
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_f1",
        "state_file",
        "summary_yaml",
        "classifier_metrics",
        "confusion_matrix_plot",
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
    header = [
        "# STP / IP Ablation Report",
        "",
        "| Condition | STP | IP | Accuracy | Balanced Acc | Macro F1 | Mean Rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    body = [
        (
            f"| {row['condition']} | "
            f"{'on' if row['enable_stp'] else 'off'} | "
            f"{'on' if row['enable_intrinsic_plasticity'] else 'off'} | "
            f"{row['validation_accuracy']:.4f} | "
            f"{row['validation_balanced_accuracy']:.4f} | "
            f"{row['validation_macro_f1']:.4f} | "
            f"{row['mean_global_firing_rate']:.4f} |"
        )
        for row in rows
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join([*header, *body, ""]))


def save_comparison_plot(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [row["condition"] for row in rows]
    metrics = {
        "Accuracy": [row["validation_accuracy"] for row in rows],
        "Balanced Acc": [row["validation_balanced_accuracy"] for row in rows],
        "Macro F1": [row["validation_macro_f1"] for row in rows],
    }

    x = np.arange(len(names), dtype=np.float32)
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for idx, (label, values) in enumerate(metrics.items()):
        ax.bar(x + (idx - 1) * width, values, width=width, label=label)

    ax.set_title("STP / IP Ablation Metrics", fontsize=13, weight="bold")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def print_table(rows: list[dict]) -> None:
    print("Condition Comparison")
    print(
        "condition         accuracy   balanced_acc   macro_f1   mean_rate"
    )
    print(
        "---------------   --------   ------------   --------   ---------"
    )
    for row in rows:
        print(
            f"{row['condition']:<15}   "
            f"{row['validation_accuracy']:.4f}     "
            f"{row['validation_balanced_accuracy']:.4f}        "
            f"{row['validation_macro_f1']:.4f}     "
            f"{row['mean_global_firing_rate']:.4f}"
        )


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict] = []
    run_payload = {
        "experiment_root": str(EXPERIMENT_ROOT),
        "workspace": str(ROOT),
        "seed": args.seed,
        "frame_repeat": args.frame_repeat,
        "input_gain": args.input_gain,
        "use_full_dataset": bool(args.no_limit),
        "conditions": [],
    }

    for condition in CONDITIONS:
        print(
            f"[Run] {condition['name']} "
            f"(STP={'on' if condition['enable_stp'] else 'off'}, "
            f"IP={'on' if condition['enable_intrinsic_plasticity'] else 'off'})"
        )
        cfg = build_config(condition, args)
        summary = run_pipeline(cfg)
        row = summarize_condition(condition, summary)
        comparison_rows.append(row)
        run_payload["conditions"].append(
            {
                "condition": condition["name"],
                "description": condition["description"],
                "summary": summary,
            }
        )

    comparison_rows.sort(key=lambda item: item["condition"])
    comparison_dir = OUTPUT_ROOT
    csv_path = comparison_dir / "comparison_metrics.csv"
    yaml_path = comparison_dir / "comparison_summary.yaml"
    report_path = comparison_dir / "comparison_report.md"
    plot_path = comparison_dir / "comparison_metrics.png"

    save_comparison_csv(comparison_rows, csv_path)
    save_comparison_yaml(
        {
            "settings": {
                "seed": args.seed,
                "frame_repeat": args.frame_repeat,
                "input_gain": args.input_gain,
                "use_full_dataset": bool(args.no_limit),
                "max_samples": None if args.no_limit else args.max_samples,
                "max_samples_per_emotion": None if args.no_limit else args.max_per_emotion,
            },
            "results": comparison_rows,
            "runs": run_payload["conditions"],
        },
        yaml_path,
    )
    save_comparison_report(comparison_rows, report_path)
    save_comparison_plot(comparison_rows, plot_path)

    print()
    print_table(comparison_rows)
    print()
    print(json.dumps({"comparison_csv": str(csv_path), "comparison_yaml": str(yaml_path)}, indent=2))


if __name__ == "__main__":
    main()
