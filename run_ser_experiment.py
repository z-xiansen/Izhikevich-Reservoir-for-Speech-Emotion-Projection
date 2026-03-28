from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ser_reservoir import ExperimentConfig, run_pipeline
from ser_reservoir.classifier import format_classifier_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Izhikevich SNN reservoir for TESS speech emotion state projection."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples", type=int, default=420)
    parser.add_argument("--max-per-emotion", type=int, default=70)
    parser.add_argument("--no-limit", action="store_true", help="Use all available samples.")
    parser.add_argument("--frame-repeat", type=int, default=2)
    parser.add_argument("--input-gain", type=float, default=14.0)
    parser.add_argument(
        "--save-model",
        "--save-checkpoint",
        dest="save_model",
        action="store_true",
        help="Save the current reservoir model after this run.",
    )
    parser.add_argument(
        "--model-path",
        "--checkpoint-path",
        dest="model_path",
        type=str,
        default="results/reservoir_model.npz",
        help="Path used when saving the model and as the default output path.",
    )
    parser.add_argument(
        "--load-model",
        "--load-checkpoint",
        dest="load_model",
        type=str,
        default=None,
        help="Load an existing reservoir model before processing samples.",
    )
    parser.add_argument(
        "--no-classifier",
        action="store_true",
        help="Skip classifier training on the 800-D reservoir features.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        workspace=ROOT,
        seed=args.seed,
        frame_repeat=args.frame_repeat,
        input_gain=args.input_gain,
        verbose=not args.quiet,
        train_classifier=not args.no_classifier,
        save_model=args.save_model,
        model_path=Path(args.model_path),
        load_model_path=Path(args.load_model) if args.load_model else None,
    )

    if args.no_limit:
        cfg.max_samples = None
        cfg.max_samples_per_emotion = None
    else:
        cfg.max_samples = args.max_samples
        cfg.max_samples_per_emotion = args.max_per_emotion

    summary = run_pipeline(cfg)
    if summary.get("classifier", {}).get("enabled"):
        print(format_classifier_report(summary["classifier"]))
        print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
