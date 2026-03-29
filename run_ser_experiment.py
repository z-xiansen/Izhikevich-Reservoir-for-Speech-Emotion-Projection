from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ser_reservoir import ExperimentConfig, run_pipeline
from ser_reservoir.classifier import format_classifier_report


DEFAULT_MODEL_DIR = Path("results/models")
DEFAULT_MODEL_STEM = "reservoir_model"
DEFAULT_LOAD_SENTINEL = "__DEFAULT_LATEST_MODEL__"


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
        default=None,
        help=(
            "Optional save path. If omitted, models are saved to results/models/ "
            "with an automatic timestamp suffix."
        ),
    )
    parser.add_argument(
        "--load-model",
        "--load-checkpoint",
        dest="load_model",
        type=str,
        nargs="?",
        const=DEFAULT_LOAD_SENTINEL,
        default=None,
        help=(
            "Load an existing reservoir model before processing samples. "
            "If used without a path, the latest model in results/models/ is loaded."
        ),
    )
    parser.add_argument(
        "--no-classifier",
        action="store_true",
        help="Skip classifier training on the 800-D reservoir features.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _latest_model_path(directory: Path) -> Path:
    model_files = sorted(
        directory.glob("*.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not model_files:
        raise FileNotFoundError(f"No model files found in: {directory}")
    return model_files[0]


def _resolve_save_model_path(raw_path: str | None) -> Path:
    timestamp = _timestamp()
    if raw_path is None:
        return DEFAULT_MODEL_DIR / f"{DEFAULT_MODEL_STEM}_{timestamp}.npz"

    candidate = Path(raw_path)
    if candidate.suffix.lower() != ".npz":
        return candidate / f"{DEFAULT_MODEL_STEM}_{timestamp}.npz"

    return candidate.with_name(f"{candidate.stem}_{timestamp}{candidate.suffix}")


def _resolve_load_model_path(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None

    if raw_path == DEFAULT_LOAD_SENTINEL:
        return _latest_model_path(ROOT / DEFAULT_MODEL_DIR)

    candidate = Path(raw_path)
    if candidate.suffix.lower() == ".npz":
        return candidate

    return _latest_model_path(ROOT / candidate)


def main() -> None:
    args = parse_args()
    save_model_path = _resolve_save_model_path(args.model_path) if args.save_model else None
    load_model_path = _resolve_load_model_path(args.load_model)

    cfg = ExperimentConfig(
        workspace=ROOT,
        seed=args.seed,
        frame_repeat=args.frame_repeat,
        input_gain=args.input_gain,
        verbose=not args.quiet,
        train_classifier=not args.no_classifier,
        save_model=args.save_model,
        model_path=save_model_path if save_model_path is not None else DEFAULT_MODEL_DIR,
        load_model_path=load_model_path,
    )

    if args.no_limit:
        cfg.max_samples = None
        cfg.max_samples_per_emotion = None
    else:
        cfg.max_samples = args.max_samples
        cfg.max_samples_per_emotion = args.max_per_emotion

    if cfg.load_model_path is not None:
        print(f"[Model] Loading reservoir model from: {cfg.load_model_path}")
    else:
        print("[Model] Loading reservoir model: disabled, using a new random reservoir.")

    if cfg.save_model:
        print(f"[Model] Saving reservoir model to: {cfg.model_path}")
    else:
        print("[Model] Saving reservoir model: disabled.")

    summary = run_pipeline(cfg)
    if summary.get("classifier", {}).get("enabled"):
        print(format_classifier_report(summary["classifier"]))
        print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
