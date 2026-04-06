from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_experiment import (
    FEATURE_FILE,
    TSNE_EMBEDDING_FILE,
    TSNE_PLOT_PATH,
    build_data_config,
    load_or_build_features,
    save_tsne_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the t-SNE plot for direct MFCC features."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional total sample cap when rebuilding features.",
    )
    parser.add_argument(
        "--max-per-emotion",
        type=int,
        default=None,
        help="Optional per-emotion sample cap when rebuilding features.",
    )
    parser.add_argument(
        "--refresh-features",
        action="store_true",
        help="Re-extract MFCC features before computing t-SNE.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_data_config(args)
    features, labels, _, metadata = load_or_build_features(
        cfg=cfg,
        feature_file=FEATURE_FILE,
        refresh=args.refresh_features,
    )
    outputs = save_tsne_outputs(features, labels, cfg)
    print(
        json.dumps(
            {
                "feature_file": str(FEATURE_FILE),
                "tsne_embedding_file": outputs["tsne_embedding_file"],
                "tsne_plot": outputs["tsne_plot"],
                "feature_metadata": metadata,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
