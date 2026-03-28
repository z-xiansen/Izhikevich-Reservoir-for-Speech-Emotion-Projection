from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import librosa
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from tqdm import tqdm

from .config import ExperimentConfig

_EMOTION_ALIAS = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "ps": "pleasant_surprise",
    "pleasant_surprise": "pleasant_surprise",
    "pleasantsurprise": "pleasant_surprise",
    "surprise": "pleasant_surprise",
}


@dataclass(slots=True, frozen=True)
class AudioSample:
    path: Path
    label: str


def ensure_tess_dataset(cfg: ExperimentConfig) -> Path:
    cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.tess_target_dir.mkdir(parents=True, exist_ok=True)

    if _has_wavs(cfg.tess_target_dir):
        return cfg.tess_target_dir

    providers = _build_dataset_providers()
    errors: list[str] = []
    for name, fn in providers:
        try:
            root = fn(cfg)
            if _has_wavs(root):
                return root
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")

    joined = "\n".join(errors) if errors else "unknown"
    raise RuntimeError(
        "TESS dataset download failed from all providers.\n"
        "If Kaggle is not configured, the code will skip Kaggle providers and "
        "fall back to the HuggingFace parquet mirror.\n"
        "To enable Kaggle download, set KAGGLE_USERNAME/KAGGLE_KEY or place "
        "~/.kaggle/kaggle.json.\n"
        f"Provider errors:\n{joined}"
    )


def collect_audio_samples(dataset_root: Path, cfg: ExperimentConfig) -> list[AudioSample]:
    wav_files = sorted(dataset_root.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No wav files found under {dataset_root}")

    parsed: list[AudioSample] = []
    for wav_path in wav_files:
        label = parse_emotion_label(wav_path)
        if label is not None:
            parsed.append(AudioSample(path=wav_path, label=label))

    if not parsed:
        raise RuntimeError("No valid labeled samples found after parsing emotion labels.")

    rng = np.random.default_rng(cfg.seed)
    by_label: dict[str, list[AudioSample]] = {}
    for sample in parsed:
        by_label.setdefault(sample.label, []).append(sample)

    selected: list[AudioSample] = []
    for label in cfg.label_order:
        items = by_label.get(label, [])
        if not items:
            continue
        rng.shuffle(items)
        if cfg.max_samples_per_emotion is not None:
            items = items[: cfg.max_samples_per_emotion]
        selected.extend(items)

    if cfg.max_samples is not None and len(selected) > cfg.max_samples:
        rng.shuffle(selected)
        selected = selected[: cfg.max_samples]

    selected.sort(key=lambda s: (s.label, str(s.path)))
    return selected


def extract_mfcc(path: Path, cfg: ExperimentConfig) -> np.ndarray:
    waveform, sr = librosa.load(path, sr=cfg.sample_rate, mono=True)
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
    )
    return _normalize_to_unit_interval(mfcc).astype(np.float32)


def parse_emotion_label(path: Path) -> str | None:
    fname_tokens = path.stem.lower().replace("-", "_").split("_")
    folder_tokens = "_".join(path.parts[-3:]).lower().replace("-", "_").split("_")
    tokens = fname_tokens + folder_tokens
    for token in reversed(tokens):
        if token in _EMOTION_ALIAS:
            return _EMOTION_ALIAS[token]
    return None


def _download_from_kagglehub(cfg: ExperimentConfig) -> Path:
    import kagglehub

    dl_path = Path(kagglehub.dataset_download(cfg.dataset_id))
    if not _has_wavs(dl_path):
        raise RuntimeError(f"kagglehub finished but no wav files found in {dl_path}")
    return dl_path


def _download_from_kaggle_api(cfg: ExperimentConfig) -> Path:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        dataset=cfg.dataset_id,
        path=str(cfg.raw_data_dir),
        unzip=True,
        quiet=True,
    )
    if not _has_wavs(cfg.raw_data_dir):
        raise RuntimeError(f"kaggle API download finished but no wav files in {cfg.raw_data_dir}")
    return cfg.raw_data_dir


def _download_from_hf_parquet_mirror(cfg: ExperimentConfig) -> Path:
    cache_dir = Path(
        snapshot_download(
            repo_id=cfg.hf_dataset_id,
            repo_type="dataset",
            allow_patterns=["data/*.parquet"],
        )
    )
    parquet_files = sorted((cache_dir / "data").glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in mirror snapshot {cache_dir}")

    target_root = cfg.tess_target_dir
    target_root.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in tqdm(
            parquet.iter_batches(batch_size=128, columns=["WavPath", "audio"]),
            desc=f"Extracting {parquet_path.name}",
            unit="batch",
            leave=False,
        ):
            frame = batch.to_pydict()
            for wav_rel, audio_obj in zip(frame["WavPath"], frame["audio"]):
                audio_bytes = audio_obj.get("bytes") if isinstance(audio_obj, dict) else None
                if not audio_bytes:
                    continue
                out_path = _target_path_from_wav_rel(target_root, wav_rel)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if not out_path.exists():
                    out_path.write_bytes(bytes(audio_bytes))
                    total_written += 1

    if total_written == 0 and not _has_wavs(target_root):
        raise RuntimeError("HF parquet mirror download succeeded but no wav was extracted.")
    return target_root


def _target_path_from_wav_rel(target_root: Path, wav_rel: str) -> Path:
    rel = Path(wav_rel)
    if len(rel.parts) >= 2:
        rel = Path(*rel.parts[-2:])
    rel = Path(rel.name.replace(" ", "_")) if rel.parent == Path(".") else rel
    return target_root / rel


def _has_wavs(root: Path) -> bool:
    if not root.exists():
        return False
    return any(root.rglob("*.wav"))


def _build_dataset_providers() -> list[tuple[str, Callable[[ExperimentConfig], Path]]]:
    providers: list[tuple[str, Callable[[ExperimentConfig], Path]]] = []
    if _has_kaggle_credentials():
        # Prefer the official Kaggle API when credentials are configured.
        providers.append(("kaggle_api", _download_from_kaggle_api))
        providers.append(("kagglehub", _download_from_kagglehub))
    providers.append(("huggingface_parquet_mirror", _download_from_hf_parquet_mirror))
    return providers


def _has_kaggle_credentials() -> bool:
    import os

    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def summarize_label_counts(samples: Iterable[AudioSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.label] = counts.get(sample.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _normalize_to_unit_interval(mfcc: np.ndarray) -> np.ndarray:
    row_min = mfcc.min(axis=1, keepdims=True)
    row_max = mfcc.max(axis=1, keepdims=True)
    denom = np.maximum(row_max - row_min, 1e-8)
    return (mfcc - row_min) / denom
