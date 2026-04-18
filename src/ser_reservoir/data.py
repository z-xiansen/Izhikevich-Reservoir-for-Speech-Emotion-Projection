from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Callable, Iterable

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download
from scipy.signal import resample_poly
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

RAW_WAVEFORM_TARGET_LENGTH = 10320
RAW_WAVEFORM_INPUT_CHANNELS = 40


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


def extract_raw_waveform_feature(
    path: Path,
    cfg: ExperimentConfig,
    target_length: int = RAW_WAVEFORM_TARGET_LENGTH,
) -> np.ndarray:
    waveform, _ = _load_audio_mono_resampled(path, target_sr=cfg.sample_rate)
    normalized = _peak_normalize_waveform(waveform)
    resampled = _resample_1d_linear(normalized, target_length=target_length)
    shifted = np.clip((resampled + 1.0) * 0.5, 0.0, 1.0)
    return shifted.astype(np.float32, copy=False)


def raw_waveform_feature_to_reservoir_input(
    feature: np.ndarray,
    input_channels: int = RAW_WAVEFORM_INPUT_CHANNELS,
) -> np.ndarray:
    flat = np.asarray(feature, dtype=np.float32).reshape(-1)
    metadata = raw_waveform_feature_metadata(
        target_length=int(flat.size),
        input_channels=input_channels,
    )
    target_frames = metadata["target_frames"]
    return flat.reshape(target_frames, input_channels).T.astype(np.float32, copy=False)


def raw_waveform_feature_metadata(
    target_length: int = RAW_WAVEFORM_TARGET_LENGTH,
    input_channels: int = RAW_WAVEFORM_INPUT_CHANNELS,
    sample_rate: int | None = None,
) -> dict[str, int]:
    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")
    if input_channels <= 0:
        raise ValueError(f"input_channels must be positive, got {input_channels}")
    if target_length % input_channels != 0:
        raise ValueError(
            f"target_length={target_length} must be divisible by input_channels={input_channels}"
        )

    metadata = {
        "target_length": int(target_length),
        "input_channels": int(input_channels),
        "target_frames": int(target_length // input_channels),
    }
    if sample_rate is not None:
        metadata["sample_rate"] = int(sample_rate)
    return metadata


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


def _peak_normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    arr = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Waveform is empty.")

    peak = float(np.max(np.abs(arr)))
    if peak <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / peak, -1.0, 1.0).astype(np.float32, copy=False)


def _resample_1d_linear(signal: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")

    arr = np.asarray(signal, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("Signal is empty.")
    if arr.size == target_length:
        return arr.astype(np.float32, copy=True)
    if arr.size == 1:
        return np.full(target_length, float(arr[0]), dtype=np.float32)

    xp = np.arange(arr.size, dtype=np.float32)
    x_new = np.linspace(0.0, float(arr.size - 1), num=target_length, dtype=np.float32)
    return np.interp(x_new, xp, arr).astype(np.float32)


def _load_audio_mono_resampled(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    waveform, src_sr = sf.read(str(path), dtype="float32")
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)

    if src_sr == target_sr:
        return arr.astype(np.float32, copy=False), target_sr

    factor = gcd(int(src_sr), int(target_sr))
    resampled = resample_poly(
        arr,
        up=int(target_sr // factor),
        down=int(src_sr // factor),
    )
    return np.asarray(resampled, dtype=np.float32), target_sr
