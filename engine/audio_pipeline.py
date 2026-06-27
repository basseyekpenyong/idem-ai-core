"""
Audio processing pipeline.

Responsibilities:
- Resample any input audio to 16 kHz mono WAV (the standard for ASR training).
- Estimate signal quality: SNR and clipping.
- Hash input audio bytes (SHA-256[:16]) to produce a stable, content-addressed ID.
- Write a ManifestEntry to master_manifest.jsonl.

scipy is used for resampling (lighter than librosa for this task). soundfile for I/O.
"""

import hashlib
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from schemas.manifest_entry import ManifestEntry

TARGET_SR = 16_000
CLIPPING_THRESHOLD = 0.99   # peak amplitude considered clipped
SNR_FRAME_MS = 10           # frame size for SNR estimation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def _resample(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SR:
        return audio
    g = gcd(TARGET_SR, orig_sr)
    return resample_poly(audio, TARGET_SR // g, orig_sr // g).astype(np.float32)


def _content_hash(path: Path) -> str:
    """SHA-256 of raw file bytes, truncated to 16 hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _check_clipping(audio: np.ndarray) -> bool:
    return float(np.max(np.abs(audio))) >= CLIPPING_THRESHOLD


def _estimate_snr_db(audio: np.ndarray, sr: int) -> float:
    """
    Frame-energy SNR estimate.

    Signal energy  = mean frame energy across all frames.
    Noise floor    = 10th-percentile frame energy (quietest 10% of frames).

    Returns 60.0 dB when the noise floor is effectively zero (very clean signal).
    Returns 0.0 dB for very short clips that cannot be framed.
    """
    frame_len = max(1, sr * SNR_FRAME_MS // 1000)
    n_complete_frames = len(audio) // frame_len
    if n_complete_frames < 2:
        return 0.0

    frames = audio[: n_complete_frames * frame_len].reshape(n_complete_frames, frame_len)
    energies = np.mean(frames ** 2, axis=1)

    noise_floor = float(np.percentile(energies, 10))
    signal_energy = float(np.mean(energies))

    if noise_floor <= 1e-12:
        return 60.0
    return float(10 * np.log10(signal_energy / noise_floor))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_audio(
    *,
    audio_path: Path,
    transcription: str,
    language_code: str,
    speaker_id: str,
    speaker_gender: str,
    speaker_age_range: str,
    dialect: str,
    split: str,
    output_dir: Path,
    manifest_path: Path,
) -> ManifestEntry:
    """
    Process one audio file and append a ManifestEntry to manifest_path.

    Args:
        audio_path:      Source audio (any format soundfile can read).
        transcription:   Already-validated NFC-normalized text for this recording.
        language_code:   ISO 639-3 code ("yo" | "efi" | "ibb").
        speaker_id:      Stable opaque speaker identifier.
        speaker_gender:  "M" | "F" | "U".
        speaker_age_range: "18-30" | "31-45" | "46-60" | "60+".
        dialect:         Free-text dialect label, "" if unknown.
        split:           "train" | "dev" | "test".
        output_dir:      Directory to write the processed 16 kHz WAV.
        manifest_path:   Path to master_manifest.jsonl (appended, not overwritten).

    Returns:
        The ManifestEntry written to the manifest.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_audio, orig_sr = sf.read(str(audio_path), always_2d=False)
    audio = _to_mono_float32(raw_audio)
    audio = _resample(audio, orig_sr)

    hash_id = _content_hash(audio_path)
    is_clipping = _check_clipping(audio)
    snr_db = _estimate_snr_db(audio, TARGET_SR)
    duration = round(len(audio) / TARGET_SR, 3)

    out_path = output_dir / f"{language_code}_{hash_id}.wav"
    sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")

    entry = ManifestEntry(
        audio_filepath=str(out_path),
        text=transcription,
        duration=duration,
        language=language_code,
        speaker_id=speaker_id,
        speaker_gender=speaker_gender,
        speaker_age_range=speaker_age_range,
        dialect=dialect,
        sample_rate=TARGET_SR,
        hash_id=hash_id,
        split=split,
        quality_snr_db=round(snr_db, 1),
        quality_clipping=is_clipping,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(entry.to_jsonl() + "\n")

    return entry


def read_manifest(manifest_path: Path) -> list[ManifestEntry]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return []
    entries = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(ManifestEntry.from_jsonl(line))
    return entries


def export_dataset(
    manifest_path: Path,
    output_path: Path,
    language_code: str | None = None,
    min_snr_db: float = 15.0,
    include_clipped: bool = False,
    splits: list[str] | None = None,
) -> dict:
    """
    Export a clean dataset.json from the manifest, ready for model training.

    Filters:
      - Optionally restricts to one language_code (None = all languages)
      - Removes clipped recordings (unless include_clipped=True)
      - Removes low-SNR recordings (below min_snr_db)
      - Optionally restricts to specific splits

    Output format (HuggingFace-compatible):
    {
      "meta": { "language": "yo", "min_snr_db": 15.0, ... },
      "train": [{ "audio_filepath", "text", "duration", "speaker_id", ... }],
      "dev":   [...],
      "test":  [...]
    }

    Args:
        manifest_path: Path to master_manifest.jsonl
        output_path:   Where to write dataset.json
        language_code: Filter to one language, or None for all
        min_snr_db:    Minimum SNR to include (default 15.0 dB)
        include_clipped: If False (default), drop clipped recordings
        splits:        List of splits to include, e.g. ["train", "dev"] (default: all)

    Returns:
        Summary dict with entry counts per split.
    """
    import json

    entries = read_manifest(manifest_path)
    splits = splits or ["train", "dev", "test"]

    filtered = [
        e for e in entries
        if (language_code is None or e.language == language_code)
        and e.split in splits
        and e.quality_snr_db >= min_snr_db
        and (include_clipped or not e.quality_clipping)
    ]

    buckets: dict[str, list[dict]] = {s: [] for s in splits}
    for e in filtered:
        record = {
            "audio_filepath": e.audio_filepath,
            "text":           e.text,
            "duration":       e.duration,
            "language":       e.language,
            "speaker_id":     e.speaker_id,
            "speaker_gender": e.speaker_gender,
            "dialect":        e.dialect,
            "hash_id":        e.hash_id,
        }
        buckets[e.split].append(record)

    dataset = {
        "meta": {
            "language":       language_code or "all",
            "min_snr_db":     min_snr_db,
            "include_clipped":include_clipped,
            "total_entries":  len(filtered),
            "splits":         splits,
        },
        **buckets,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    summary = {
        "output_path":   str(output_path),
        "total_clean":   len(filtered),
        "by_split":      {s: len(buckets[s]) for s in splits},
        "total_hours":   round(sum(e.duration for e in filtered) / 3600, 3),
    }
    return summary


def manifest_stats(manifest_path: Path) -> dict:
    """Return summary statistics for the manifest (used by the dashboard)."""
    entries = read_manifest(manifest_path)
    clean = [e for e in entries if e.is_clean()]

    by_lang: dict[str, float] = {}
    for e in clean:
        by_lang[e.language] = round(by_lang.get(e.language, 0.0) + e.duration, 3)

    return {
        "total_entries": len(entries),
        "clean_entries": len(clean),
        "total_hours": round(sum(e.duration for e in entries) / 3600, 3),
        "clean_hours": round(sum(e.duration for e in clean) / 3600, 3),
        "by_language_hours": {k: round(v / 3600, 3) for k, v in by_lang.items()},
        "clipped": sum(1 for e in entries if e.quality_clipping),
        "low_snr": sum(1 for e in entries if e.quality_snr_db < 15.0),
    }
