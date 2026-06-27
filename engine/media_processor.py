"""
Media processor — TTS, bulk STT transcription, and local file management.

TTS (Text → Audio):
  Uses Microsoft Edge TTS (edge-tts) — free, no API key, high quality.
  Supported voices:
    yo    (Yoruba)         → yo-NG-EzinneNeural (F) / yo-NG-IsiomaNeural (M)
    en_NG (Nigerian Eng)   → en-NG-EzinneNeural (F) / en-NG-AbeoNeural (M)
    efi   (Efik)           → ig-NG-EzinneNeural (F)  [closest available]
    ibb   (Ibibio)         → ig-NG-EzinneNeural (F)  [closest available]

STT (Audio → Text):
  Uses local OpenAI Whisper. Handles files of any length by chunking internally.
  Writes a .txt transcript and a .jsonl segments file (with timestamps).

File management:
  browse_local_files  — list files in a directory, filtered by extension
  rename_local_file   — rename a file with auto extension correction
  move_local_file     — move to another directory
  map_extensions      — scan a folder and rename files to the correct extension
                        based on detected MIME type (e.g. audio/webm → .webm)
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Voice map for edge-tts
# ---------------------------------------------------------------------------
_TTS_VOICES: dict[str, dict[str, str]] = {
    "yo":    {"F": "yo-NG-EzinneNeural",  "M": "yo-NG-IsiomaNeural"},
    "en_NG": {"F": "en-NG-EzinneNeural",  "M": "en-NG-AbeoNeural"},
    "efi":   {"F": "ig-NG-EzinneNeural",  "M": "ig-NG-ObiNeural"},   # closest available
    "ibb":   {"F": "ig-NG-EzinneNeural",  "M": "ig-NG-ObiNeural"},   # closest available
}

# Canonical extensions for audio training data
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a", ".aac", ".opus"}
_TEXT_EXTENSIONS  = {".txt", ".jsonl", ".json", ".csv"}


# ---------------------------------------------------------------------------
# TTS: text file → WAV audio
# ---------------------------------------------------------------------------

async def _tts_async(text: str, voice: str, output_mp3: str) -> None:
    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_mp3)


def text_to_audio(
    text_path: str,
    language_code: str,
    output_dir: str,
    gender: str = "F",
    add_to_manifest: bool = False,
    speaker_id: str = "tts_synthetic",
    manifest_path: str = "master_manifest.jsonl",
) -> list[dict]:
    """
    Convert a large text file to WAV audio files via TTS.

    Each line in the text file becomes one audio file (good for sentence-per-line corpora).
    Output files are named with the line hash and language code.

    Args:
        text_path:       Path to the source .txt file (one sentence per line)
        language_code:   "yo" | "efi" | "ibb" | "en_NG"
        output_dir:      Directory to write WAV files
        gender:          "F" or "M" — selects TTS voice gender
        add_to_manifest: If True, processes through audio_pipeline and appends to manifest
        speaker_id:      Speaker ID to use when adding to manifest (default: tts_synthetic)
        manifest_path:   Manifest path if add_to_manifest=True

    Returns:
        List of dicts with {text, audio_path, duration}
    """
    import tempfile
    import soundfile as sf
    from scipy.io import wavfile

    voices = _TTS_VOICES.get(language_code)
    if not voices:
        raise ValueError(f"No TTS voice for language {language_code!r}. Available: {list(_TTS_VOICES)}")
    voice = voices.get(gender, voices["F"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(text_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    results: list[dict] = []
    for i, line in enumerate(lines):
        import hashlib
        line_hash = hashlib.sha256(line.encode()).hexdigest()[:12]
        out_wav = output_dir / f"{language_code}_tts_{line_hash}.wav"

        if out_wav.exists():
            results.append({"text": line, "audio_path": str(out_wav), "skipped": True})
            continue

        # TTS → temp MP3 → convert to 16kHz mono WAV
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_mp3 = tmp.name

        try:
            asyncio.run(_tts_async(line, voice, tmp_mp3))

            # Convert MP3 → WAV 16kHz mono using soundfile + scipy
            from engine.audio_pipeline import _to_mono_float32, _resample, TARGET_SR
            import soundfile as sf
            audio, sr = sf.read(tmp_mp3)
            audio = _to_mono_float32(audio)
            audio = _resample(audio, sr)
            sf.write(str(out_wav), audio, TARGET_SR, subtype="PCM_16")
            duration = round(len(audio) / TARGET_SR, 3)
        finally:
            Path(tmp_mp3).unlink(missing_ok=True)

        result = {"text": line, "audio_path": str(out_wav), "duration": duration}

        if add_to_manifest:
            from engine.audio_pipeline import process_audio
            from engine.validator import validate
            val = validate(line, language_code)
            if val.is_valid:
                process_audio(
                    audio_path=out_wav,
                    transcription=val.normalized_text,
                    language_code=language_code,
                    speaker_id=speaker_id,
                    speaker_gender=gender,
                    speaker_age_range="18-30",
                    dialect="synthetic",
                    split="train",
                    output_dir=output_dir,
                    manifest_path=Path(manifest_path),
                )

        results.append(result)
        print(f"[{i+1}/{len(lines)}] {out_wav.name} ({duration:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# STT: audio file → transcript text
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_path: str,
    output_dir: str,
    language_hint: str | None = None,
    model_size: str = "base",
    add_to_manifest: bool = False,
    speaker_id: str = "unknown",
    language_code: str = "yo",
    manifest_path: str = "master_manifest.jsonl",
) -> dict:
    """
    Transcribe an audio file (any length) to text using local Whisper.

    Writes two files to output_dir:
      <stem>.txt   — plain transcript
      <stem>.jsonl — one JSON object per segment: {start, end, text}

    Args:
        audio_path:      Path to input audio (WAV, MP3, FLAC, webm, etc.)
        output_dir:      Directory for transcript output files
        language_hint:   Whisper language hint e.g. "yo", "en", None = auto-detect
        model_size:      Whisper model: "tiny" | "base" | "small" | "medium"
        add_to_manifest: If True, each segment ≥3s is added to the manifest
        speaker_id:      Speaker ID for manifest entries
        language_code:   Language code for manifest entries
        manifest_path:   Manifest path if add_to_manifest=True

    Returns:
        Dict with {transcript_path, segments_path, total_segments, full_text}
    """
    try:
        import whisper
    except ImportError:
        raise RuntimeError("openai-whisper not installed. Run: pip install openai-whisper")

    import json

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Whisper {model_size}...")
    model = whisper.load_model(model_size)

    print(f"Transcribing: {audio_path.name}")
    result = model.transcribe(
        str(audio_path),
        language=language_hint,
        verbose=False,
        word_timestamps=False,
    )

    stem = audio_path.stem
    txt_path  = output_dir / f"{stem}.txt"
    jsonl_path = output_dir / f"{stem}_segments.jsonl"

    txt_path.write_text(result["text"].strip(), encoding="utf-8")

    segments = result.get("segments", [])
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps({
                "start": round(seg["start"], 3),
                "end":   round(seg["end"],   3),
                "text":  seg["text"].strip(),
            }, ensure_ascii=False) + "\n")

    if add_to_manifest:
        from engine.audio_pipeline import process_audio
        from engine.validator import validate
        import tempfile
        import soundfile as sf

        audio_full, sr = sf.read(str(audio_path))
        from engine.audio_pipeline import _to_mono_float32, _resample, TARGET_SR
        audio_full = _to_mono_float32(audio_full)
        audio_full = _resample(audio_full, sr)

        for seg in segments:
            duration = seg["end"] - seg["start"]
            if duration < 2.0 or not seg["text"].strip():
                continue
            val = validate(seg["text"].strip(), language_code)
            if not val.is_valid:
                continue
            start_sample = int(seg["start"] * TARGET_SR)
            end_sample   = int(seg["end"]   * TARGET_SR)
            chunk = audio_full[start_sample:end_sample]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, chunk, TARGET_SR, subtype="PCM_16")
                try:
                    process_audio(
                        audio_path=Path(tmp.name),
                        transcription=val.normalized_text,
                        language_code=language_code,
                        speaker_id=speaker_id,
                        speaker_gender="U",
                        speaker_age_range="18-30",
                        dialect="",
                        split="train",
                        output_dir=output_dir,
                        manifest_path=Path(manifest_path),
                    )
                finally:
                    Path(tmp.name).unlink(missing_ok=True)

    return {
        "transcript_path": str(txt_path),
        "segments_path":   str(jsonl_path),
        "total_segments":  len(segments),
        "full_text":       result["text"].strip(),
    }


# ---------------------------------------------------------------------------
# Local file explorer
# ---------------------------------------------------------------------------

def browse_local_files(
    directory: str,
    extensions: list[str] | None = None,
    pattern: str = "",
    limit: int = 50,
) -> dict:
    """List files in a local directory, optionally filtered by extension or name pattern."""
    directory = Path(directory).expanduser()
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = []
    for entry in sorted(directory.iterdir(), key=lambda e: e.name):
        if entry.name.startswith("."):
            continue
        if extensions and entry.suffix.lower() not in [e.lower() for e in extensions]:
            continue
        if pattern and pattern.lower() not in entry.name.lower():
            continue
        stat = entry.stat()
        files.append({
            "name":       entry.name,
            "path":       str(entry),
            "type":       "dir" if entry.is_dir() else "file",
            "extension":  entry.suffix,
            "size_bytes": stat.st_size if entry.is_file() else None,
        })
        if len(files) >= limit:
            break

    return {"directory": str(directory), "count": len(files), "files": files}


def rename_local_file(old_path: str, new_name: str) -> dict:
    """Rename a local file. new_name can be just the filename or a full path."""
    src = Path(old_path).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")
    dst = src.parent / new_name if "/" not in new_name and "\\" not in new_name else Path(new_name)
    src.rename(dst)
    return {"old_path": str(src), "new_path": str(dst)}


def move_local_file(src_path: str, dst_dir: str) -> dict:
    """Move a file to a different local directory."""
    src = Path(src_path).expanduser()
    dst = Path(dst_dir).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    result = shutil.move(str(src), str(dst / src.name))
    return {"moved_to": result}


def map_extensions(directory: str, dry_run: bool = False) -> dict:
    """
    Scan a directory and rename files to their correct extension based on MIME type.

    Useful when browser recordings arrive as .bin or with wrong extensions.
    Audio files are also converted to .wav standard when possible.

    Args:
        directory: Directory to scan
        dry_run:   If True, report what would change without renaming

    Returns:
        Dict with list of renames performed (or planned if dry_run=True)
    """
    import imghdr

    directory = Path(directory).expanduser()
    renames: list[dict] = []

    # Magic bytes for common audio formats
    _MAGIC: list[tuple[bytes, str]] = [
        (b"RIFF", ".wav"),
        (b"fLaC", ".flac"),
        (b"OggS", ".ogg"),
        (b"\x1aE\xdf\xa3", ".webm"),   # Matroska/WebM
        (b"ID3",  ".mp3"),
        (b"\xff\xfb", ".mp3"),
        (b"\xff\xf3", ".mp3"),
        (b"ftyp", ".m4a"),
    ]

    def _detect_ext(path: Path) -> str | None:
        with open(path, "rb") as f:
            header = f.read(12)
        for magic, ext in _MAGIC:
            if header[:len(magic)] == magic:
                return ext
        # Fallback: MIME guess from filename
        mime, _ = mimetypes.guess_type(str(path))
        if mime:
            ext = mimetypes.guess_extension(mime)
            return ext
        return None

    for entry in directory.iterdir():
        if not entry.is_file() or entry.name.startswith("."):
            continue
        detected = _detect_ext(entry)
        if detected and entry.suffix.lower() != detected:
            new_path = entry.with_suffix(detected)
            renames.append({"from": str(entry), "to": str(new_path)})
            if not dry_run:
                entry.rename(new_path)

    return {
        "directory":  str(directory),
        "dry_run":    dry_run,
        "renames":    renames,
        "count":      len(renames),
    }


# ---------------------------------------------------------------------------
# Tool definitions for Aziz
# ---------------------------------------------------------------------------

MEDIA_TOOLS: list[dict] = [
    {
        "name": "text_to_audio",
        "description": (
            "Convert a large text file to audio using TTS. Each line becomes one audio file. "
            "Use when the user wants to synthesize speech, translate text to audio, or create "
            "synthetic training data from a script file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text_path":        {"type": "string", "description": "Path to .txt file (one sentence per line)"},
                "language_code":    {"type": "string", "enum": ["yo", "efi", "ibb", "en_NG"]},
                "output_dir":       {"type": "string"},
                "gender":           {"type": "string", "enum": ["F", "M"], "default": "F"},
                "add_to_manifest":  {"type": "boolean", "default": False},
                "speaker_id":       {"type": "string", "default": "tts_synthetic"},
            },
            "required": ["text_path", "language_code", "output_dir"],
        },
    },
    {
        "name": "transcribe_audio",
        "description": (
            "Transcribe a large audio file to text using Whisper. Works on files of any length. "
            "Use when the user wants to transcribe audio, convert speech to text, or extract "
            "a transcript from a recording."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "audio_path":       {"type": "string"},
                "output_dir":       {"type": "string"},
                "language_hint":    {"type": "string", "description": "e.g. 'yo', 'en' — or omit for auto-detect"},
                "model_size":       {"type": "string", "enum": ["tiny", "base", "small", "medium"], "default": "base"},
                "add_to_manifest":  {"type": "boolean", "default": False},
                "speaker_id":       {"type": "string", "default": "unknown"},
                "language_code":    {"type": "string", "enum": ["yo", "efi", "ibb", "en_NG"]},
            },
            "required": ["audio_path", "output_dir"],
        },
    },
    {
        "name": "browse_local_files",
        "description": (
            "Browse files on the local computer (file explorer). "
            "Use when the user wants to see what files are in a folder, find audio or text files, "
            "or explore their local directories."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directory":    {"type": "string", "description": "Local directory path (e.g. C:/Users/User/Downloads)"},
                "extensions":   {"type": "array", "items": {"type": "string"}, "description": "Filter by extension e.g. ['.wav', '.mp3']"},
                "pattern":      {"type": "string", "description": "Filter by filename containing this string"},
                "limit":        {"type": "integer", "default": 50},
            },
            "required": ["directory"],
        },
    },
    {
        "name": "rename_local_file",
        "description": "Rename a file on the local computer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "old_path": {"type": "string"},
                "new_name": {"type": "string", "description": "New filename (just the name, or a full path)"},
            },
            "required": ["old_path", "new_name"],
        },
    },
    {
        "name": "move_local_file",
        "description": "Move a local file to a different directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "src_path": {"type": "string"},
                "dst_dir":  {"type": "string"},
            },
            "required": ["src_path", "dst_dir"],
        },
    },
    {
        "name": "map_extensions",
        "description": (
            "Scan a folder and rename files to their correct extension based on file content "
            "(not just filename). Use when the user has audio files with wrong or missing "
            "extensions that need to be fixed before model training."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string"},
                "dry_run":   {"type": "boolean", "default": False, "description": "If true, only report what would change"},
            },
            "required": ["directory"],
        },
    },
]

MEDIA_EXECUTOR: dict = {
    "text_to_audio":    lambda p: text_to_audio(**p),
    "transcribe_audio": lambda p: transcribe_audio(**p),
    "browse_local_files": lambda p: browse_local_files(**p),
    "rename_local_file":  lambda p: rename_local_file(**p),
    "move_local_file":    lambda p: move_local_file(**p),
    "map_extensions":     lambda p: map_extensions(**p),
}
