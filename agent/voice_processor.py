"""
Voice processor — mic capture + Speech-to-Text transcription.

Captures audio from the default microphone, detects silence to auto-stop,
then transcribes using OpenAI Whisper (local model, runs offline).

The transcription is passed directly to AzizOrchestrator.run() so the user
can issue agent commands by speaking English into the mic.

Language note: Whisper is used for AGENT COMMANDS only (expected in English).
For recording African-language training data, see app/recording_server.py
which uses the browser MediaRecorder API — a much better fit for that workflow.
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
    _HAS_AUDIO = True
except ImportError:
    _HAS_AUDIO = False

try:
    import whisper as openai_whisper
    _HAS_WHISPER = True
except ImportError:
    _HAS_WHISPER = False


SAMPLE_RATE = 16_000
SILENCE_THRESHOLD = 0.01    # RMS below this is considered silence
SILENCE_DURATION = 1.5      # seconds of silence before auto-stop
MAX_RECORD_SECONDS = 30     # hard stop
WHISPER_MODEL = "base"      # "tiny" is faster; "small" is more accurate


class VoiceProcessor:
    """
    Stateful voice processor. Call .start_listening() to begin, .stop() to end.
    Supports a callback for when a transcription is ready.

    Usage:
        def on_transcription(text: str):
            response = aziz.run(text)
            print(response.result)

        vp = VoiceProcessor(on_transcription=on_transcription)
        vp.listen_once()   # blocking — records until silence then transcribes
    """

    def __init__(
        self,
        on_transcription: Callable[[str], None] | None = None,
        model_name: str = WHISPER_MODEL,
    ):
        if not _HAS_AUDIO:
            raise RuntimeError(
                "sounddevice and soundfile are required for voice capture. "
                "Install with: pip install sounddevice soundfile"
            )
        if not _HAS_WHISPER:
            raise RuntimeError(
                "openai-whisper is required for transcription. "
                "Install with: pip install openai-whisper"
            )

        self._on_transcription = on_transcription
        self._model: openai_whisper.Whisper | None = None
        self._model_name = model_name
        self._recording = False
        self._frames: list[np.ndarray] = []

    def _load_model(self) -> None:
        if self._model is None:
            self._model = openai_whisper.load_model(self._model_name)

    def _is_silent(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame ** 2)))
        return rms < SILENCE_THRESHOLD

    def listen_once(self) -> str:
        """
        Record one utterance (blocking).
        Auto-stops after SILENCE_DURATION seconds of silence or MAX_RECORD_SECONDS.
        Returns the transcribed text.
        """
        self._load_model()
        self._frames = []
        silent_chunks = 0
        chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks
        silence_chunks_needed = int(SILENCE_DURATION / 0.1)
        max_chunks = int(MAX_RECORD_SECONDS / 0.1)

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            for _ in range(max_chunks):
                frame, _ = stream.read(chunk_size)
                frame = frame.flatten()
                self._frames.append(frame)

                if self._is_silent(frame):
                    silent_chunks += 1
                    if silent_chunks >= silence_chunks_needed and len(self._frames) > silence_chunks_needed:
                        break
                else:
                    silent_chunks = 0

        audio = np.concatenate(self._frames)
        transcription = self._transcribe(audio)

        if self._on_transcription and transcription.strip():
            self._on_transcription(transcription.strip())

        return transcription.strip()

    def _transcribe(self, audio: np.ndarray) -> str:
        """Write audio to a temp WAV and run Whisper."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, audio, SAMPLE_RATE, subtype="PCM_16")
            result = self._model.transcribe(tmp_path, language="en", fp16=False)
            return result.get("text", "").strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def listen_loop(self, stop_event: threading.Event | None = None) -> None:
        """
        Continuously listen and transcribe until stop_event is set (or KeyboardInterrupt).
        Designed to run in a background thread from the dashboard.
        """
        if stop_event is None:
            stop_event = threading.Event()

        try:
            while not stop_event.is_set():
                self.listen_once()
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
