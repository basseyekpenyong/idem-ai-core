"""
Aziz Agent orchestrator.

Architecture (per design advice):
- Claude handles ONLY intent parsing and parameter extraction from natural language.
- All actual file I/O and pipeline execution is done deterministically in Python.
- The LLM is never given direct access to the filesystem or manifest.

Flow:
  user text/transcription
       │
       ▼
  Claude (tool-use) ──► identifies intent + extracts params
       │
       ▼
  _ROUTER dispatch ──► Python engine function
       │
       ▼
  result dict returned to caller
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import anthropic

from engine.audio_pipeline import manifest_stats, process_audio
from engine.script_generator import chunk_file, mock_scripts
from engine.validator import validate

MODEL = "claude-haiku-4-5-20251001"  # Fast and cheap for intent parsing


class Intent(str, Enum):
    INGEST = "ingest_audio"
    VALIDATE = "validate_text"
    GENERATE_SCRIPTS = "generate_scripts"
    BUILD_MANIFEST = "build_manifest"
    GET_STATUS = "get_status"


@dataclass
class AgentResponse:
    intent: Intent | None
    params: dict[str, Any]
    result: dict[str, Any]
    raw_reply: str


# ---------------------------------------------------------------------------
# Tool definitions — Claude uses these to extract structured intent
# ---------------------------------------------------------------------------
_TOOLS: list[dict] = [
    {
        "name": Intent.INGEST,
        "description": (
            "Ingest and process a recorded audio file into the training dataset. "
            "Use when the user mentions recording, adding audio, or ingesting a file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string", "description": "Path to the audio file"},
                "transcription": {"type": "string", "description": "The text spoken in the recording"},
                "language_code": {"type": "string", "enum": ["yo", "efi", "ibb"]},
                "speaker_id": {"type": "string", "description": "Unique speaker identifier"},
                "speaker_gender": {"type": "string", "enum": ["M", "F", "U"]},
                "speaker_age_range": {"type": "string", "enum": ["18-30", "31-45", "46-60", "60+"]},
                "dialect": {"type": "string", "default": ""},
                "split": {"type": "string", "enum": ["train", "dev", "test"], "default": "train"},
            },
            "required": ["audio_path", "transcription", "language_code", "speaker_id"],
        },
    },
    {
        "name": Intent.VALIDATE,
        "description": (
            "Validate a text string against the character rules for a target language. "
            "Use when the user asks to check, validate, or verify text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "language_code": {"type": "string", "enum": ["yo", "efi", "ibb"]},
            },
            "required": ["text", "language_code"],
        },
    },
    {
        "name": Intent.GENERATE_SCRIPTS,
        "description": (
            "Generate recordable script chunks from a text file or use mock data. "
            "Use when the user asks to generate scripts, get sentences to record, or chunk a document."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "language_code": {"type": "string", "enum": ["yo", "efi", "ibb"]},
                "file_path": {"type": "string", "description": "Path to source text file (optional)"},
                "count": {"type": "integer", "default": 5, "description": "Number of chunks to return"},
            },
            "required": ["language_code"],
        },
    },
    {
        "name": Intent.GET_STATUS,
        "description": (
            "Get the current dataset status: total hours, entries, quality flags. "
            "Use when the user asks about progress, stats, how many hours, or dataset size."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "manifest_path": {"type": "string", "description": "Path to manifest file (optional)"},
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Execution router — deterministic, no LLM involved
# ---------------------------------------------------------------------------
_DEFAULT_MANIFEST = Path("master_manifest.jsonl")
_DEFAULT_OUTPUT = Path("data/processed")


def _execute(intent: Intent, params: dict, config: dict) -> dict[str, Any]:
    manifest_path = Path(config.get("manifest_path", _DEFAULT_MANIFEST))
    output_dir = Path(config.get("output_dir", _DEFAULT_OUTPUT))

    if intent == Intent.VALIDATE:
        result = validate(params["text"], params["language_code"])
        return {
            "is_valid": result.is_valid,
            "normalized_text": result.normalized_text,
            "errors": result.errors,
        }

    if intent == Intent.GENERATE_SCRIPTS:
        file_path = params.get("file_path")
        lang = params["language_code"]
        count = int(params.get("count", 5))
        if file_path and Path(file_path).exists():
            chunks = chunk_file(file_path, lang)[:count]
        else:
            chunks = mock_scripts(lang, count)
        return {
            "chunks": [
                {"text": c.text, "word_count": c.word_count, "in_target_range": c.in_target_range}
                for c in chunks
            ]
        }

    if intent == Intent.INGEST:
        entry = process_audio(
            audio_path=Path(params["audio_path"]),
            transcription=params["transcription"],
            language_code=params["language_code"],
            speaker_id=params["speaker_id"],
            speaker_gender=params.get("speaker_gender", "U"),
            speaker_age_range=params.get("speaker_age_range", "18-30"),
            dialect=params.get("dialect", ""),
            split=params.get("split", "train"),
            output_dir=output_dir,
            manifest_path=manifest_path,
        )
        return {
            "hash_id": entry.hash_id,
            "duration": entry.duration,
            "quality_snr_db": entry.quality_snr_db,
            "quality_clipping": entry.quality_clipping,
            "is_clean": entry.is_clean(),
        }

    if intent == Intent.GET_STATUS:
        mp = Path(params.get("manifest_path", manifest_path))
        return manifest_stats(mp)

    return {"error": f"Unhandled intent: {intent}"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class AzizOrchestrator:
    """
    Stateless agent. Each call to .run() is independent.

    Args:
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        config:  Runtime configuration (manifest_path, output_dir, etc.).
    """

    def __init__(self, api_key: str | None = None, config: dict | None = None):
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._config = config or {}

    def run(self, user_input: str) -> AgentResponse:
        """
        Parse intent from user_input with Claude, then execute deterministically.

        Returns an AgentResponse with the intent, extracted params, and result dict.
        """
        response = self._client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=(
                "You are Aziz, an ASR data factory assistant. "
                "Your only job is to identify the user's intent and extract structured parameters. "
                "Always use a tool — never reply with plain text. "
                "Supported languages: Yoruba (yo), Efik (efi), Ibibio (ibb)."
            ),
            messages=[{"role": "user", "content": user_input}],
            tools=_TOOLS,
        )

        # Extract tool use block
        tool_use = next(
            (block for block in response.content if block.type == "tool_use"),
            None,
        )

        if tool_use is None:
            raw = " ".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            return AgentResponse(
                intent=None,
                params={},
                result={"error": "Could not determine intent", "raw": raw},
                raw_reply=raw,
            )

        intent = Intent(tool_use.name)
        params = tool_use.input
        result = _execute(intent, params, self._config)

        return AgentResponse(
            intent=intent,
            params=params,
            result=result,
            raw_reply=str(result),
        )
