"""
Aziz Agent orchestrator.

Architecture:
- An LLM handles ONLY intent parsing and parameter extraction.
- All file I/O, Drive calls, and pipeline execution are deterministic Python.
- The LLM never touches the filesystem or API credentials directly.

Supported backends (auto-detected from env / api_key argument):
  Gemini   — GOOGLE_API_KEY or GEMINI_API_KEY  (free tier: 1500 req/day)
  Anthropic — ANTHROPIC_API_KEY

Supported tool categories:
  Pipeline  — ingest_audio, validate_text, generate_scripts, get_status, export_dataset
  Drive     — list_drive_files, download_drive_file, upload_to_drive,
              rename_drive_file, move_drive_file, generate_and_upload_notebook
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.audio_pipeline import export_dataset, manifest_stats, process_audio
from engine.script_generator import chunk_file, mock_scripts
from engine.validator import validate
from engine.media_processor import MEDIA_EXECUTOR, MEDIA_TOOLS
from engine.vocab_manager import VOCAB_EXECUTOR, VOCAB_TOOLS
from agent.tools.drive_tools import DRIVE_EXECUTOR, DRIVE_TOOLS

_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_GEMINI_MODEL    = "gemini-flash-lite-latest"

_ALL_LANGUAGES = ["efi", "ibb", "en_NG"]

SYSTEM_PROMPT = (
    "You are Aziz, a fully autonomous ASR data factory assistant running on the user's laptop. "
    "You can do anything the user needs without them having to do it manually. "
    "Identify intent and extract parameters — always use a tool, never plain text. "
    "Capabilities: "
    "(1) Pipeline — ingest audio, validate text, generate recording scripts, export clean dataset.json for training. "
    "(2) Google Drive — list, download, upload, rename, move files; generate and upload Whisper training notebooks. "
    "(3) Media — convert text files to speech (TTS), transcribe any audio file to text (STT), "
    "browse local files (file explorer), rename/move local files, fix file extensions automatically. "
    "Supported languages: Efik (efi), Ibibio (ibb), Nigerian English (en_NG). "
    "(4) Vocabulary — add/remove characters from a language's whitelist, test words, "
    "list the full character inventory, export a reference sheet for linguist sign-off. "
    "Custom character changes are saved permanently to data/vocab/<lang>_custom.json."
)


@dataclass
class AgentResponse:
    tool_name: str | None
    params: dict[str, Any]
    result: dict[str, Any]
    raw_reply: str


# ---------------------------------------------------------------------------
# Pipeline tool definitions
# ---------------------------------------------------------------------------
_PIPELINE_TOOLS: list[dict] = [
    {
        "name": "ingest_audio",
        "description": (
            "Ingest and process a recorded audio file into the training dataset. "
            "Use when the user mentions recording, adding audio, or ingesting a file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "audio_path":      {"type": "string"},
                "transcription":   {"type": "string"},
                "language_code":   {"type": "string", "enum": _ALL_LANGUAGES},
                "speaker_id":      {"type": "string"},
                "speaker_gender":  {"type": "string", "enum": ["M", "F", "U"]},
                "speaker_age_range": {"type": "string", "enum": ["18-30", "31-45", "46-60", "60+"]},
                "dialect":         {"type": "string", "default": ""},
                "split":           {"type": "string", "enum": ["train", "dev", "test"], "default": "train"},
            },
            "required": ["audio_path", "transcription", "language_code", "speaker_id"],
        },
    },
    {
        "name": "validate_text",
        "description": (
            "Validate a text string against the character rules for a target language. "
            "Use when the user asks to check, validate, or verify text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text":          {"type": "string"},
                "language_code": {"type": "string", "enum": _ALL_LANGUAGES},
            },
            "required": ["text", "language_code"],
        },
    },
    {
        "name": "generate_scripts",
        "description": (
            "Generate recordable script chunks from a text file or mock data. "
            "Use when the user asks for sentences to record, script generation, or chunking."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "language_code": {"type": "string", "enum": _ALL_LANGUAGES},
                "file_path":     {"type": "string", "description": "Path to source text file (optional)"},
                "count":         {"type": "integer", "default": 5},
            },
            "required": ["language_code"],
        },
    },
    {
        "name": "get_status",
        "description": (
            "Get the current dataset status: total hours, entries, quality flags, per-language breakdown. "
            "Use when the user asks about progress, stats, hours collected, or dataset size."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "manifest_path": {"type": "string"},
            },
        },
    },
    {
        "name": "export_dataset",
        "description": (
            "Export a clean dataset.json from the manifest for model training. "
            "Filters out clipped and low-quality recordings. Splits into train/dev/test. "
            "Use when the user asks to export data, generate a training file, or prepare data for training."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "output_path":      {"type": "string", "description": "Where to write dataset.json"},
                "language_code":    {"type": "string", "enum": _ALL_LANGUAGES + ["all"], "description": "Language to export, or 'all'"},
                "min_snr_db":       {"type": "number", "default": 15.0},
                "include_clipped":  {"type": "boolean", "default": False},
            },
            "required": ["output_path"],
        },
    },
]

_ALL_TOOLS = _PIPELINE_TOOLS + DRIVE_TOOLS + MEDIA_TOOLS + VOCAB_TOOLS


# ---------------------------------------------------------------------------
# Execution router
# ---------------------------------------------------------------------------
_DEFAULT_MANIFEST = Path("master_manifest.jsonl")
_DEFAULT_OUTPUT   = Path("data/processed")


def _execute(tool_name: str, params: dict, config: dict) -> dict[str, Any]:
    manifest_path = Path(config.get("manifest_path", _DEFAULT_MANIFEST))
    output_dir    = Path(config.get("output_dir",    _DEFAULT_OUTPUT))

    # ── Pipeline tools ──────────────────────────────────────────────────────

    if tool_name == "validate_text":
        r = validate(params["text"], params["language_code"])
        return {"is_valid": r.is_valid, "normalized_text": r.normalized_text, "errors": r.errors}

    if tool_name == "generate_scripts":
        lang  = params["language_code"]
        count = int(params.get("count", 5))
        fp    = params.get("file_path")
        chunks = chunk_file(fp, lang)[:count] if fp and Path(fp).exists() else mock_scripts(lang, count)
        return {"chunks": [{"text": c.text, "word_count": c.word_count, "in_target_range": c.in_target_range} for c in chunks]}

    if tool_name == "ingest_audio":
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
        return {"hash_id": entry.hash_id, "duration": entry.duration,
                "quality_snr_db": entry.quality_snr_db, "quality_clipping": entry.quality_clipping,
                "is_clean": entry.is_clean()}

    if tool_name == "get_status":
        mp = Path(params.get("manifest_path", manifest_path))
        return manifest_stats(mp)

    if tool_name == "export_dataset":
        lang = params.get("language_code")
        lang = None if lang == "all" else lang
        return export_dataset(
            manifest_path=manifest_path,
            output_path=Path(params["output_path"]),
            language_code=lang,
            min_snr_db=float(params.get("min_snr_db", 15.0)),
            include_clipped=bool(params.get("include_clipped", False)),
        )

    # ── Drive tools ─────────────────────────────────────────────────────────
    if tool_name in DRIVE_EXECUTOR:
        return DRIVE_EXECUTOR[tool_name](params)

    # ── Media / file tools ──────────────────────────────────────────────────
    if tool_name in MEDIA_EXECUTOR:
        return MEDIA_EXECUTOR[tool_name](params)

    # ── Vocabulary management ────────────────────────────────────────────────
    if tool_name in VOCAB_EXECUTOR:
        return VOCAB_EXECUTOR[tool_name](params)

    return {"error": f"Unknown tool: {tool_name!r}"}


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _detect_backend(api_key: str | None) -> tuple[str, str]:
    """
    Return (backend, resolved_key) — checks the supplied key first, then env vars.
    Priority: Gemini > Anthropic (Gemini has a free tier so it's the easier default).
    """
    # If a key is supplied directly, detect by prefix — anything that isn't
    # an Anthropic key is treated as Gemini (covers both AIza... and AQ.Ab8... formats)
    if api_key and api_key.strip():
        key = api_key.strip()
        if key.startswith("sk-ant-"):
            return "anthropic", key
        return "gemini", key  # Gemini keys: AIza... or AQ.... both valid

    candidates = [
        ("gemini",    os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
        ("anthropic", os.environ.get("ANTHROPIC_API_KEY")),
    ]
    for backend, key in candidates:
        if key:
            return backend, key
    raise ValueError(
        "No API key found. Set one of these in your .env file:\n"
        "  GOOGLE_API_KEY=AIza...          (Gemini — free tier)\n"
        "  ANTHROPIC_API_KEY=sk-ant-...    (Claude Haiku)\n"
        "Or paste your key into the sidebar in the dashboard."
    )


# ---------------------------------------------------------------------------
# Gemini tool-call helpers  (uses google-genai >= 1.0)
# ---------------------------------------------------------------------------

def _to_gemini_schema(spec: dict):
    from google.genai import types
    type_map = {
        "string": types.Type.STRING, "integer": types.Type.INTEGER,
        "number": types.Type.NUMBER, "boolean": types.Type.BOOLEAN,
        "array":  types.Type.ARRAY,  "object":  types.Type.OBJECT,
    }
    t = type_map.get(spec.get("type", "string"), types.Type.STRING)
    kwargs = dict(
        type=t,
        description=spec.get("description", ""),
        enum=spec.get("enum") or [],
    )
    # Arrays need an items sub-schema; default to STRING items if unspecified
    if t == types.Type.ARRAY:
        items_spec = spec.get("items", {"type": "string"})
        kwargs["items"] = _to_gemini_schema(items_spec)
    return types.Schema(**kwargs)


def _to_gemini_function(t: dict):
    from google.genai import types
    props = {
        name: _to_gemini_schema(spec)
        for name, spec in t["input_schema"].get("properties", {}).items()
    }
    return types.FunctionDeclaration(
        name=t["name"],
        description=t["description"],
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties=props,
            required=t["input_schema"].get("required", []),
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class AzizOrchestrator:
    """
    Stateless agent. Each .run() call is independent.

    Args:
        api_key: Gemini (AIza...) or Anthropic (sk-ant-...) key.
                 Falls back to GOOGLE_API_KEY / ANTHROPIC_API_KEY env vars.
        config:  Runtime config — keys: manifest_path, output_dir.
    """

    def __init__(self, api_key: str | None = None, config: dict | None = None):
        self._backend, self._api_key = _detect_backend(api_key)
        self._config = config or {}

    @property
    def backend(self) -> str:
        return self._backend

    def run(self, user_input: str) -> AgentResponse:
        if self._backend == "gemini":
            return self._run_gemini(user_input)
        return self._run_anthropic(user_input)

    # ── Anthropic ────────────────────────────────────────────────────────────

    def _run_anthropic(self, user_input: str) -> AgentResponse:
        import anthropic
        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_input}],
            tools=_ALL_TOOLS,
        )
        tool_use = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_use is None:
            raw = " ".join(b.text for b in response.content if hasattr(b, "text"))
            return AgentResponse(tool_name=None, params={},
                                 result={"error": "No tool called", "raw": raw}, raw_reply=raw)
        result = _execute(tool_use.name, tool_use.input, self._config)
        return AgentResponse(tool_name=tool_use.name, params=tool_use.input,
                             result=result, raw_reply=str(result))

    # ── Gemini ───────────────────────────────────────────────────────────────

    def _run_gemini(self, user_input: str) -> AgentResponse:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        tool = types.Tool(function_declarations=[_to_gemini_function(t) for t in _ALL_TOOLS])
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[tool],
        )
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=user_input,
            config=config,
        )

        fc = None
        for part in response.candidates[0].content.parts:
            if part.function_call and part.function_call.name:
                fc = part.function_call
                break

        if fc is None:
            raw = response.text if hasattr(response, "text") else ""
            return AgentResponse(tool_name=None, params={},
                                 result={"error": "No tool called", "raw": raw}, raw_reply=raw)

        params = dict(fc.args)
        result = _execute(fc.name, params, self._config)
        return AgentResponse(tool_name=fc.name, params=params,
                             result=result, raw_reply=str(result))
