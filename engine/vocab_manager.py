"""
Vocabulary and character set manager.

Extends the base language profiles in config/language_profiles.py with
persistent custom additions — so a linguist can refine the Efik, Ibibio,
or Yoruba character inventory without editing source code.

Custom additions are saved to data/vocab/<lang>_custom.json and loaded
automatically on every run.

Usage (direct):
    mgr = VocabManager("efi")
    mgr.add_char("ʌ", note="open back unrounded vowel, confirmed by linguist")
    mgr.add_combining("̂", note="circumflex for falling tone")
    result = mgr.test_text("ʌkɔ mme eka")
    print(mgr.export_reference())

Usage (via Aziz):
    aziz.run("add the character ʌ to Efik vocabulary")
    aziz.run("test this Efik word: ʌkɔ mme")
    aziz.run("show me the full Efik character set")
    aziz.run("export the Efik reference sheet")
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import FrozenSet

from config.language_profiles import get_profile, LanguageProfile, REGISTRY

_VOCAB_DIR = Path(__file__).resolve().parent.parent / "data" / "vocab"


# ---------------------------------------------------------------------------
# Custom additions schema
# ---------------------------------------------------------------------------

@dataclass
class CustomAdditions:
    language_code: str
    added_base_chars: list[str] = field(default_factory=list)
    added_combining: list[str] = field(default_factory=list)
    removed_chars: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)  # char → note

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CustomAdditions":
        if not path.exists():
            return cls(language_code=path.stem.replace("_custom", ""))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# VocabManager
# ---------------------------------------------------------------------------

class VocabManager:
    """
    Manages the effective character set for one language, combining
    the base profile with persistent custom additions.

    All changes are auto-saved to data/vocab/<lang>_custom.json.
    """

    def __init__(self, language_code: str):
        self.language_code = language_code
        self._base = get_profile(language_code)
        self._custom_path = _VOCAB_DIR / f"{language_code}_custom.json"
        self._custom = CustomAdditions.load(self._custom_path)

    # ── Effective character sets ───────────────────────────────────────────

    @property
    def effective_base_chars(self) -> FrozenSet[str]:
        added   = frozenset(self._custom.added_base_chars)
        removed = frozenset(self._custom.removed_chars)
        return (self._base.base_chars | added) - removed

    @property
    def effective_combining(self) -> FrozenSet[str]:
        added   = frozenset(self._custom.added_combining)
        removed = frozenset(self._custom.removed_chars)
        return (self._base.extra_combining | added) - removed

    def is_char_allowed(self, char: str) -> bool:
        cat = unicodedata.category(char)
        if cat == "Mn":
            return char in self.effective_combining
        return char in self.effective_base_chars or char in self._base.punctuation

    # ── Mutations ─────────────────────────────────────────────────────────

    def add_char(self, char: str, note: str = "") -> dict:
        """Add a character to the language's base character set."""
        char = unicodedata.normalize("NFC", char)
        cat  = unicodedata.category(char)
        name = unicodedata.name(char, "UNKNOWN")

        if cat == "Mn":
            if char not in self._custom.added_combining:
                self._custom.added_combining.append(char)
        else:
            if char not in self._custom.added_base_chars:
                self._custom.added_base_chars.append(char)

        # Remove from removed list if it was there
        if char in self._custom.removed_chars:
            self._custom.removed_chars.remove(char)

        if note:
            self._custom.notes[char] = note

        self._save()
        return {
            "action": "added",
            "char": char,
            "codepoint": f"U+{ord(char):04X}",
            "unicode_name": name,
            "category": cat,
            "note": note,
            "language": self.language_code,
        }

    def add_combining(self, char: str, note: str = "") -> dict:
        """Add a combining mark to the allowed set (explicitly as combining)."""
        char = unicodedata.normalize("NFC", char)
        if char not in self._custom.added_combining:
            self._custom.added_combining.append(char)
        if char in self._custom.removed_chars:
            self._custom.removed_chars.remove(char)
        if note:
            self._custom.notes[char] = note
        self._save()
        return {
            "action": "added_combining",
            "char": char,
            "codepoint": f"U+{ord(char):04X}",
            "unicode_name": unicodedata.name(char, "UNKNOWN"),
        }

    def remove_char(self, char: str, note: str = "") -> dict:
        """Remove a character from the effective set (overrides base profile)."""
        char = unicodedata.normalize("NFC", char)
        if char not in self._custom.removed_chars:
            self._custom.removed_chars.append(char)
        # Remove from added lists too
        if char in self._custom.added_base_chars:
            self._custom.added_base_chars.remove(char)
        if char in self._custom.added_combining:
            self._custom.added_combining.remove(char)
        if note:
            self._custom.notes[char] = note
        self._save()
        return {
            "action": "removed",
            "char": char,
            "codepoint": f"U+{ord(char):04X}",
            "language": self.language_code,
        }

    def reset_custom(self) -> dict:
        """Remove all custom additions, restoring the base profile."""
        self._custom = CustomAdditions(language_code=self.language_code)
        self._save()
        return {"action": "reset", "language": self.language_code}

    # ── Inspection ────────────────────────────────────────────────────────

    def test_text(self, text: str) -> dict:
        """Test whether a text string is valid under the current effective profile."""
        normalized = unicodedata.normalize("NFC", text).strip()
        illegal: list[dict] = []
        for char in normalized:
            if not self.is_char_allowed(char):
                illegal.append({
                    "char": char,
                    "codepoint": f"U+{ord(char):04X}",
                    "unicode_name": unicodedata.name(char, "UNKNOWN"),
                })
        return {
            "is_valid": len(illegal) == 0,
            "normalized_text": normalized,
            "illegal_chars": illegal,
            "language": self.language_code,
        }

    def list_chars(self) -> dict:
        """Return the full effective character inventory."""

        def _describe(chars: FrozenSet[str]) -> list[dict]:
            result = []
            for c in sorted(chars, key=lambda ch: ord(ch)):
                result.append({
                    "char":         c,
                    "codepoint":    f"U+{ord(c):04X}",
                    "unicode_name": unicodedata.name(c, "UNKNOWN"),
                    "custom_note":  self._custom.notes.get(c, ""),
                    "is_custom":    c in self._custom.added_base_chars or c in self._custom.added_combining,
                    "is_removed":   c in self._custom.removed_chars,
                })
            return result

        return {
            "language":          self.language_code,
            "language_name":     self._base.name,
            "base_chars":        _describe(self.effective_base_chars),
            "combining_marks":   _describe(self.effective_combining),
            "punctuation":       sorted(self._base.punctuation),
            "total_base":        len(self.effective_base_chars),
            "custom_added":      len(self._custom.added_base_chars) + len(self._custom.added_combining),
            "custom_removed":    len(self._custom.removed_chars),
        }

    def export_reference(self, output_path: str | None = None) -> dict:
        """
        Export the effective character inventory as a human-readable Markdown
        reference document. Useful for sharing with linguists for sign-off.

        Args:
            output_path: Where to save the .md file. If None, saves to
                         data/vocab/<lang>_reference.md
        """
        inv = self.list_chars()
        lang_name = inv["language_name"]
        lang_code = inv["language"]

        lines = [
            f"# {lang_name} ({lang_code}) — Character Reference",
            "",
            f"Generated by IdemAI Vocab Manager.  "
            f"Total base chars: {inv['total_base']}  |  "
            f"Custom additions: {inv['custom_added']}  |  "
            f"Overrides removed: {inv['custom_removed']}",
            "",
            "---",
            "",
            "## Base Characters",
            "",
            "| Char | Codepoint | Unicode Name | Custom | Note |",
            "|------|-----------|--------------|--------|------|",
        ]
        for c in inv["base_chars"]:
            custom_flag = "✅ Added" if c["is_custom"] else ""
            note = c["custom_note"]
            lines.append(
                f"| `{c['char']}` | {c['codepoint']} | {c['unicode_name']} | {custom_flag} | {note} |"
            )

        lines += [
            "",
            "## Combining Marks (Tone Marks)",
            "",
            "| Mark | Codepoint | Unicode Name | Custom | Note |",
            "|------|-----------|--------------|--------|------|",
        ]
        for c in inv["combining_marks"]:
            custom_flag = "✅ Added" if c["is_custom"] else ""
            note = c["custom_note"]
            lines.append(
                f"| `{c['char']}` | {c['codepoint']} | {c['unicode_name']} | {custom_flag} | {note} |"
            )

        lines += [
            "",
            "## Allowed Punctuation",
            "",
            "```",
            " ".join(repr(p) for p in inv["punctuation"]),
            "```",
            "",
            "---",
            "*Share this document with a native speaker or linguist for sign-off before recording begins.*",
        ]

        content = "\n".join(lines)

        if output_path is None:
            output_path = str(_VOCAB_DIR / f"{lang_code}_reference.md")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(content, encoding="utf-8")

        return {
            "exported_to":  output_path,
            "language":     lang_code,
            "total_chars":  inv["total_base"],
            "custom_added": inv["custom_added"],
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        self._custom.save(self._custom_path)


# ---------------------------------------------------------------------------
# Convenience: get a manager by language code
# ---------------------------------------------------------------------------

def get_manager(language_code: str) -> VocabManager:
    return VocabManager(language_code)


# ---------------------------------------------------------------------------
# Aziz tool definitions
# ---------------------------------------------------------------------------

VOCAB_TOOLS: list[dict] = [
    {
        "name": "manage_vocabulary",
        "description": (
            "Manage the character set / vocabulary for a language. "
            "Actions: add a character, remove a character, list all characters, "
            "test a word or sentence, export a reference sheet for linguist review. "
            "Use when the user wants to update Efik/Ibibio/Yoruba/Nigerian English characters, "
            "check if a word uses valid characters, or produce a character inventory document."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "add_combining", "remove", "test", "list", "export", "reset"],
                    "description": (
                        "add=add base character, add_combining=add tone/combining mark, "
                        "remove=remove character, test=validate text, list=show inventory, "
                        "export=write reference Markdown, reset=restore base profile"
                    ),
                },
                "language_code": {
                    "type": "string",
                    "enum": ["yo", "efi", "ibb", "en_NG"],
                },
                "char": {
                    "type": "string",
                    "description": "The character to add or remove (single Unicode char). Required for add/remove.",
                },
                "text": {
                    "type": "string",
                    "description": "The text to validate. Required for test action.",
                },
                "note": {
                    "type": "string",
                    "description": "Linguist note explaining why this character was added/removed.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Where to write the reference .md file (export action only).",
                },
            },
            "required": ["action", "language_code"],
        },
    },
]


def execute_vocab_tool(params: dict) -> dict:
    """Router for the manage_vocabulary tool."""
    action = params["action"]
    lang   = params["language_code"]
    mgr    = get_manager(lang)

    if action == "add":
        char = params.get("char", "")
        if not char:
            return {"error": "char is required for action=add"}
        return mgr.add_char(char, note=params.get("note", ""))

    if action == "add_combining":
        char = params.get("char", "")
        if not char:
            return {"error": "char is required for action=add_combining"}
        return mgr.add_combining(char, note=params.get("note", ""))

    if action == "remove":
        char = params.get("char", "")
        if not char:
            return {"error": "char is required for action=remove"}
        return mgr.remove_char(char, note=params.get("note", ""))

    if action == "test":
        text = params.get("text", "")
        if not text:
            return {"error": "text is required for action=test"}
        return mgr.test_text(text)

    if action == "list":
        return mgr.list_chars()

    if action == "export":
        return mgr.export_reference(output_path=params.get("output_path"))

    if action == "reset":
        return mgr.reset_custom()

    return {"error": f"Unknown action: {action!r}"}


VOCAB_EXECUTOR: dict = {
    "manage_vocabulary": execute_vocab_tool,
}
