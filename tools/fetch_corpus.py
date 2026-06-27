"""
Corpus fetcher — downloads public-domain text for Yoruba, Efik, and Ibibio
from verified open sources, cleans it, and writes one paragraph per line
to data/corpus/<language_code>.txt.

Sources used:
  Yoruba  — Yorùbá Bible (JW / public domain via bible.com API, public export)
            Afriqa corpus (CC-BY) https://github.com/masakhane-io/afriqa
            Global Voices Yoruba (CC-BY) https://opus.nlpl.eu/GlobalVoices.php
  Efik    — JW public Bible text (public domain)
            Masakhane NER dataset (CC-BY-4.0)
  Ibibio  — Limited sources; JW Bible subset; manual entry recommended

Usage:
    python tools/fetch_corpus.py --lang yo          # Yoruba only
    python tools/fetch_corpus.py --lang efi ibb     # Efik + Ibibio
    python tools/fetch_corpus.py                    # All languages

Output:
    data/corpus/yo.txt   — one paragraph per line, NFC-normalised, blank lines stripped
    data/corpus/efi.txt
    data/corpus/ibb.txt

After downloading, run the script generator:
    from engine.script_generator import chunk_file
    chunks = chunk_file("data/corpus/yo.txt", "yo")
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CORPUS_DIR = _REPO_ROOT / "data" / "corpus"

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
# Each entry is (url, description, encoding).
# These are plain-text files from open/public-domain sources.
# Replace or extend with your own verified sources as the project grows.

SOURCES: dict[str, list[dict]] = {
    "yo": [
        {
            "url": "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data/yo/train.txt",
            "description": "Masakhane NER Yoruba train split (CC-BY-4.0)",
            "encoding": "utf-8",
            "is_conll": True,  # CoNLL format — extract text tokens only
        },
        {
            "url": "https://raw.githubusercontent.com/masakhane-io/masakhane-mt/master/benchmarks/jw300-yo-en/train.yo",
            "description": "JW300 Yoruba-English parallel corpus — Yoruba side (public domain)",
            "encoding": "utf-8",
            "is_conll": False,
        },
    ],
    "efi": [
        {
            "url": "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data/efi/train.txt",
            "description": "Masakhane NER Efik train split (CC-BY-4.0)",
            "encoding": "utf-8",
            "is_conll": True,
        },
    ],
    "ibb": [
        # Ibibio has very limited digitised resources — start with the Masakhane NER set
        # and supplement with manual transcriptions.
        {
            "url": "https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main/data/yo/train.txt",
            "description": "PLACEHOLDER — replace with real Ibibio source",
            "encoding": "utf-8",
            "is_conll": True,
            "placeholder": True,
        },
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch(url: str, encoding: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "idem-ai-corpus-fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode(encoding)


def _extract_conll_text(raw: str) -> list[str]:
    """
    Extract word tokens from CoNLL-format NER data and reconstruct sentences.
    CoNLL format: one token per line, blank lines = sentence boundary.
    """
    sentences: list[str] = []
    current: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            if current:
                sentences.append(" ".join(current))
                current = []
        else:
            token = line.split()[0]
            if not token.startswith("-"):
                current.append(token)
    if current:
        sentences.append(" ".join(current))
    return sentences


def _clean_lines(lines: list[str]) -> list[str]:
    """NFC-normalise, strip, and drop blank or very short lines."""
    cleaned = []
    for line in lines:
        line = unicodedata.normalize("NFC", line).strip()
        if len(line.split()) >= 3:  # skip stubs
            cleaned.append(line)
    return cleaned


def fetch_language(lang: str, output_dir: Path) -> Path:
    sources = SOURCES.get(lang)
    if not sources:
        print(f"  No sources configured for {lang!r}. Skipping.", file=sys.stderr)
        return None

    all_lines: list[str] = []

    for src in sources:
        if src.get("placeholder"):
            print(f"  ⚠️  {lang}: source is a placeholder — replace in tools/fetch_corpus.py")
            continue

        print(f"  Fetching: {src['description']}")
        try:
            raw = _fetch(src["url"], src["encoding"])
        except Exception as e:
            print(f"  ✗ Failed ({e})", file=sys.stderr)
            continue

        if src.get("is_conll"):
            lines = _extract_conll_text(raw)
        else:
            lines = raw.splitlines()

        cleaned = _clean_lines(lines)
        all_lines.extend(cleaned)
        print(f"  ✓ {len(cleaned):,} lines")

    if not all_lines:
        print(f"  No lines collected for {lang}. Check sources.", file=sys.stderr)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{lang}.txt"
    out_path.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"  → Saved {len(all_lines):,} lines to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public-domain corpus text for IdemAI languages.")
    parser.add_argument(
        "--lang", nargs="*", default=list(SOURCES.keys()),
        help="Language code(s) to fetch: yo efi ibb (default: all)",
    )
    parser.add_argument(
        "--output-dir", default=str(_CORPUS_DIR),
        help=f"Output directory (default: {_CORPUS_DIR})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Corpus output directory: {output_dir}\n")

    for lang in args.lang:
        print(f"[{lang.upper()}]")
        fetch_language(lang, output_dir)
        print()

    print("Done. Next step:")
    print("  from engine.script_generator import chunk_file")
    print("  chunks = chunk_file('data/corpus/yo.txt', 'yo')")


if __name__ == "__main__":
    main()
