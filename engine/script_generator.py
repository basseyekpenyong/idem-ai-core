"""
Script generator — chunks large text documents into recordable sentence-length segments.

Target: 8–20 words per chunk (~3–10 seconds at a moderate speaking rate of 2 words/sec).
Hard limits: 6 words min, 30 words max. Chunks outside this range are flagged.

Sentence splitting strategy:
1. Split on sentence-ending punctuation (. ! ?) while preserving the mark.
2. If a sentence is within the word-count window, emit it as-is.
3. If a sentence is too long, split further on clause boundaries (, ; :).
4. If still too long, split on word count.
5. If a sentence is too short, merge with the next until the window is satisfied
   or a natural boundary is hit.

No ML model is used — this is deterministic and runs offline.
"""

import re
from dataclasses import dataclass, field

MIN_WORDS = 6
MAX_WORDS = 30
TARGET_MIN = 8
TARGET_MAX = 20


@dataclass
class ScriptChunk:
    text: str
    word_count: int
    in_target_range: bool
    source_line: int   # 0-based index into the original paragraph list


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping trailing punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_clauses(sentence: str) -> list[str]:
    """Split a long sentence on clause boundaries."""
    parts = re.split(r"(?<=[,;:])\s+", sentence)
    return [p.strip() for p in parts if p.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def _merge_short(parts: list[str]) -> list[str]:
    """
    Merge consecutive short parts until each merged segment is at least MIN_WORDS.
    This avoids emitting tiny stubs that are too short to record usefully.
    """
    merged: list[str] = []
    buffer = ""
    for part in parts:
        candidate = (buffer + " " + part).strip() if buffer else part
        if _word_count(candidate) < MIN_WORDS:
            buffer = candidate
        else:
            merged.append(candidate)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] = (merged[-1] + " " + buffer).strip()
        else:
            merged.append(buffer)
    return merged


def _chunk_sentence(sentence: str) -> list[str]:
    """Break a single sentence into ≤MAX_WORDS chunks."""
    if _word_count(sentence) <= MAX_WORDS:
        return [sentence]

    # Try clause splitting first
    clauses = _split_clauses(sentence)
    if len(clauses) > 1:
        result: list[str] = []
        for clause in clauses:
            result.extend(_chunk_sentence(clause))
        return _merge_short(result)

    # Fall back to hard word-count split
    words = sentence.split()
    chunks: list[str] = []
    for i in range(0, len(words), MAX_WORDS):
        chunks.append(" ".join(words[i: i + MAX_WORDS]))
    return chunks


def chunk_document(text: str, source_line: int = 0) -> list[ScriptChunk]:
    """
    Chunk a paragraph or document into recordable script segments.

    Args:
        text:        Input text (one or more paragraphs).
        source_line: Origin line number in the source document (for traceability).

    Returns:
        List of ScriptChunk objects, each ready for a recording session.
    """
    sentences = _split_sentences(text)
    raw_chunks: list[str] = []
    for sentence in sentences:
        raw_chunks.extend(_chunk_sentence(sentence))

    raw_chunks = _merge_short(raw_chunks)

    result: list[ScriptChunk] = []
    for chunk in raw_chunks:
        wc = _word_count(chunk)
        result.append(
            ScriptChunk(
                text=chunk,
                word_count=wc,
                in_target_range=TARGET_MIN <= wc <= TARGET_MAX,
                source_line=source_line,
            )
        )
    return result


def chunk_file(file_path: str, language_code: str = "") -> list[ScriptChunk]:
    """
    Read a plain-text file (one paragraph per non-empty line) and chunk every paragraph.

    Args:
        file_path:     Path to the source text file.
        language_code: Optional — stored in logs but not used for chunking logic.

    Returns:
        Flat list of ScriptChunk objects across all paragraphs.
    """
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    all_chunks: list[ScriptChunk] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            all_chunks.extend(chunk_document(line, source_line=i))
    return all_chunks


# ---------------------------------------------------------------------------
# Mock translation database (for testing without real translated data)
# ---------------------------------------------------------------------------
_MOCK_DB: dict[str, list[str]] = {
    "yo": [
        "Ẹ káàárọ̀, báwo ni o ṣe wà?",
        "Mo fẹ́ kọ èdè Yorùbá.",
        "Ilé wa wà ní ìlú Èkó.",
        "Omi tí ó tutù dára fún ilera.",
        "Àwọn ọmọ ń kọ́ nínú ilé ẹ̀kọ́.",
    ],
    "efi": [
        "Mme eyen ọkọ ama ikọ.",
        "Ụtọm ọkọ edien eka.",
        "Ọfọn mme eka edi ufọk.",
    ],
    "ibb": [
        "Ànyị bịa n'ụlọ akwụkwọ.",
        "Mmiri dị mma maka ahụike.",
        "Anyị hụrụ ụlọ n'ụzọ.",
    ],
}


def mock_scripts(language_code: str, n: int = 5) -> list[ScriptChunk]:
    """Return mock script chunks for a given language (used in tests / demo mode)."""
    sentences = _MOCK_DB.get(language_code, [])
    if not sentences:
        raise ValueError(
            f"No mock data for language {language_code!r}. "
            f"Available: {list(_MOCK_DB)}"
        )
    selected = (sentences * ((n // len(sentences)) + 1))[:n]
    chunks: list[ScriptChunk] = []
    for i, s in enumerate(selected):
        wc = _word_count(s)
        chunks.append(ScriptChunk(text=s, word_count=wc,
                                  in_target_range=TARGET_MIN <= wc <= TARGET_MAX,
                                  source_line=i))
    return chunks
