"""
idem.normalize — the single text normalizer shared by training and scoring.

Stage C of the v5 data pipeline (see CLAUDE-BRIEF-v5-pipeline.md). This is the
only place transcript text gets cleaned. Both the training path and score.py
must import normalize() from here rather than each defining their own
cleaning logic — that split is what caused the 30.21% vs 18.54% WER
discrepancy the brief describes.

Also runnable as a corpus-level check:

    python -m idem.normalize --check v5_inventory.jsonl

See check_corpus() below for what it reports.

Per CLAUDE-BRIEF §3a: transcripts record what was spoken, not the written
form ("fifty dollars", not "$50"; "third", not "3rd"). normalize() never
sees the audio, so a digit, currency symbol, or percent sign reaching it
means the transcript itself is wrong, not something normalize() can repair.
It raises ConventionViolation rather than guess which spoken form the
numeral stood for.

NOT yet implemented (out of scope until real corpus text justifies a specific
choice — see the TODO in tests/test_normalize.py):
  - noise tokens (<UNK>, <NON_SPEECH_NOISE>)
"""

import argparse
import json
import re
import unicodedata

ALLOWED = set("abcdefghijklmnopqrstuvwxyz' ")


class ConventionViolation(Exception):
    """Raised when text_raw breaks the §3a transcription convention.

    Not a ValueError: this is a data problem in the corpus, not a bad
    argument, and it shouldn't be caught by accident by generic
    `except ValueError` handling elsewhere.
    """


# A digit, a currency symbol, or a percent sign means the transcript recorded
# a written form rather than what was spoken (CLAUDE-BRIEF §3a: "fifty
# dollars", not "$50"; "third", not "3rd"). normalize() cannot repair this —
# it never sees the audio, so it has no way to know which spoken reading a
# numeral stood for. This detector is the one place that question gets
# answered, so that check_corpus() and normalize()'s step 3 share it instead
# of each guessing separately.
_CONVENTION_VIOLATION_RE = re.compile(r"[0-9$₦£€%]")


def violates_transcription_convention(text: str) -> bool:
    """True if text_raw contains a digit, currency symbol, or percent sign."""
    return bool(_CONVENTION_VIOLATION_RE.search(text))


# Step 2: Unicode lookalikes mapped to their plain-ASCII equivalents, done
# before punctuation removal so curly apostrophes survive as straight ones
# instead of being stripped out along with real punctuation.
_LOOKALIKES = {
    "‘": "'",   # ‘ left single quotation mark
    "’": "'",   # ’ right single quotation mark
    "′": "'",   # ′ prime — used as an apostrophe stand-in by some sources
    "“": '"',   # “ left double quotation mark
    "”": '"',   # ” right double quotation mark
    "″": '"',   # ″ double prime
    "–": "-",   # – en dash
    "—": "-",   # — em dash
}


def _is_removable_punctuation(ch: str) -> bool:
    """True for punctuation characters that step 5 should delete.

    The straight apostrophe is exempt — it's part of the allowed output
    charset, not punctuation to be removed.
    """
    if ch == "'":
        return False
    return unicodedata.category(ch).startswith("P")


def normalize(text: str) -> str:
    """Raw transcript -> training/scoring text. The ONLY normalizer."""

    # 1. Unicode NFC. Must run first — every later step matches on specific
    # characters, and NFC decides which characters those are.
    text = unicodedata.normalize("NFC", text)

    # 2. Map lookalike punctuation to plain ASCII equivalents.
    for lookalike, plain in _LOOKALIKES.items():
        text = text.replace(lookalike, plain)

    # 3. Do not expand digits. A written numeral has already lost which
    # spoken form it stood for (§3a) — normalize() never sees the audio, so
    # any expansion it performed would be a silent guess. Raise instead, so
    # the corpus scan (check_corpus) can catch these in bulk rather than
    # training or scoring silently on a guessed reading.
    if violates_transcription_convention(text):
        raise ConventionViolation(
            "text violates the §3a transcription convention (contains a "
            f"digit, currency symbol, or percent sign): {text!r}"
        )

    # 4. Hyphens become spaces.
    text = text.replace("-", " ")

    # 5. Replace all remaining punctuation with a space (apostrophe excepted).
    # A space, not deletion — punctuation sitting directly between two words
    # with no surrounding whitespace (a colon, a comma) would otherwise fuse
    # them into one unreadable token. Step 7 collapses any resulting extra
    # whitespace.
    text = "".join(" " if _is_removable_punctuation(ch) else ch for ch in text)

    # 6. Lowercase. Runs after the token-based steps above.
    text = text.lower()

    # 7. Collapse whitespace runs and strip leading/trailing space.
    text = re.sub(r"\s+", " ", text).strip()

    return text


_MAX_EXAMPLES = 5  # brief's Stage C acceptance: "five example transcripts"


def _iter_inventory_records(path: str):
    """Yield (line_number, record) from a JSONL inventory file.

    Fails loudly on anything unparseable or missing a field this scan needs
    (hard rule 2) — a corpus scan that silently skipped bad records could
    report a clean corpus that isn't. "id" is required alongside text_raw
    and source (not just used if present) because every example this scan
    reports needs one: a human reading "slr70: 412" with five bare sentences
    has no way back to the actual records without it.
    """
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            for field in ("id", "text_raw", "source"):
                if field not in record:
                    raise ValueError(
                        f"{path}:{lineno}: record missing required field {field!r}"
                    )
            yield lineno, record


def check_corpus(path: str) -> dict:
    """Scan a v5_inventory.jsonl-shaped file and report three things the
    brief's Stage C acceptance requires, each grouped by source:

    1. "violations": records whose text_raw contains a digit, currency
       symbol, or percent sign (CLAUDE-BRIEF §3a). Checked directly on the
       raw text before calling normalize() — normalize() raises
       ConventionViolation on exactly this, and a corpus scan must keep
       going and count every offending record rather than dying on the
       first one (hard rule 3).
    2. "oov": characters in normalize()'s output that aren't in ALLOWED.
       Only checked for records that passed the violation check above,
       since normalize() would otherwise raise before returning anything.
    3. "empties": records whose text_raw is non-empty but normalizes to "".

    Also returns "totals" (record count per source) and "total_records"
    (overall). A bare count like "slr70: 412" is meaningless without knowing
    whether that's out of 500 records or 130,000, and hard rule 3's "a
    filter dropping over 1% stops for human investigation" can't be applied
    to this report's own output without a denominator to measure against.

    Each example carries its "id" and source-file "line" alongside the
    text, not just the text — a human reading a bucket has to go find and
    fix the record it names, and grepping the inventory for a sentence
    fragment doesn't scale.

    Does not decide anything — per the brief, a human reads this output and
    decides what (if anything) to do about it.
    """
    violations: dict[str, dict] = {}
    oov: dict[str, dict] = {}
    empties: dict[str, dict] = {}
    totals: dict[str, int] = {}
    total_records = 0

    for lineno, record in _iter_inventory_records(path):
        text_raw = record["text_raw"]
        source = record["source"]
        example = {"id": record["id"], "line": lineno, "text": text_raw}

        total_records += 1
        totals[source] = totals.get(source, 0) + 1

        if violates_transcription_convention(text_raw):
            bucket = violations.setdefault(source, {"count": 0, "examples": []})
            bucket["count"] += 1
            if len(bucket["examples"]) < _MAX_EXAMPLES:
                bucket["examples"].append(example)
            continue

        normalized = normalize(text_raw)

        for ch in set(normalized) - ALLOWED:
            entry = oov.setdefault(ch, {"count": 0, "by_source": {}, "examples": {}})
            entry["count"] += 1
            entry["by_source"][source] = entry["by_source"].get(source, 0) + 1
            examples = entry["examples"].setdefault(source, [])
            if len(examples) < _MAX_EXAMPLES:
                examples.append(example)

        if text_raw.strip() and not normalized:
            bucket = empties.setdefault(source, {"count": 0, "examples": []})
            bucket["count"] += 1
            if len(bucket["examples"]) < _MAX_EXAMPLES:
                bucket["examples"].append(example)

    return {
        "violations": violations,
        "oov": oov,
        "empties": empties,
        "totals": totals,
        "total_records": total_records,
    }


def _format_count(count: int, total: int) -> str:
    """"count" against "total" as a reader can judge against hard rule 3's
    1% threshold without doing the division themselves."""
    if total == 0:
        return str(count)
    return f"{count} / {total:,} ({100 * count / total:.1f}%)"


def _print_example(example: dict, indent: str) -> None:
    print(f"{indent}e.g. [id={example['id']} line={example['line']}] {example['text']!r}")


def _print_report(report: dict) -> None:
    totals = report["totals"]
    print(f"Scanned {report['total_records']:,} records.")

    print()
    print("=== §3a convention violations (digit, currency symbol, or %) ===")
    if not report["violations"]:
        print("  none")
    for source in sorted(report["violations"]):
        bucket = report["violations"][source]
        print(f"  {source}: {_format_count(bucket['count'], totals.get(source, 0))}")
        for example in bucket["examples"]:
            _print_example(example, "    ")

    print()
    print("=== Out-of-vocabulary characters (after normalize()) ===")
    if not report["oov"]:
        print("  none")
    for ch in sorted(report["oov"]):
        entry = report["oov"][ch]
        print(f"  {ch!r} (U+{ord(ch):04X}): {entry['count']} occurrences")
        for source in sorted(entry["by_source"]):
            n = entry["by_source"][source]
            print(f"    {source}: {_format_count(n, totals.get(source, 0))}")
            for example in entry["examples"].get(source, []):
                _print_example(example, "      ")

    print()
    print("=== Non-empty transcripts that normalize to empty ===")
    if not report["empties"]:
        print("  none")
    for source in sorted(report["empties"]):
        bucket = report["empties"][source]
        print(f"  {source}: {_format_count(bucket['count'], totals.get(source, 0))}")
        for example in bucket["examples"]:
            _print_example(example, "    ")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m idem.normalize")
    parser.add_argument(
        "--check",
        metavar="INVENTORY_JSONL",
        required=True,
        help="JSONL file with 'id', 'text_raw', and 'source' fields per "
        "record (e.g. v5_inventory.jsonl)",
    )
    args = parser.parse_args(argv)
    report = check_corpus(args.check)
    _print_report(report)


if __name__ == "__main__":
    main()
