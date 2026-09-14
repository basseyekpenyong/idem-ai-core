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

NOT yet implemented (out of scope until real corpus text justifies a specific
choice — see the TODO in tests/test_normalize.py):
  - noise tokens (<UNK>, <NON_SPEECH_NOISE>)
  - ordinals ("23rd")
  - currency ("$100")
"""

import argparse
import json
import re
import unicodedata

from num2words import num2words

ALLOWED = set("abcdefghijklmnopqrstuvwxyz' ")

# A digit, a currency symbol, or a percent sign means the transcript recorded
# a written form rather than what was spoken (CLAUDE-BRIEF §3a: "fifty
# dollars", not "$50"; "third", not "3rd"). normalize() cannot repair this —
# it never sees the audio, so it has no way to know which spoken reading a
# numeral stood for. This detector is the one place that question gets
# answered, so that check_corpus() and any future revision of step 3 (which
# will raise on exactly this) share it instead of each guessing separately.
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
    "“": '"',   # “ left double quotation mark
    "”": '"',   # ” right double quotation mark
    "–": "-",   # – en dash
    "—": "-",   # — em dash
}

# A comma-grouped number ("3,500", "350,000") must be matched as one token —
# tried first, since regex alternation tries options left-to-right at each
# position — otherwise a plain \d+ would split it into "3" and "500" and
# read them as two unrelated numbers instead of one.
_DIGIT_RUN_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\d+")


def _expand_digit_run(match: re.Match) -> str:
    """
    Expand one run of digits to words.

    ASSUMPTION — flag before relying on it: a bare, non-comma-grouped 4-digit
    run is treated as a year ("1995" -> "nineteen ninety-five"); everything
    else is a plain cardinal number ("23" -> "twenty-three", "3,500" ->
    "three thousand, five hundred"). A comma-grouped run is never treated as
    a year — nobody writes a year as "1,998". This satisfies the one digit
    test the brief specifies today, but it's a guess about which transcripts
    contain years vs. plain counts. It's also known wrong for phone numbers
    and reference IDs (e.g. "08031245678"), which real speech reads
    digit-by-digit rather than as one huge number — that's an open decision,
    not yet implemented; see the Stage C findings sent for review.
    """
    raw = match.group()
    digits = raw.replace(",", "")
    value = int(digits)
    if "," not in raw and len(digits) == 4:
        return num2words(value, to="year")
    return num2words(value)


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

    # 3. Expand digit runs to words. Must run before step 4 (hyphens), since
    # num2words emits hyphens ("twenty-three") that step 4 needs to see.
    text = _DIGIT_RUN_RE.sub(_expand_digit_run, text)

    # 4. Hyphens become spaces — both literal hyphens already in the text and
    # ones num2words just introduced in step 3.
    text = text.replace("-", " ")

    # 5. Replace all remaining punctuation with a space (apostrophe excepted).
    # A space, not deletion — punctuation sitting directly between two tokens
    # with no surrounding whitespace (the comma num2words leaves inside
    # "three thousand, five hundred", or a source comma in "3,500" before
    # step 3 even runs on it) would otherwise fuse the two tokens into one
    # unreadable word. Step 7 collapses any resulting extra whitespace.
    text = "".join(" " if _is_removable_punctuation(ch) else ch for ch in text)

    # 6. Lowercase. Runs after the token-based steps above.
    text = text.lower()

    # 7. Collapse whitespace runs and strip leading/trailing space.
    text = re.sub(r"\s+", " ", text).strip()

    return text


_MAX_EXAMPLES = 5  # brief's Stage C acceptance: "five example transcripts"


def _iter_inventory_records(path: str):
    """Yield (line_number, record) from a JSONL inventory file.

    Fails loudly on anything unparseable or missing the two fields this scan
    needs (hard rule 2) — a corpus scan that silently skipped bad records
    could report a clean corpus that isn't.
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
            for field in ("text_raw", "source"):
                if field not in record:
                    raise ValueError(
                        f"{path}:{lineno}: record missing required field {field!r}"
                    )
            yield lineno, record


def check_corpus(path: str) -> dict:
    """Scan a v5_inventory.jsonl-shaped file and report three things the
    brief's Stage C acceptance requires, each grouped by source:

    1. "violations": records whose text_raw contains a digit, currency
       symbol, or percent sign (CLAUDE-BRIEF §3a). Detected directly on the
       raw text, independent of what normalize() currently does with it —
       even though step 3 today still expands these instead of rejecting
       them, they must still be counted, not made to disappear by expanding
       them (the brief is explicit: "do not expand numerals to make the
       count go down").
    2. "oov": characters in normalize()'s output that aren't in ALLOWED.
       Skipped for records already flagged as a §3a violation, since those
       are reported there instead.
    3. "empties": records whose text_raw is non-empty but normalizes to "".

    Returns counts and up to _MAX_EXAMPLES example transcripts per bucket.
    Does not decide anything — per the brief, a human reads this output and
    decides what (if anything) to do about it.
    """
    violations: dict[str, dict] = {}
    oov: dict[str, dict] = {}
    empties: dict[str, dict] = {}

    for _, record in _iter_inventory_records(path):
        text_raw = record["text_raw"]
        source = record["source"]

        if violates_transcription_convention(text_raw):
            bucket = violations.setdefault(source, {"count": 0, "examples": []})
            bucket["count"] += 1
            if len(bucket["examples"]) < _MAX_EXAMPLES:
                bucket["examples"].append(text_raw)
            continue

        normalized = normalize(text_raw)

        for ch in set(normalized) - ALLOWED:
            entry = oov.setdefault(ch, {"count": 0, "by_source": {}, "examples": {}})
            entry["count"] += 1
            entry["by_source"][source] = entry["by_source"].get(source, 0) + 1
            examples = entry["examples"].setdefault(source, [])
            if len(examples) < _MAX_EXAMPLES:
                examples.append(text_raw)

        if text_raw.strip() and not normalized:
            bucket = empties.setdefault(source, {"count": 0, "examples": []})
            bucket["count"] += 1
            if len(bucket["examples"]) < _MAX_EXAMPLES:
                bucket["examples"].append(text_raw)

    return {"violations": violations, "oov": oov, "empties": empties}


def _print_report(report: dict) -> None:
    print("=== §3a convention violations (digit, currency symbol, or %) ===")
    if not report["violations"]:
        print("  none")
    for source in sorted(report["violations"]):
        bucket = report["violations"][source]
        print(f"  {source}: {bucket['count']}")
        for example in bucket["examples"]:
            print(f"    e.g. {example!r}")

    print()
    print("=== Out-of-vocabulary characters (after normalize()) ===")
    if not report["oov"]:
        print("  none")
    for ch in sorted(report["oov"]):
        entry = report["oov"][ch]
        by_source = ", ".join(
            f"{source}={n}" for source, n in sorted(entry["by_source"].items())
        )
        print(f"  {ch!r} (U+{ord(ch):04X}): {entry['count']} occurrences [{by_source}]")
        for source, examples in entry["examples"].items():
            for example in examples:
                print(f"    e.g. ({source}) {example!r}")

    print()
    print("=== Non-empty transcripts that normalize to empty ===")
    if not report["empties"]:
        print("  none")
    for source in sorted(report["empties"]):
        bucket = report["empties"][source]
        print(f"  {source}: {bucket['count']}")
        for example in bucket["examples"]:
            print(f"    e.g. {example!r}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m idem.normalize")
    parser.add_argument(
        "--check",
        metavar="INVENTORY_JSONL",
        required=True,
        help="JSONL file with 'text_raw' and 'source' fields per record "
        "(e.g. v5_inventory.jsonl)",
    )
    args = parser.parse_args(argv)
    report = check_corpus(args.check)
    _print_report(report)


if __name__ == "__main__":
    main()
