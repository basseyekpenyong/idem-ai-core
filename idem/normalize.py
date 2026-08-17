"""
idem.normalize — the single text normalizer shared by training and scoring.

Stage C of the v5 data pipeline (see CLAUDE-BRIEF-v5-pipeline.md). This is the
only place transcript text gets cleaned. Both the training path and score.py
must import normalize() from here rather than each defining their own
cleaning logic — that split is what caused the 30.21% vs 18.54% WER
discrepancy the brief describes.

NOT yet implemented (out of scope until real corpus text justifies a specific
choice — see the TODO in tests/test_normalize.py):
  - noise tokens (<UNK>, <NON_SPEECH_NOISE>)
  - ordinals ("23rd")
  - currency ("$100")
"""

import re
import unicodedata

from num2words import num2words

ALLOWED = set("abcdefghijklmnopqrstuvwxyz' ")

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
