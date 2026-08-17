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

_DIGIT_RUN_RE = re.compile(r"\d+")


def _expand_digit_run(match: re.Match) -> str:
    """
    Expand one run of digits to words.

    ASSUMPTION — flag before relying on it: a 4-digit run is treated as a
    year ("1995" -> "nineteen ninety-five"); anything else is a plain
    cardinal number ("23" -> "twenty-three"). This satisfies the one digit
    test the brief specifies today, but it's a guess about which transcripts
    contain years vs. plain counts vs. phone numbers vs. IDs. Needs checking
    against real corpus text once Stage A produces v5_inventory.jsonl.
    """
    digits = match.group()
    value = int(digits)
    if len(digits) == 4:
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

    # 5. Remove all remaining punctuation (apostrophe excepted).
    text = "".join(ch for ch in text if not _is_removable_punctuation(ch))

    # 6. Lowercase. Runs after the token-based steps above.
    text = text.lower()

    # 7. Collapse whitespace runs and strip leading/trailing space.
    text = re.sub(r"\s+", " ", text).strip()

    return text
