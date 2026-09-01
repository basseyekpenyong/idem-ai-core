"""
Stage C — normalize() test suite.

Per CLAUDE-BRIEF-v5-pipeline.md §6 Stage C: tests written before the
implementation, so a human reviews the intent before any code exists.
`engine/normalize.py` does not exist yet — this file is expected to fail
to collect until it does. That is the stage boundary, not a bug.

normalize() is the ONE function used identically at training time and
scoring time. The bug this whole stage exists to fix: training and scoring
previously used different text cleaning, which cost 30.21% vs 18.54% WER
on the same model and audio (see brief §5). Every test below protects
against that class of bug recurring.
"""

from engine.normalize import normalize, ALLOWED

# Representative strings covering each normalization scenario named in the
# brief. Synthetic placeholders — Stage A (inventory) doesn't exist yet, so
# there is no real corpus to draw samples from. Replace/augment with real
# transcript text once v5_inventory.jsonl exists (brief §6 Stage C
# acceptance criteria calls for a corpus-level OOV-character pass separately
# from these unit tests).
SAMPLES = [
    "Don't stop believing.",
    "Don’t stop believing.",
    "It's a well-being check-up.",
    "In 1995 I moved to Lagos.",
    "The price is $1,234.56.",
    "She said <UNK> and <NON_SPEECH_NOISE> during the call.",
    "Twenty-three going on twenty-four.",
    "  Multiple   spaces   here.  ",
    "MIXED Case SENTENCE.",
    "He asked, “what time is it?”",
    "Wait — really?",
]


def test_curly_and_straight_apostrophe_agree():
    """
    Invariant: a curly apostrophe (’) and a straight one (') must produce
    identical training text. If they don't, the same word trains as two
    different tokens depending on which transcription tool produced the
    source file — silently splitting what should be one vocabulary entry.
    """
    assert normalize("Don't") == normalize("Don’t") == "don't"


def test_hyphen_becomes_space():
    """
    Invariant: hyphens split into separate words (step 4, after digit
    expansion in step 3). If this ran before digit expansion, num2words'
    own hyphens (e.g. "twenty-three") would be destroyed prematurely.
    """
    assert normalize("well-being") == "well being"


def test_digits_expand():
    """
    Invariant: digits expand to words before hyphen-splitting, so a decoder
    trained on spoken-form text can score digit-containing references at all.
    """
    assert normalize("in 1995") == "in nineteen ninety five"


def test_idempotent():
    """
    Invariant: normalize(normalize(x)) == normalize(x). If normalization
    isn't a fixed point, applying it once at training time and (incorrectly)
    twice at scoring time — or vice versa — produces different text, which
    is exactly the training/scoring divergence this stage exists to close.
    """
    for s in SAMPLES:
        assert normalize(normalize(s)) == normalize(s)


def test_output_charset():
    """
    Invariant: every normalized string uses only ALLOWED characters. A
    stray character reaching the CTC output layer shifts the vocabulary's
    index mapping and silently scrambles the model's output with no error
    raised (brief §3a) — this is the last line of defense against that.
    """
    for s in SAMPLES:
        assert set(normalize(s)) <= ALLOWED


def test_no_empty_from_nonempty():
    """
    Invariant: normalizing non-empty, non-trivial text never yields an
    empty string. An empty result silently drops a training/scoring record
    — the `startswith` bug in a new costume (brief §7).
    """
    assert normalize("Hello there.") != ""
