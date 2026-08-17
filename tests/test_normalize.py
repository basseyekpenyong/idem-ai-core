"""
Tests for idem.normalize — the single normalizer shared by training and scoring.

No implementation exists yet (Stage C, v5 pipeline brief). These tests are
written first so the implementation has something concrete to satisfy.
"""

from idem.normalize import normalize

# Placeholder sample sentences for the idempotence/charset checks below.
# These are illustrative, not drawn from the real corpus — Stage A (inventory)
# hasn't run yet. Extend or replace with real transcript lines once it has.
SAMPLES = [
    "Hello there.",
    "Don't worry, we go see am tomorrow.",
    "It's a well-being programme for twenty-three families.",
    'She said, "no wahala" and left.',
    "In 1995, the government changed the policy.",
]


def test_curly_and_straight_apostrophe_agree():
    # A curly apostrophe (’) and a straight one (') must normalize to the same
    # text. They're visually identical in most fonts but different Unicode
    # code points — treating them as different characters silently forks
    # "don't" into two distinct forms depending on which tool produced the
    # transcript, one way at training time and maybe the other at scoring
    # time. That kind of quiet mismatch is exactly what this rebuild exists
    # to catch.
    assert normalize("Don't") == normalize("Don’t") == "don't"


def test_hyphen_becomes_space():
    # Hyphens must become spaces, not disappear. If they disappeared,
    # "well-being" would become "wellbeing" — one token instead of two,
    # silently changing what the model is scored against. This also protects
    # the required step ordering: hyphen-replacement runs after digit
    # expansion, because num2words emits hyphens ("twenty-three") that must
    # also turn into spaces rather than survive as literal hyphens.
    assert normalize("well-being") == "well being"


def test_digits_expand():
    # Digits must be spelled out as words, matching what was actually spoken.
    # A model trained on "1995" as a numeral but scored against "nineteen
    # ninety five" (or the reverse) would look wrong for a reason that has
    # nothing to do with the model itself — a text-representation bug, not a
    # model bug — and it needs to be caught here, not discovered later as an
    # unexplained bad WER.
    assert normalize("in 1995") == "in nineteen ninety five"


def test_idempotent():
    # Normalizing already-normalized text must be a no-op. If it isn't,
    # scoring code that re-normalizes a reference that arrived pre-normalized
    # would silently corrupt it further on a second pass, and training and
    # scoring could drift apart with no error raised on either side.
    for s in SAMPLES:
        assert normalize(normalize(s)) == normalize(s)


def test_output_charset():
    # Every character in the output must belong to the CTC output vocabulary
    # (lowercase a-z, apostrophe, space) and nothing else. The character set
    # IS the model's output layer — one stray character reaching training
    # data shifts every index after it and silently scrambles what the model
    # produces, with no error raised anywhere.
    ALLOWED = set("abcdefghijklmnopqrstuvwxyz' ")
    for s in SAMPLES:
        assert set(normalize(s)) <= ALLOWED


def test_no_empty_from_nonempty():
    # Real, non-empty transcript text must never normalize to an empty
    # string. An empty reference is a landmine for WER computation (jiwer and
    # evaluate disagree on how to score one), and it's the same class of bug
    # as the brief's `normalize_transcripts` anti-pattern — a missing
    # `return` silently turning every transcript into a no-op. Different
    # failure, same shape: looks fine, quietly isn't.
    assert normalize("Hello there.") != ""


# ---------------------------------------------------------------------------
# TODO — corpus-derived digit edge cases (years, ordinals, currency)
#
# The brief requires test cases for the num2words forms that actually appear
# in the transcripts, not hypothetical ones. That needs real transcript text,
# which doesn't exist yet — Stage A (inventory) hasn't run. Once
# v5_inventory.jsonl exists, pull real examples containing years, ordinals,
# and currency amounts and add tests here before treating normalize() as done.
# ---------------------------------------------------------------------------
