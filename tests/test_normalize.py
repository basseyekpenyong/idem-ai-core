"""
Tests for idem.normalize — the single normalizer shared by training and scoring.

No implementation exists yet (Stage C, v5 pipeline brief). These tests are
written first so the implementation has something concrete to satisfy.
"""

import pytest

from idem.normalize import ConventionViolation, normalize

# Placeholder sample sentences for the idempotence/charset checks below.
# These are illustrative, not drawn from the real corpus — Stage A (inventory)
# hasn't run yet. Extend or replace with real transcript lines once it has.
# None of these contain a digit, currency symbol, or percent sign — per §3a,
# normalize() raises on those, so a sample used for a generic "normalize()
# succeeds and produces well-formed output" check can't contain one.
SAMPLES = [
    "Hello there.",
    "Don't worry, we go see am tomorrow.",
    "It's a well-being programme for many families.",
    'She said, "no wahala" and left.',
    "The government changed the policy that year.",
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


def test_prime_mark_apostrophe_agrees():
    # A prime mark (′, U+2032) is sometimes used as a stand-in apostrophe by
    # transcription tools, visually close enough to ' that it's easy to miss.
    # Before this fix it wasn't in _LOOKALIKES, so it fell through to step 5
    # (punctuation removal) as ordinary punctuation and was deleted outright
    # rather than kept as a straight apostrophe — splitting "don′t" into two
    # words, "don t", instead of normalizing to "don't" like every other
    # apostrophe form. Same invariant as the curly-apostrophe test above, for
    # a different Unicode code point.
    assert normalize("don′t") == normalize("don't") == "don't"


def test_hyphen_becomes_space():
    # Hyphens must become spaces, not disappear. If they disappeared,
    # "well-being" would become "wellbeing" — one token instead of two,
    # silently changing what the model is scored against.
    assert normalize("well-being") == "well being"


def test_digits_raise():
    # A numeral means the transcript broke the §3a convention: it recorded
    # the written form, not what was spoken. normalize() never sees the
    # audio, so it can't know whether "1995" was said as a year, and
    # expanding it anyway would be a silent guess — the same class of bug as
    # the divergent-cleaning discrepancy this rebuild exists to fix, just
    # moved into the guess itself. It must refuse rather than guess.
    with pytest.raises(ConventionViolation):
        normalize("in 1995")


def test_currency_symbol_raises():
    # "$50" is the written form of several possible spoken readings ("fifty
    # dollars", "fifty bucks"); normalize() can't tell which was said, so a
    # currency symbol must raise for the same reason a bare digit does.
    with pytest.raises(ConventionViolation):
        normalize("it cost $50")


def test_percent_sign_raises():
    with pytest.raises(ConventionViolation):
        normalize("50% of the class")


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


def test_comma_without_whitespace_becomes_space():
    # A comma directly between two words with no surrounding whitespace must
    # become a space, not disappear — otherwise the words on either side
    # fuse into one unreadable token. This used to be tested with a
    # comma-grouped number ("3,500" -> "three thousand five hundred"), found
    # against a real transcript where the old digit regex matched "3" and
    # "500" separately and the comma between them vanished, producing
    # "threefive hundred". Digits now raise before reaching this step
    # (§3a, rev 5), so the same word-gluing invariant is tested here against
    # non-digit text instead of deleting the test.
    assert normalize("yes,no") == "yes no"


def test_punctuation_removal_does_not_glue_words():
    # Punctuation with no surrounding whitespace (a colon, here) must become
    # a space, not disappear — otherwise the words on either side fuse into
    # one unreadable token. This used to be tested with a decimal point
    # ("12.5" -> "twelve five"), found against a real transcript
    # ("...weighs 12.5 kilograms...") that produced "twelvefive kilograms"
    # before that fix. Digits now raise before reaching this step (§3a,
    # rev 5), so the invariant is tested here against non-digit text instead
    # of deleting the test.
    assert normalize("wait:go") == "wait go"
