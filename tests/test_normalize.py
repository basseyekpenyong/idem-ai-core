from idem.normalize import normalize, ALLOWED


# ---------------------------------------------------------------------------
# Sample transcripts used across multiple tests below.
# Plain ASCII Nigerian English only -- no digits, accents, or curly
# punctuation, since those are exercised by their own targeted tests.
# ---------------------------------------------------------------------------

SAMPLES = [
    "The pain in my eye is so wicked.",
    "I don't like how this network is behaving today.",
    "She is coming from the market right now.",
    "We will meet at the junction by six o'clock.",
]


# ---------------------------------------------------------------------------
# Apostrophe handling
# ---------------------------------------------------------------------------

def test_curly_and_straight_apostrophe_agree():
    # Protects against the exact bug that caused an 11.7-point WER
    # discrepancy: a curly apostrophe and a straight one must produce
    # identical training text, not two different tokens for "don't".
    assert normalize("Don't") == normalize("Don’t") == "don't"


# ---------------------------------------------------------------------------
# Hyphen handling
# ---------------------------------------------------------------------------

def test_hyphen_becomes_space():
    # Protects the operation ordering (step 4 runs after step 3): hyphenated
    # compounds split into separate words rather than fusing into one
    # unrecognizable token, and any hyphen num2words emits during digit
    # expansion is also turned into a space rather than surviving into the
    # output.
    assert normalize("well-being") == "well being"


# ---------------------------------------------------------------------------
# Digit expansion
# ---------------------------------------------------------------------------

def test_digits_expand():
    # Protects the operation ordering (step 3 runs before step 4): digits
    # must be spelled out as words before hyphen handling runs, since
    # num2words emits hyphens (e.g. "ninety-five") that step 4 must still
    # catch.
    assert normalize("in 1995") == "in nineteen ninety five"


# ---------------------------------------------------------------------------
# Idempotence and output charset
# ---------------------------------------------------------------------------

def test_idempotent():
    # Protects against a normalizer that isn't stable under repeated
    # application -- if normalize(normalize(x)) != normalize(x), then
    # re-running a pipeline stage on already-normalized text would silently
    # change the data.
    for s in SAMPLES:
        assert normalize(normalize(s)) == normalize(s)


def test_output_charset():
    # Protects the CTC output layer: every character in normalized text must
    # be in the trained charset. One stray character here silently shifts
    # the model's index mapping and scrambles its output with no error
    # raised.
    for s in SAMPLES:
        assert set(normalize(s)) <= ALLOWED


# ---------------------------------------------------------------------------
# Non-degenerate output
# ---------------------------------------------------------------------------

def test_no_empty_from_nonempty():
    # Protects against a normalizer that strips real speech down to nothing
    # (e.g. over-aggressive punctuation or noise-token removal), which would
    # silently drop a training or scoring example.
    assert normalize("Hello there.") != ""
