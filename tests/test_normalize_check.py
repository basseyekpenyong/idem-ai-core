"""
Tests for the idem.normalize --check corpus scan (Stage C, brief §6).

Per the brief: "The --check harness itself does not wait on Stage A. Write it
and test it against a small hand-made fixture file now." tests/fixtures/
check_sample.jsonl is that fixture — it does not come from a real inventory.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from idem.normalize import check_corpus

FIXTURE = str(Path(__file__).parent / "fixtures" / "check_sample.jsonl")


def test_convention_violations_grouped_by_source():
    # "In 1995..." (asr-studio), "$50" and "50%" (slr70) each contain a
    # digit, currency symbol, or percent sign, which per §3a means the
    # transcript itself broke the "record what was spoken" convention. This
    # must be counted even though normalize() still silently expands these
    # today (step 3 hasn't been revised to raise yet) -- the brief is
    # explicit that expanding a numeral must never make this count go down.
    report = check_corpus(FIXTURE)
    violations = report["violations"]
    assert violations["asr-studio"]["count"] == 1
    assert violations["slr70"]["count"] == 2
    assert "In 1995 the government changed the policy." in violations["asr-studio"]["examples"]


def test_oov_character_found_and_grouped_by_source():
    # "café" survives normalize() unchanged apart from casing -- "é" is a
    # letter, not punctuation, so step 5 doesn't remove it, and it's not in
    # ALLOWED. This is exactly the é/ẹ/ọ question the brief flags as still
    # open (§3a): the scan's job is to surface it, not resolve it.
    report = check_corpus(FIXTURE)
    assert "é" in report["oov"]
    entry = report["oov"]["é"]
    assert entry["count"] == 1
    assert entry["by_source"] == {"asr-studio": 1}


def test_records_with_digits_are_not_double_counted_as_oov():
    # A record already flagged as a §3a violation must not also show up in
    # the OOV bucket -- it's one problem, reported once, in the bucket a
    # human should act on.
    report = check_corpus(FIXTURE)
    all_oov_examples = {
        example
        for entry in report["oov"].values()
        for examples in entry["examples"].values()
        for example in examples
    }
    assert "It cost $50 for the trip." not in all_oov_examples
    assert "She scored 50% on the test." not in all_oov_examples


def test_non_empty_to_empty_conversion_detected():
    # "..." is non-empty raw text, but normalize() strips all punctuation and
    # collapses whitespace, leaving "". An empty reference is a landmine for
    # WER computation, so this must be caught and reported, not silently
    # dropped.
    report = check_corpus(FIXTURE)
    assert report["empties"]["asr-studio"]["count"] == 1
    assert "..." in report["empties"]["asr-studio"]["examples"]


def test_clean_record_produces_no_findings():
    # "Well-being matters a lot." (slr70) has no digits, no OOV characters,
    # and doesn't normalize to empty -- it must not appear in any bucket.
    report = check_corpus(FIXTURE)
    clean = "Well-being matters a lot."
    for bucket_name in ("violations", "empties"):
        for source_bucket in report[bucket_name].values():
            assert clean not in source_bucket["examples"]
    for entry in report["oov"].values():
        for examples in entry["examples"].values():
            assert clean not in examples


def test_missing_required_field_raises(tmp_path):
    # Hard rule 2: fail loudly, never skip a record and continue. A record
    # missing "source" must stop the scan, not be silently omitted from
    # every count -- a corpus scan that can silently drop records could
    # report a clean corpus that isn't.
    bad_file = tmp_path / "missing_source.jsonl"
    bad_file.write_text('{"text_raw": "Hello there."}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        check_corpus(str(bad_file))


def test_invalid_json_line_raises(tmp_path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        check_corpus(str(bad_file))


def test_cli_runs_against_fixture():
    # Smoke test for `python -m idem.normalize --check <file>` exactly as
    # documented in the module docstring and the brief's launcher example.
    result = subprocess.run(
        [sys.executable, "-m", "idem.normalize", "--check", FIXTURE],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "§3a convention violations" in result.stdout
    assert "Out-of-vocabulary characters" in result.stdout
    assert "Non-empty transcripts that normalize to empty" in result.stdout
