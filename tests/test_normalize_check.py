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
    example = violations["asr-studio"]["examples"][0]
    assert example["id"] == "a2"
    assert example["text"] == "In 1995 the government changed the policy."


def test_example_carries_id_and_line_not_just_text():
    # A human reading "slr70: 412" with five bare sentences has no way back
    # to the actual records to go fix them -- Stage A gives every record an
    # id, and this scan must carry it through rather than throw it away.
    report = check_corpus(FIXTURE)
    example = report["empties"]["asr-studio"]["examples"][0]
    assert example.keys() == {"id", "line", "text"}
    assert example["id"] == "a6"
    assert example["line"] == 6  # "..." is the 6th line of the fixture


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
    assert entry["examples"]["asr-studio"][0]["id"] == "a5"


def test_records_with_digits_are_not_double_counted_as_oov():
    # A record already flagged as a §3a violation must not also show up in
    # the OOV bucket -- it's one problem, reported once, in the bucket a
    # human should act on.
    report = check_corpus(FIXTURE)
    all_oov_ids = {
        example["id"]
        for entry in report["oov"].values()
        for examples in entry["examples"].values()
        for example in examples
    }
    assert "a3" not in all_oov_ids  # "It cost $50 for the trip."
    assert "a4" not in all_oov_ids  # "She scored 50% on the test."


def test_non_empty_to_empty_conversion_detected():
    # "..." is non-empty raw text, but normalize() strips all punctuation and
    # collapses whitespace, leaving "". An empty reference is a landmine for
    # WER computation, so this must be caught and reported, not silently
    # dropped.
    report = check_corpus(FIXTURE)
    assert report["empties"]["asr-studio"]["count"] == 1
    assert report["empties"]["asr-studio"]["examples"][0]["id"] == "a6"


def test_clean_record_produces_no_findings():
    # "Well-being matters a lot." (id a7, slr70) has no digits, no OOV
    # characters, and doesn't normalize to empty -- it must not appear in
    # any bucket.
    report = check_corpus(FIXTURE)
    for bucket_name in ("violations", "empties"):
        for source_bucket in report[bucket_name].values():
            assert all(ex["id"] != "a7" for ex in source_bucket["examples"])
    for entry in report["oov"].values():
        for examples in entry["examples"].values():
            assert all(ex["id"] != "a7" for ex in examples)


def test_totals_and_overall_count():
    # Every record counts toward its source's total and the overall total,
    # regardless of which bucket (if any) it lands in. Without a
    # denominator, a bare count like "slr70: 412" can't be judged against
    # hard rule 3's "a filter dropping over 1% stops for human
    # investigation" -- there's nothing to compute the 1% of.
    report = check_corpus(FIXTURE)
    assert report["totals"] == {"asr-studio": 4, "slr70": 3}
    assert report["total_records"] == 7


def test_missing_required_field_raises(tmp_path):
    # Hard rule 2: fail loudly, never skip a record and continue. A record
    # missing "source" must stop the scan, not be silently omitted from
    # every count -- a corpus scan that can silently drop records could
    # report a clean corpus that isn't.
    bad_file = tmp_path / "missing_source.jsonl"
    bad_file.write_text('{"id": "x1", "text_raw": "Hello there."}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        check_corpus(str(bad_file))


def test_missing_id_raises(tmp_path):
    # "id" is required, not just used when present -- every example this
    # scan reports needs one to be actionable (see
    # test_example_carries_id_and_line_not_just_text).
    bad_file = tmp_path / "missing_id.jsonl"
    bad_file.write_text(
        '{"text_raw": "Hello there.", "source": "asr-studio"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="id"):
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
    assert "Scanned 7 records." in result.stdout
    assert "§3a convention violations" in result.stdout
    assert "Out-of-vocabulary characters" in result.stdout
    assert "Non-empty transcripts that normalize to empty" in result.stdout
    # Percentage against a denominator, and the id, not just bare text.
    assert "asr-studio: 1 / 4 (25.0%)" in result.stdout
    assert "[id=a2 line=2]" in result.stdout
