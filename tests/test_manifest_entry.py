import json
import pytest
from schemas.manifest_entry import ManifestEntry


def _entry(**overrides) -> ManifestEntry:
    defaults = dict(
        audio_filepath="/data/processed/yo_abc123def45678.wav",
        text="Ẹ káàárọ̀ bàyé",
        duration=3.5,
        language="yo",
        speaker_id="spk_001",
        speaker_gender="F",
        speaker_age_range="18-30",
        dialect="",
        sample_rate=16000,
        hash_id="abc123def45678ab",
        split="train",
        quality_snr_db=25.0,
        quality_clipping=False,
        created_at="2026-06-27T00:00:00+00:00",
    )
    return ManifestEntry(**{**defaults, **overrides})


# --- Serialisation ---

def test_roundtrip_jsonl():
    entry = _entry()
    assert ManifestEntry.from_jsonl(entry.to_jsonl()) == entry


def test_jsonl_is_single_line():
    line = _entry().to_jsonl()
    assert "\n" not in line


def test_jsonl_is_valid_json():
    parsed = json.loads(_entry().to_jsonl())
    assert parsed["language"] == "yo"
    assert parsed["sample_rate"] == 16000


def test_unicode_preserved_through_jsonl():
    entry = _entry(text="Ẹ káàárọ̀ — ọmọ ìlú")
    restored = ManifestEntry.from_jsonl(entry.to_jsonl())
    assert restored.text == entry.text


def test_from_jsonl_ignores_trailing_whitespace():
    line = _entry().to_jsonl() + "   \n"
    ManifestEntry.from_jsonl(line)  # should not raise


# --- Quality flags ---

def test_is_clean_passes_good_recording():
    assert _entry(quality_snr_db=25.0, quality_clipping=False).is_clean()


def test_is_clean_fails_on_clipping():
    assert not _entry(quality_clipping=True).is_clean()


def test_is_clean_fails_on_low_snr():
    assert not _entry(quality_snr_db=10.0).is_clean()


def test_is_clean_custom_threshold():
    entry = _entry(quality_snr_db=18.0, quality_clipping=False)
    assert entry.is_clean(min_snr_db=15.0)
    assert not entry.is_clean(min_snr_db=20.0)


# --- Metadata validation ---

def test_validate_metadata_valid_entry():
    assert _entry().validate_metadata() == []


@pytest.mark.parametrize("split", ["validation", "holdout", ""])
def test_validate_metadata_rejects_bad_split(split):
    errors = _entry(split=split).validate_metadata()
    assert any("split" in e for e in errors)


@pytest.mark.parametrize("gender", ["X", "male", "female", ""])
def test_validate_metadata_rejects_bad_gender(gender):
    errors = _entry(speaker_gender=gender).validate_metadata()
    assert any("gender" in e for e in errors)


@pytest.mark.parametrize("age", ["25", "young", ""])
def test_validate_metadata_rejects_bad_age_range(age):
    errors = _entry(speaker_age_range=age).validate_metadata()
    assert any("age_range" in e for e in errors)


def test_validate_metadata_rejects_wrong_sample_rate():
    errors = _entry(sample_rate=44100).validate_metadata()
    assert any("sample_rate" in e for e in errors)


def test_validate_metadata_rejects_zero_duration():
    errors = _entry(duration=0.0).validate_metadata()
    assert any("duration" in e for e in errors)
