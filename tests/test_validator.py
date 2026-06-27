import unicodedata
import pytest
from engine.validator import validate


# ---------------------------------------------------------------------------
# Yoruba
# ---------------------------------------------------------------------------

class TestYoruba:
    LANG = "yo"

    def test_valid_basic(self):
        result = validate("Mo fẹ́ kọ èdè Yorùbá", self.LANG)
        assert result.is_valid, result.errors

    def test_valid_with_tone_marks(self):
        result = validate("Ẹ káàárọ̀ bàyé mi", self.LANG)
        assert result.is_valid, result.errors

    def test_output_is_nfc(self):
        # Input in NFD, output must be NFC
        nfd = unicodedata.normalize("NFD", "Ẹ káàárọ̀ bàyé mi")
        result = validate(nfd, self.LANG)
        assert unicodedata.is_normalized("NFC", result.normalized_text)

    def test_nfd_input_accepted(self):
        # NFD and NFC of the same text must both pass
        text = "Mo fẹ́ kọ èdè"
        nfd = unicodedata.normalize("NFD", text)
        nfc = unicodedata.normalize("NFC", text)
        assert validate(nfd, self.LANG).is_valid
        assert validate(nfc, self.LANG).is_valid

    def test_illegal_at_symbol(self):
        result = validate("Mo fẹ́ @ ilé", self.LANG)
        assert not result.is_valid
        assert any("Illegal" in e for e in result.errors)

    def test_illegal_digit(self):
        result = validate("Mo fẹ́ kọ 123", self.LANG)
        assert not result.is_valid

    def test_too_short_single_word(self):
        result = validate("Yorùbá", self.LANG)
        assert not result.is_valid
        assert any("short" in e.lower() for e in result.errors)

    def test_empty_string(self):
        result = validate("", self.LANG)
        assert not result.is_valid

    def test_whitespace_only(self):
        result = validate("   \t\n  ", self.LANG)
        assert not result.is_valid

    def test_normalized_text_stripped(self):
        result = validate("  Mo fẹ́ kọ  ", self.LANG)
        assert result.normalized_text == result.normalized_text.strip()


# ---------------------------------------------------------------------------
# Efik
# ---------------------------------------------------------------------------

class TestEfik:
    LANG = "efi"

    def test_valid_with_open_vowels(self):
        result = validate("Mme eyen ọkọ ama ikọ", self.LANG)
        assert result.is_valid, result.errors

    def test_open_o_accepted(self):
        result = validate("ɔkɔ mme eka edi ufɔk yak", self.LANG)
        assert result.is_valid, result.errors

    def test_open_e_accepted(self):
        result = validate("ɛded ɛkɔ mme eka yak", self.LANG)
        assert result.is_valid, result.errors

    def test_yoruba_specific_char_rejected(self):
        # ṣ (s-dot-below) is Yoruba, not Efik
        result = validate("mme ṣinṣin eka edi", self.LANG)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# Ibibio
# ---------------------------------------------------------------------------

class TestIbibio:
    LANG = "ibb"

    def test_valid_basic(self):
        result = validate("Mmiri dị mma maka ahụike", self.LANG)
        assert result.is_valid, result.errors

    def test_eng_nasal_accepted(self):
        result = validate("ŋ ŋkpa ŋkpa eka edi mma", self.LANG)
        assert result.is_valid, result.errors


# ---------------------------------------------------------------------------
# Unknown language code
# ---------------------------------------------------------------------------

def test_unknown_language_raises():
    with pytest.raises(KeyError, match="Unknown language code"):
        validate("some text here today", "xx")


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

def test_validate_batch():
    from engine.validator import validate_batch
    pairs = [
        ("Mo fẹ́ kọ èdè Yorùbá", "yo"),
        ("hello @ world bad", "yo"),
    ]
    results = validate_batch(pairs)
    assert results[0].is_valid
    assert not results[1].is_valid
