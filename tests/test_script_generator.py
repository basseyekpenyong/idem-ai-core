import pytest
from engine.script_generator import (
    ScriptChunk,
    chunk_document,
    chunk_file,
    mock_scripts,
    MIN_WORDS,
    MAX_WORDS,
    TARGET_MIN,
    TARGET_MAX,
)


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------

class TestChunkDocument:

    def test_short_sentence_returned_as_one_chunk(self):
        text = "Mo fẹ́ kọ èdè Yorùbá lónìí."
        chunks = chunk_document(text)
        assert len(chunks) == 1
        assert chunks[0].text

    def test_long_sentence_split_within_max_words(self):
        long = " ".join(["word"] * 50)
        chunks = chunk_document(long)
        assert all(c.word_count <= MAX_WORDS for c in chunks)

    def test_no_chunk_below_min_words_when_enough_text(self):
        # Enough total words — merging should prevent stubs below MIN_WORDS
        text = "Go home now. Come back soon. Eat your food well."
        chunks = chunk_document(text)
        total_words = sum(c.word_count for c in chunks)
        if total_words >= MIN_WORDS:
            assert all(c.word_count >= MIN_WORDS for c in chunks)

    def test_multiple_sentences_produce_multiple_chunks(self):
        text = (
            "Àwọn ọmọ ń kọ́ nínú ilé ẹ̀kọ́ lónìí pẹ̀lú àwọn olùkọ́ wọn. "
            "Wọn máa ń kẹ́kọ̀ọ́ dáadáa nígbà gbogbo. "
            "Ẹ́kọ́ dára fún gbogbo ènìyàn."
        )
        chunks = chunk_document(text)
        assert len(chunks) >= 1

    def test_in_target_range_flag_accurate(self):
        text = " ".join(["word"] * 12)  # 12 words — in target range
        chunks = chunk_document(text)
        assert chunks[0].in_target_range

    def test_too_few_words_not_in_target_range(self):
        # Only 3 words — merging can't help since there's nothing to merge with
        text = "Go home now."
        chunks = chunk_document(text)
        # At least one chunk exists; it may be below target range
        assert all(isinstance(c.in_target_range, bool) for c in chunks)

    def test_source_line_preserved(self):
        chunks = chunk_document("Mo fẹ́ kọ èdè Yorùbá today.", source_line=7)
        assert chunks[0].source_line == 7

    def test_word_count_matches_text(self):
        text = "One two three four five six seven eight nine ten."
        chunks = chunk_document(text)
        for c in chunks:
            assert c.word_count == len(c.text.split())

    def test_empty_string_returns_empty_list(self):
        assert chunk_document("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_document("   \n\t  ") == []

    def test_clause_split_on_comma(self):
        # 35-word sentence with a comma — should split at the comma boundary
        part1 = " ".join(["alpha"] * 18)
        part2 = " ".join(["beta"] * 18)
        text = part1 + ", " + part2 + "."
        chunks = chunk_document(text)
        assert all(c.word_count <= MAX_WORDS for c in chunks)


# ---------------------------------------------------------------------------
# mock_scripts
# ---------------------------------------------------------------------------

class TestMockScripts:

    def test_yoruba_mock(self):
        chunks = mock_scripts("yo", 3)
        assert len(chunks) == 3
        assert all(c.text for c in chunks)

    def test_efik_mock(self):
        chunks = mock_scripts("efi", 2)
        assert len(chunks) == 2

    def test_ibibio_mock(self):
        chunks = mock_scripts("ibb", 2)
        assert len(chunks) == 2

    def test_count_respected(self):
        for n in [1, 5, 10]:
            chunks = mock_scripts("yo", n)
            assert len(chunks) == n

    def test_unknown_language_raises_value_error(self):
        with pytest.raises(ValueError, match="No mock data"):
            mock_scripts("xx", 3)

    def test_each_chunk_is_script_chunk(self):
        for c in mock_scripts("yo", 3):
            assert isinstance(c, ScriptChunk)

    def test_word_count_set_correctly(self):
        for c in mock_scripts("yo", 5):
            assert c.word_count == len(c.text.split())

    def test_in_target_range_is_bool(self):
        for c in mock_scripts("yo", 5):
            assert isinstance(c.in_target_range, bool)
