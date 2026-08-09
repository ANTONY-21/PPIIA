"""Text preparation tests for both engines (no models, no network)."""

import ppiia
import ppiia_ai


def test_prepare_text_removes_page_numbers_and_joins_hyphens():
    raw = "The Reorgani-\nsation of the State.\n 12 \nSection 5 applies."
    text, truncated = ppiia_ai.prepare_text(raw)
    assert "Reorganisation" in text
    assert "\n12\n" not in text and " 12 " not in text
    assert truncated is False


def test_prepare_text_truncates_when_over_limit(monkeypatch):
    monkeypatch.setattr(ppiia_ai, "MAX_INPUT_CHARS", 100)
    text, truncated = ppiia_ai.prepare_text("word " * 200)
    assert truncated is True
    assert len(text) <= 100


def test_build_source_info_counts():
    info = ppiia_ai.build_source_info("one two three", "text", "note.txt")
    assert info.word_count == 3
    assert info.char_count == 13
    assert info.source_type == "text"


def test_local_clean_text_collapses_whitespace():
    cleaned = ppiia.clean_text("A  bill\n\n 3 \nPage 2 of 9 text")
    assert "  " not in cleaned
    assert "Page 2 of 9" not in cleaned


def test_local_chunker_respects_word_budget():
    text = ". ".join(f"Sentence number {i} with several words" for i in range(200))
    chunks = ppiia.chunk_text(text, max_words=100)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 130 for c in chunks)  # small tolerance


def test_local_entity_extraction():
    # the ministry regex expects punctuation right after the name
    text = (
        "The Finance Act, 2020 amended earlier rules. "
        "Notices were issued by the Ministry of Home Affairs, in 2014."
    )
    ents = ppiia.extract_entities(text)
    assert "2014" in ents["years"] and "2020" in ents["years"]
    assert any("Ministry of Home Affairs" in m for m in ents["ministries"])
    assert any("Finance Act" in a for a in ents["acts_cited"])
