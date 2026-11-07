"""Checks for local analysis on documents beyond the bundled sample."""

import ppiia
from schemas import BillAnalysis


def test_local_report_does_not_invent_impact_or_controversy():
    report = BillAnalysis.from_local({
        "title": "Example Bill", "summary": "Example summary.",
        "sectors": ["Technology"], "impacts": {},
    })
    assert report.detected_sectors == ["Technology"]
    assert not report.sector_impacts
    assert report.controversy_level is None
    assert not report.term_impacts.short_term.effects


def test_title_can_span_parliamentary_pdf_lines():
    text = "AS INTRODUCED\nTHE CONSTITUTION (SCHEDULED TRIBES) ORDER\n(AMENDMENT) BILL, 2024\nA BILL"
    assert ppiia.guess_title(text) == "The Constitution (Scheduled Tribes) Order (Amendment) Bill, 2024"


def test_key_passage_identifies_the_amendment():
    source = 'Introduction. In entry 55 of the Schedule, for the words "old", the words "new" shall be substituted. A note follows.'
    passages = ppiia.extract_key_passages(source)
    assert len(passages) == 1
    assert 'the words "new" shall be substituted' in passages[0]


def test_summarization_passes_every_token_to_a_first_stage(monkeypatch):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()

        def decode(self, tokens, **kwargs):
            return " ".join(tokens)

    observed = []

    def fake_summary(text, *_args):
        observed.extend(text.split())
        return "short summary"

    monkeypatch.setattr(ppiia, "_load_hf_models", lambda: (Tokenizer(), None, None, None))
    monkeypatch.setattr(ppiia, "_summarize_chunk", fake_summary)
    words = [f"word{i}" for i in range(1801)]
    assert ppiia.summarize_long_text(" ".join(words))
    assert observed == words
