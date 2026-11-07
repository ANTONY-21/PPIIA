"""Behavioral checks for source-grounded local summaries and topic suggestions."""

import evidence
import ppiia


TRIBAL_BILL = """THE CONSTITUTION (SCHEDULED TRIBES) ORDER (AMENDMENT) BILL, 2024
A
BILL
further to amend the Constitution (Scheduled Tribes) Order, 1950.
BE it enacted by Parliament as follows:
1. This Act may be called the Constitution (Scheduled Tribes) Order Amendment Act.
2. In entry 55 of the Schedule, for the words "Jhodia Paroja", the words
"Jhodia, Jhodia Paroja" shall be substituted.
"""


def test_summary_contains_the_source_provision_and_no_invented_forecast(monkeypatch):
    monkeypatch.setattr(ppiia, "summarize_long_text", lambda _: (_ for _ in ()).throw(AssertionError("AI draft should be optional")))
    report = ppiia.analyze(TRIBAL_BILL)
    assert '"Jhodia, Jhodia Paroja"' in report.summary
    assert "shall be substituted" in report.summary
    assert report.impacts == {}
    assert report.ai_draft == ""
    assert [item["sector"] for item in report.topic_evidence] == ["Social welfare and tribal affairs"]
    assert all(item["matched_phrase"].lower() in item["excerpt"].lower() for item in report.topic_evidence)


def test_relevant_data_topic_does_not_turn_an_illustration_into_finance():
    text = """THE DIGITAL PERSONAL DATA PROTECTION BILL, 2023
A BILL to provide for the processing of digital personal data.
BE it enacted by Parliament as follows:
4. (1) Personal data may be processed for a lawful purpose with consent.
Illustration.
X uses a banking regulation service and a banking regulation example.
(2) The Data Fiduciary shall give notice before processing personal data.
"""
    matches = evidence.identify_topics(text, "Digital Personal Data Protection Bill", evidence.stated_purpose(text))
    labels = [item.sector for item in matches]
    assert "Data and technology" in labels
    assert "Finance and insurance" not in labels
    assert all("banking regulation" not in item.excerpt.lower() for item in matches)


def test_unclear_bill_abstains_instead_of_guessing():
    text = "An unspecified proposal is discussed. " * 12
    assert evidence.identify_topics(text, "Untitled", "") == []
    assert "No sufficiently clear" in evidence.build_source_summary(text, [])
    assert ppiia.guess_title(text) == "Untitled Bill"


def test_evidence_after_first_500_words_is_considered():
    text = "Opening procedural wording. " * 600 + " Personal data must be protected. Personal data is subject to consent."
    matches = evidence.identify_topics(text, "Untitled", "")
    assert any(item.sector == "Data and technology" for item in matches)


def test_optional_ai_draft_is_separate_from_source_summary(monkeypatch):
    monkeypatch.setattr(ppiia, "summarize_long_text", lambda _: "An optional model draft.")
    report = ppiia.analyze(TRIBAL_BILL, use_ai_draft=True)
    assert report.ai_draft == "An optional model draft."
    assert "shall be substituted" in report.summary
    assert "optional model draft" not in report.summary
