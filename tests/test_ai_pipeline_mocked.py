"""Four-stage Claude pipeline assembly, with the API mocked out."""

import pytest

import ppiia_ai
from schemas import (
    BillAnalysis,
    LLMChronology,
    LLMFAQ,
    LLMHorizon,
    LLMImpact,
    LLMMetadata,
    LLMPersona,
    LLMProvision,
    LLMRelatedAct,
    LLMRiskFlag,
    LLMSectorImpact,
    LLMSummary,
    LLMTimelineEvent,
)

FAKE_STAGES = {
    LLMMetadata: LLMMetadata(
        title="The Testing Bill, 2026",
        ministry="Ministry of Testing",
        category="Technology",
        secondary_categories=["Governance"],
        objective="Regulate testing.",
        one_line_essence="Testing becomes law.",
        key_provisions=[
            LLMProvision(title="Provision A", description="Does A.", clause_reference="Clause 2")
        ],
        stakeholders=["Testers"],
    ),
    LLMChronology: LLMChronology(
        policy_background="Testing was unregulated.",
        related_legislation=[
            LLMRelatedAct(
                name="Old Testing Act, 1999",
                year=1999,
                relationship="predecessor",
                note="Cited in text.",
                confidence="high",
            )
        ],
        timeline=[
            LLMTimelineEvent(year="1999", label="Old act", description="First rules."),
            LLMTimelineEvent(year="2026", label="This bill", description="Modern rules."),
        ],
    ),
    LLMImpact: LLMImpact(
        sector_impacts=[
            LLMSectorImpact(
                sector="Technology",
                direction="positive",
                magnitude="high",
                rationale="Helps tech.",
                opportunities=["New tools"],
                restrictions=["Audits"],
            )
        ],
        short_term=LLMHorizon(effects=["Quick effect"], risks=["Quick risk"]),
        medium_term=LLMHorizon(effects=["Middle effect"], risks=[]),
        long_term=LLMHorizon(effects=["Late effect"], risks=[]),
        positives=["Good thing"],
        negatives=["Bad thing"],
        risk_flags=[
            LLMRiskFlag(
                topic="Cost", concern="Might cost money.", severity="medium",
                affected_groups=["Startups"],
            )
        ],
        controversy_level="low",
    ),
    LLMSummary: LLMSummary(
        plain_summary="This bill makes testing simple.\nEveryone can understand it now.\n"
        "It starts next year.\nNothing costs extra money.\nSchools will teach it.\n"
        "Companies must follow easy rules.\nPeople get clear reports.\nHelp lines will open.\n"
        "Old rules go away.\nThe change helps everyone.",
        citizen_faq=[LLMFAQ(question="Is it free?", answer="Yes.")],
        persona_impacts=[LLMPersona(persona="A student", impact="Learns testing.")],
    ),
}


class FakeEngine:
    model = "fake-model"

    def __init__(self):
        self.calls = []

    def run_stage(self, bill_text, instruction, output_model):
        self.calls.append((instruction[:40], output_model.__name__))
        return FAKE_STAGES[output_model]


def test_pipeline_assembles_full_analysis():
    fake = FakeEngine()
    text = "Section 1. The Testing Bill. " * 30  # > 200 chars
    analysis = ppiia_ai.analyze_with_claude(text, engine=fake)

    assert isinstance(analysis, BillAnalysis)
    assert analysis.title == "The Testing Bill, 2026"
    assert analysis.category == "Technology"
    assert analysis.sector_impacts[0].magnitude == "high"
    assert analysis.term_impacts.medium_term.effects == ["Middle effect"]
    assert analysis.risk_flags[0].affected_groups == ["Startups"]
    assert analysis.citizen_faq[0].question == "Is it free?"
    assert analysis.generated_at is not None
    # all four stages ran, in order
    assert [name for _, name in fake.calls] == [
        "LLMMetadata", "LLMChronology", "LLMImpact", "LLMSummary",
    ]


def test_pipeline_computes_metrics():
    analysis = ppiia_ai.analyze_with_claude("Sample bill text. " * 30, engine=FakeEngine())
    m = analysis.metrics
    assert m is not None
    assert m.engine == "claude"
    assert m.model == "fake-model"
    assert m.processing_seconds is not None
    assert m.readability_grade is not None  # summary is long enough to score
    assert m.summary_words and m.input_words


def test_pipeline_rejects_tiny_documents():
    with pytest.raises(ppiia_ai.AnalysisError, match="too short"):
        ppiia_ai.analyze_with_claude("tiny", engine=FakeEngine())


def test_progress_callback_receives_stage_updates():
    seen = []
    ppiia_ai.analyze_with_claude(
        "Sample bill text. " * 30, engine=FakeEngine(), progress=seen.append
    )
    assert any("Stage 1/4" in msg for msg in seen)
    assert any("Stage 4/4" in msg for msg in seen)


def test_engine_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ppiia_ai.AnalysisError, match="API key"):
        ppiia_ai.ClaudeEngine(api_key=None)
