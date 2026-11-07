"""
PPIIA — Structured analysis schema
==================================
Pydantic models for the rich bill analysis produced by the Claude engine
(`ppiia_ai.py`) and rendered by the dashboard (`app.py`).

Two tiers of models live here:

1. Storage models (`BillAnalysis` and friends) — lenient, with defaults and
   value coercion, used everywhere in the app and for the JSON report.
2. LLM stage payloads (`LLM*` models) — strict, every field required, used as
   structured-output schemas for the four Claude analysis stages.

`BillAnalysis.from_local()` adapts the local Hugging Face engine's
`BillReport` dict (see `ppiia.py`) into the same schema, so the dashboard has
a single rendering path for both engines.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

Direction = Literal["positive", "negative", "mixed", "neutral"]
Magnitude = Literal["low", "medium", "high"]
Severity = Literal["low", "medium", "high"]
Relationship = Literal["predecessor", "amends", "repeals", "implements", "related"]
Confidence = Literal["high", "medium", "low"]


def _norm_choice(value: object, allowed: tuple[str, ...], default: str) -> str:
    """Coerce an LLM/user-supplied label onto a closed vocabulary."""
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in allowed:
            return cleaned
    return default


# ---------------------------------------------------------------------------
# Storage models (lenient — defaults + coercion)
# ---------------------------------------------------------------------------
class KeyProvision(BaseModel):
    title: str
    description: str = ""
    clause_reference: str | None = None


class RelatedLegislation(BaseModel):
    name: str
    year: int | None = None
    relationship: Relationship = "related"
    note: str = ""
    confidence: Confidence = "medium"

    @field_validator("relationship", mode="before")
    @classmethod
    def _v_rel(cls, v: object) -> str:
        return _norm_choice(v, ("predecessor", "amends", "repeals", "implements", "related"), "related")

    @field_validator("confidence", mode="before")
    @classmethod
    def _v_conf(cls, v: object) -> str:
        return _norm_choice(v, ("high", "medium", "low"), "medium")


class TimelineEvent(BaseModel):
    year: str
    label: str
    description: str = ""

    @field_validator("year", mode="before")
    @classmethod
    def _v_year(cls, v: object) -> str:
        return str(v).strip()


class SectorImpact(BaseModel):
    sector: str
    direction: Direction = "mixed"
    magnitude: Magnitude = "medium"
    rationale: str = ""
    opportunities: list[str] = []
    restrictions: list[str] = []

    @field_validator("direction", mode="before")
    @classmethod
    def _v_dir(cls, v: object) -> str:
        return _norm_choice(v, ("positive", "negative", "mixed", "neutral"), "mixed")

    @field_validator("magnitude", mode="before")
    @classmethod
    def _v_mag(cls, v: object) -> str:
        return _norm_choice(v, ("low", "medium", "high"), "medium")


class TopicEvidence(BaseModel):
    sector: str
    matched_phrase: str
    excerpt: str


class HorizonImpact(BaseModel):
    effects: list[str] = []
    risks: list[str] = []


class TermImpacts(BaseModel):
    short_term: HorizonImpact = HorizonImpact()
    medium_term: HorizonImpact = HorizonImpact()
    long_term: HorizonImpact = HorizonImpact()


class RiskFlag(BaseModel):
    topic: str
    concern: str = ""
    severity: Severity = "medium"
    affected_groups: list[str] = []

    @field_validator("severity", mode="before")
    @classmethod
    def _v_sev(cls, v: object) -> str:
        return _norm_choice(v, ("low", "medium", "high"), "medium")


class CitizenFAQ(BaseModel):
    question: str
    answer: str


class PersonaImpact(BaseModel):
    persona: str
    impact: str


class SourceInfo(BaseModel):
    source_type: Literal["pdf", "url", "text", "sample"] = "text"
    name: str = "Untitled document"
    char_count: int = 0
    word_count: int = 0
    truncated: bool = False


class AnalysisMetrics(BaseModel):
    engine: str | None = None
    model: str | None = None
    processing_seconds: float | None = None
    flesch_reading_ease: float | None = None
    readability_grade: float | None = None
    input_words: int | None = None
    summary_words: int | None = None
    compression_ratio: float | None = None


class BillAnalysis(BaseModel):
    """The full citizen-friendly analysis of one bill."""

    title: str
    ministry: str | None = None
    category: str = "Other"
    secondary_categories: list[str] = []
    objective: str = ""
    one_line_essence: str = ""
    plain_summary: str = ""
    key_provisions: list[KeyProvision] = []
    stakeholders: list[str] = []
    policy_background: str | None = None
    related_legislation: list[RelatedLegislation] = []
    timeline: list[TimelineEvent] = []
    sector_impacts: list[SectorImpact] = []
    detected_sectors: list[str] = []
    topic_evidence: list[TopicEvidence] = []
    ai_draft: str = ""
    term_impacts: TermImpacts = TermImpacts()
    positives: list[str] = []
    negatives: list[str] = []
    risk_flags: list[RiskFlag] = []
    controversy_level: Severity | None = "low"
    citizen_faq: list[CitizenFAQ] = []
    persona_impacts: list[PersonaImpact] = []
    source: SourceInfo | None = None
    metrics: AnalysisMetrics | None = None
    generated_at: str | None = None

    @field_validator("controversy_level", mode="before")
    @classmethod
    def _v_controversy(cls, v: object) -> str | None:
        if v is None:
            return None
        return _norm_choice(v, ("low", "medium", "high"), "low")

    @classmethod
    def from_local(cls, report: dict, source: SourceInfo | None = None) -> "BillAnalysis":
        """Adapt the local engine's ``BillReport.to_dict()`` output.

        The local Hugging Face pipeline produces a flatter, shallower report;
        this maps it onto the rich schema so one dashboard renders both.
        """
        impacts = report.get("impacts") or {}

        def _horizon(key: str) -> HorizonImpact:
            text = (impacts.get(key) or "").strip()
            return HorizonImpact(effects=[text] if text else [])

        sectors = report.get("sectors") or []
        return cls(
            title=report.get("title") or "Bill analysis",
            ministry=report.get("ministry") or None,
            category="Not assessed",
            objective=report.get("objective") or "",
            plain_summary=report.get("summary") or "",
            key_provisions=[KeyProvision(title="Source excerpt", description=passage)
                            for passage in (report.get("source_passages") or [])],
            detected_sectors=sectors,
            topic_evidence=report.get("topic_evidence") or [],
            ai_draft=report.get("ai_draft") or "",
            controversy_level=None,
            related_legislation=[
                RelatedLegislation(name=act, note="Possible mention found by a text pattern; verify in the source.", confidence="medium")
                for act in (report.get("chronology") or [])[:15]
            ],
            term_impacts=TermImpacts(
                short_term=_horizon("short_term"),
                medium_term=_horizon("medium_term"),
                long_term=_horizon("long_term"),
            ),
            positives=report.get("positives") or [],
            negatives=report.get("negatives") or [],
            risk_flags=[
                RiskFlag(topic="Flagged by local model", concern=r) for r in (report.get("risks") or [])
            ],
            source=source,
        )


# ---------------------------------------------------------------------------
# LLM stage payloads (strict — used as structured-output schemas)
# ---------------------------------------------------------------------------
class LLMProvision(BaseModel):
    title: str
    description: str
    clause_reference: str | None


class LLMMetadata(BaseModel):
    title: str
    ministry: str | None
    category: str
    secondary_categories: list[str]
    objective: str
    one_line_essence: str
    key_provisions: list[LLMProvision]
    stakeholders: list[str]


class LLMRelatedAct(BaseModel):
    name: str
    year: int | None
    relationship: Literal["predecessor", "amends", "repeals", "implements", "related"]
    note: str
    confidence: Literal["high", "medium", "low"]


class LLMTimelineEvent(BaseModel):
    year: str
    label: str
    description: str


class LLMChronology(BaseModel):
    policy_background: str
    related_legislation: list[LLMRelatedAct]
    timeline: list[LLMTimelineEvent]


class LLMSectorImpact(BaseModel):
    sector: str
    direction: Literal["positive", "negative", "mixed", "neutral"]
    magnitude: Literal["low", "medium", "high"]
    rationale: str
    opportunities: list[str]
    restrictions: list[str]


class LLMHorizon(BaseModel):
    effects: list[str]
    risks: list[str]


class LLMRiskFlag(BaseModel):
    topic: str
    concern: str
    severity: Literal["low", "medium", "high"]
    affected_groups: list[str]


class LLMImpact(BaseModel):
    sector_impacts: list[LLMSectorImpact]
    short_term: LLMHorizon
    medium_term: LLMHorizon
    long_term: LLMHorizon
    positives: list[str]
    negatives: list[str]
    risk_flags: list[LLMRiskFlag]
    controversy_level: Literal["low", "medium", "high"]


class LLMFAQ(BaseModel):
    question: str
    answer: str


class LLMPersona(BaseModel):
    persona: str
    impact: str


class LLMSummary(BaseModel):
    plain_summary: str
    citizen_faq: list[LLMFAQ]
    persona_impacts: list[LLMPersona]
