"""Schema validation, coercion, and local-report adaptation tests."""

import json

from schemas import (
    BillAnalysis,
    RelatedLegislation,
    RiskFlag,
    SectorImpact,
    SourceInfo,
    TimelineEvent,
)


def test_minimal_analysis_gets_defaults():
    analysis = BillAnalysis(title="Test Bill")
    assert analysis.category == "Other"
    assert analysis.positives == []
    assert analysis.term_impacts.short_term.effects == []
    assert analysis.controversy_level == "low"


def test_choice_fields_are_coerced_not_rejected():
    impact = SectorImpact(sector="Finance", direction=" Positive ", magnitude="HUGE")
    assert impact.direction == "positive"
    assert impact.magnitude == "medium"  # unknown value falls back to default

    flag = RiskFlag(topic="x", severity="CRITICAL")
    assert flag.severity == "medium"

    act = RelatedLegislation(name="Some Act", relationship="supersedes", confidence="High")
    assert act.relationship == "related"
    assert act.confidence == "high"


def test_timeline_year_accepts_int():
    ev = TimelineEvent(year=2014, label="Bifurcation")
    assert ev.year == "2014"


def test_json_round_trip():
    analysis = BillAnalysis(
        title="Round Trip Bill",
        sector_impacts=[SectorImpact(sector="Agriculture", direction="mixed")],
        source=SourceInfo(source_type="text", name="t", char_count=10, word_count=2),
    )
    restored = BillAnalysis.model_validate(json.loads(analysis.model_dump_json()))
    assert restored == analysis


def test_from_local_maps_flat_report():
    report = {
        "title": "Local Bill",
        "ministry": "Ministry of Testing",
        "objective": "Do things.",
        "summary": "A short summary.",
        "sectors": ["Finance", "Technology"],
        "chronology": ["Old Act, 2000"],
        "impacts": {"short_term": "Soon.", "medium_term": "", "long_term": "Later."},
        "positives": ["good"],
        "negatives": [],
        "risks": ["a risk"],
    }
    analysis = BillAnalysis.from_local(report)
    assert analysis.title == "Local Bill"
    assert analysis.category == "Finance"
    assert [s.sector for s in analysis.sector_impacts] == ["Finance", "Technology"]
    assert analysis.related_legislation[0].name == "Old Act, 2000"
    assert analysis.related_legislation[0].confidence == "high"
    assert analysis.term_impacts.short_term.effects == ["Soon."]
    assert analysis.term_impacts.medium_term.effects == []
    assert analysis.risk_flags[0].concern == "a risk"
