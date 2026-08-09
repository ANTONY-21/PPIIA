"""The bundled demo artifacts stay valid against the schema."""

import json

import ppiia_ai
from schemas import BillAnalysis


def test_sample_pdf_exists():
    assert ppiia_ai.SAMPLE_PDF_PATH.exists()


def test_sample_analysis_validates_against_schema():
    data = json.loads(ppiia_ai.SAMPLE_ANALYSIS_PATH.read_text(encoding="utf-8"))
    analysis = BillAnalysis.model_validate(data)
    assert "Andhra Pradesh" in analysis.title
    assert analysis.sector_impacts, "demo should include sector impacts"
    assert analysis.timeline, "demo should include a timeline"
    assert analysis.citizen_faq, "demo should include an FAQ"


def test_load_sample_analysis_fills_metrics():
    analysis = ppiia_ai.load_sample_analysis()
    assert analysis.metrics is not None
    assert analysis.metrics.engine == "precomputed demo"
    assert analysis.metrics.readability_grade is not None
