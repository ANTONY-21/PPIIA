"""PDF report generation from the demo analysis."""

import ppiia_ai
from report_pdf import build_pdf_report
from schemas import BillAnalysis


def test_pdf_from_demo_analysis():
    analysis = ppiia_ai.load_sample_analysis()
    pdf_bytes = build_pdf_report(analysis)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 2000


def test_pdf_survives_unicode_and_sparse_analysis():
    analysis = BillAnalysis(
        title="Bill with ₹ and — dashes and “quotes”",
        plain_summary="Costs ₹500 — that's all.",
    )
    pdf_bytes = build_pdf_report(analysis)
    assert pdf_bytes.startswith(b"%PDF")
