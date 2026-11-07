"""
PPIIA — PDF report generator
============================
Renders a `BillAnalysis` into a downloadable PDF briefing using fpdf2's
built-in Helvetica fonts. Built-in fonts are latin-1 only, so text is
transliterated (₹ -> "Rs.", smart quotes -> plain quotes) before rendering.
"""

from __future__ import annotations

from datetime import datetime

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from schemas import BillAnalysis

# multi_cell must return the cursor to the left margin, else the next
# full-width cell starts at the right edge and fpdf2 raises
_WRAP = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}

_REPLACEMENTS = {
    "₹": "Rs. ",
    "—": "-",
    "–": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "•": "-",
    " ": " ",
}


def _s(text: object) -> str:
    out = str(text if text is not None else "")
    for src, dst in _REPLACEMENTS.items():
        out = out.replace(src, dst)
    return out.encode("latin-1", "replace").decode("latin-1")


class _ReportPDF(FPDF):
    def header(self) -> None:
        self.set_font("helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, "PPIIA - Public Policy Insight & Impact Analyzer", align="R")
        self.ln(8)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"Page {self.page_no()} - AI-generated analysis, verify with official sources", align="C")


def build_pdf_report(analysis: BillAnalysis) -> bytes:
    pdf = _ReportPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    def h1(text: str) -> None:
        pdf.set_text_color(20, 40, 80)
        pdf.set_font("helvetica", "B", 16)
        pdf.multi_cell(0, 8, _s(text), **_WRAP)
        pdf.ln(1)

    def h2(text: str) -> None:
        pdf.ln(2)
        pdf.set_text_color(20, 40, 80)
        pdf.set_font("helvetica", "B", 12)
        pdf.multi_cell(0, 7, _s(text), **_WRAP)
        pdf.set_text_color(0, 0, 0)

    def body(text: str, size: int = 10, style: str = "") -> None:
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("helvetica", style, size)
        pdf.multi_cell(0, 5.5, _s(text), **_WRAP)

    def bullet(text: str, indent: float = 4) -> None:
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("helvetica", "", 10)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(
            pdf.w - pdf.l_margin - pdf.r_margin - indent, 5.5, _s(f"- {text}"), **_WRAP
        )

    # --- Title block ---------------------------------------------------
    h1(analysis.title)
    meta_bits = [f"Category: {analysis.category}"]
    if analysis.ministry:
        meta_bits.append(f"Ministry: {analysis.ministry}")
    if analysis.generated_at:
        meta_bits.append(f"Generated: {analysis.generated_at}")
    else:
        meta_bits.append(f"Generated: {datetime.now().date().isoformat()}")
    body(" | ".join(meta_bits), size=9, style="I")

    if analysis.one_line_essence:
        pdf.ln(2)
        body(analysis.one_line_essence, size=11, style="B")

    if analysis.objective:
        h2("Objective")
        body(analysis.objective)

    if analysis.plain_summary:
        h2("Source-based summary" if analysis.metrics and analysis.metrics.engine == "local" else "Citizen-friendly summary")
        body(analysis.plain_summary)

    if analysis.ai_draft:
        h2("Optional AI draft - may contain errors")
        body(analysis.ai_draft)

    if analysis.key_provisions:
        h2("Selected source passages" if analysis.metrics and analysis.metrics.engine == "local" else "Key provisions")
        for prov in analysis.key_provisions:
            ref = f" ({prov.clause_reference})" if prov.clause_reference else ""
            bullet(f"{prov.title}{ref}: {prov.description}")

    if analysis.sector_impacts:
        h2("Sector and industry impact")
        for s in analysis.sector_impacts:
            bullet(f"{s.sector} - {s.direction}, {s.magnitude} impact. {s.rationale}")

    if analysis.topic_evidence:
        h2("Possible relevant sectors")
        body("Keyword-matched topics are not assessments of impact. Verify the excerpts against the full source.")
        for item in analysis.topic_evidence:
            bullet(f"{item.sector} (matched: {item.matched_phrase}): {item.excerpt}")
    elif analysis.detected_sectors:
        h2("Possible relevant sectors")
        body("These are topic labels, not assessments of impact. Verify them against the source.")
        for sector in analysis.detected_sectors:
            bullet(sector)

    horizons = [
        ("Short term (0-1 year)", analysis.term_impacts.short_term),
        ("Medium term (1-5 years)", analysis.term_impacts.medium_term),
        ("Long term (5+ years)", analysis.term_impacts.long_term),
    ]
    if any(h.effects or h.risks for _, h in horizons):
        h2("Impact forecast")
        for label, horizon in horizons:
            if not (horizon.effects or horizon.risks):
                continue
            body(label, style="B")
            for e in horizon.effects:
                bullet(e)
            for r in horizon.risks:
                bullet(f"Risk: {r}", indent=8)

    if analysis.positives:
        h2("Positives")
        for p in analysis.positives:
            bullet(p)
    if analysis.negatives:
        h2("Negatives")
        for n in analysis.negatives:
            bullet(n)

    if analysis.risk_flags:
        h2(f"Risk flags (overall controversy: {analysis.controversy_level})")
        for flag in analysis.risk_flags:
            groups = f" Affects: {', '.join(flag.affected_groups)}." if flag.affected_groups else ""
            bullet(f"[{flag.severity.upper()}] {flag.topic}: {flag.concern}{groups}")

    if analysis.related_legislation:
        h2("Related legislation and history")
        if analysis.policy_background:
            body(analysis.policy_background)
            pdf.ln(1)
        for act in analysis.related_legislation:
            year = f", {act.year}" if act.year else ""
            bullet(f"{act.name}{year} ({act.relationship}, confidence {act.confidence}). {act.note}")

    if analysis.citizen_faq:
        h2("Questions people ask")
        for faq in analysis.citizen_faq:
            body(f"Q: {faq.question}", style="B")
            body(f"A: {faq.answer}")
            pdf.ln(1)

    h2("About this report")
    engine = analysis.metrics.model if analysis.metrics and analysis.metrics.model else "AI"
    body(
        f"Generated by PPIIA using {engine}. This is an AI-generated civic education "
        "aid, not legal advice. Verify details against the official bill text on "
        "sansad.in or prsindia.org before acting on it.",
        size=9,
        style="I",
    )

    return bytes(pdf.output())
