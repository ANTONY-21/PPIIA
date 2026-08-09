"""
PPIIA — Streamlit Dashboard
===========================
Interactive web app for the Public Policy Insight & Impact Analyzer.

Two analysis engines share one dashboard:

    * Claude AI (recommended)  — four-stage LLM pipeline with structured
      outputs (`ppiia_ai.py`). Needs an Anthropic API key.
    * Local models (offline)   — Hugging Face transformers on CPU
      (`ppiia.py`). No key needed; install requirements-local.txt.

A precomputed demo of `sample_bill.pdf` can be loaded with one click —
no API key and no model download required.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

import ppiia
import ppiia_ai
from ppiia_ai import AnalysisError, MODEL_CHOICES, SAMPLE_PDF_PATH
from report_pdf import build_pdf_report
from schemas import BillAnalysis, SourceInfo

st.set_page_config(
    page_title="PPIIA — Bill Analyzer",
    page_icon="🏛️",
    layout="wide",
)

DIRECTION_COLORS = {
    "positive": "#2e7d32",
    "negative": "#c62828",
    "mixed": "#ef6c00",
    "neutral": "#757575",
}
MAGNITUDE_SCORE = {"low": 1, "medium": 2, "high": 3}
SEVERITY_BADGE = {"high": "🔴", "medium": "🟠", "low": "🟢"}


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def extract_input_text(mode: str, uploaded, url_value: str, pasted: str) -> tuple[str, SourceInfo]:
    """Resolve the chosen input into (raw_text, SourceInfo). Raises on failure."""
    if mode == "📄 Upload PDF":
        if uploaded is None:
            raise AnalysisError("Upload a bill PDF first.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        raw = ppiia.extract_text_from_pdf(tmp_path)
        return raw, ppiia_ai.build_source_info(raw, "pdf", uploaded.name)

    if mode == "🔗 Fetch from URL":
        if not url_value.strip():
            raise AnalysisError("Paste a bill URL first.")
        raw = ppiia.extract_text_from_url(url_value.strip())
        return raw, ppiia_ai.build_source_info(raw, "url", url_value.strip())

    if mode == "📝 Paste text":
        if not pasted.strip():
            raise AnalysisError("Paste the bill text first.")
        return pasted, ppiia_ai.build_source_info(pasted, "text", "Pasted text")

    # Sample bill
    raw = ppiia.extract_text_from_pdf(str(SAMPLE_PDF_PATH))
    return raw, ppiia_ai.build_source_info(raw, "sample", SAMPLE_PDF_PATH.name)


def local_models_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


def run_analysis(engine_name: str, mode: str, uploaded, url_value: str, pasted: str,
                 api_key: str | None, model_id: str) -> None:
    try:
        with st.status("Reading the document…", expanded=True) as status:
            raw, source = extract_input_text(mode, uploaded, url_value, pasted)
            st.write(f"Extracted {source.word_count:,} words from **{source.name}**.")

            if engine_name.startswith("Claude"):
                status.update(label="Analyzing with Claude…")
                analysis = ppiia_ai.analyze_with_claude(
                    raw, source, api_key=api_key, model=model_id, progress=st.write
                )
            else:
                if not local_models_available():
                    raise AnalysisError(
                        "Local models are not installed. Run "
                        "`pip install -r requirements-local.txt` (about 700 MB of "
                        "model downloads on first use), or switch to the Claude engine."
                    )
                status.update(label="Analyzing with local Hugging Face models (CPU — this can take a few minutes)…")
                report = ppiia.analyze(raw)
                analysis = BillAnalysis.from_local(report.to_dict(), source)
                analysis.metrics = ppiia_ai.compute_metrics(
                    engine="local",
                    model="distilbart-cnn + distilbart-mnli",
                    processing_seconds=None,
                    input_words=source.word_count,
                    summary_text=analysis.plain_summary,
                )

            status.update(label="Analysis complete", state="complete", expanded=False)
        st.session_state["analysis"] = analysis
    except AnalysisError as exc:
        st.error(str(exc))
    except Exception as exc:  # ingestion/network failures surface here
        st.error(f"Analysis failed: {exc}")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def sector_impact_chart(analysis: BillAnalysis):
    rows = [
        {
            "Sector": s.sector,
            "Impact strength": MAGNITUDE_SCORE.get(s.magnitude, 2),
            "Direction": s.direction,
            "Why": s.rationale,
        }
        for s in analysis.sector_impacts
    ]
    df = pd.DataFrame(rows).sort_values("Impact strength")
    fig = px.bar(
        df,
        x="Impact strength",
        y="Sector",
        color="Direction",
        orientation="h",
        color_discrete_map=DIRECTION_COLORS,
        hover_data={"Why": True, "Impact strength": False},
        category_orders={"Direction": list(DIRECTION_COLORS)},
    )
    fig.update_layout(
        height=max(260, 60 * len(df)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"], range=[0, 3.3]),
        legend_title_text="Direction for the sector",
    )
    return fig


def timeline_chart(analysis: BillAnalysis):
    rows = []
    for i, ev in enumerate(analysis.timeline):
        match = re.search(r"\d{4}", ev.year)
        if match:
            rows.append(
                {
                    "Year": int(match.group()),
                    "Milestone": ev.label,
                    "Detail": ev.description,
                    "lane": i % 2,
                }
            )
    if len(rows) < 2:
        return None
    df = pd.DataFrame(rows)
    fig = px.scatter(
        df, x="Year", y="lane", text="Milestone", hover_data={"Detail": True, "lane": False}
    )
    fig.update_traces(
        marker=dict(size=14, color="#1f77b4"), textposition="top center", textfont_size=11
    )
    fig.update_layout(
        height=280,
        yaxis=dict(visible=False, range=[-0.8, 1.9]),
        xaxis=dict(dtick=None, title=None, tickformat="d"),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )
    return fig


def render_dashboard(analysis: BillAnalysis) -> None:
    st.subheader(analysis.title)
    caption_bits = []
    if analysis.ministry:
        caption_bits.append(f"**Ministry:** {analysis.ministry}")
    if analysis.source:
        caption_bits.append(f"**Source:** {analysis.source.name}")
        if analysis.source.truncated:
            caption_bits.append("⚠️ document truncated for analysis")
    if caption_bits:
        st.caption(" · ".join(caption_bits))
    if analysis.one_line_essence:
        st.info(f"**In one line:** {analysis.one_line_essence}")

    m = analysis.metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Category", analysis.category)
    c2.metric("Sectors affected", len(analysis.sector_impacts) or "—")
    c3.metric(
        "Reading grade level",
        m.readability_grade if m and m.readability_grade is not None else "—",
        help="Flesch-Kincaid grade of the citizen summary — the project targets ≤ 8.",
    )
    c4.metric(
        "Processing time",
        f"{m.processing_seconds:.0f} s" if m and m.processing_seconds else "—",
    )
    c5.metric("Controversy", analysis.controversy_level.capitalize())

    tabs = st.tabs(
        [
            "📖 Citizen summary",
            "🏭 Sector impact",
            "🕰️ History & timeline",
            "🔮 Impact forecast",
            "⚠️ Risks",
            "🗂️ Export & data",
        ]
    )

    # --- Citizen summary ------------------------------------------------
    with tabs[0]:
        if analysis.objective:
            st.markdown(f"**Objective:** {analysis.objective}")
        st.markdown("#### What this bill means in plain language")
        st.markdown(analysis.plain_summary or "*No summary generated.*")

        if analysis.persona_impacts:
            st.markdown("#### What changes for you")
            cols = st.columns(len(analysis.persona_impacts))
            for col, persona in zip(cols, analysis.persona_impacts):
                with col, st.container(border=True):
                    st.markdown(f"**{persona.persona}**")
                    st.write(persona.impact)

        if analysis.citizen_faq:
            st.markdown("#### Questions people ask")
            for faq in analysis.citizen_faq:
                with st.expander(faq.question):
                    st.write(faq.answer)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### ✅ Positives")
            for p in analysis.positives or ["—"]:
                st.markdown(f"- {p}")
        with col_b:
            st.markdown("#### ⚠️ Negatives")
            for n in analysis.negatives or ["—"]:
                st.markdown(f"- {n}")

    # --- Sector impact ---------------------------------------------------
    with tabs[1]:
        if analysis.sector_impacts:
            st.plotly_chart(sector_impact_chart(analysis), width="stretch")
            for s in analysis.sector_impacts:
                arrow = {"positive": "🟢", "negative": "🔴", "mixed": "🟠", "neutral": "⚪"}[s.direction]
                with st.expander(f"{arrow} {s.sector} — {s.direction}, {s.magnitude} impact"):
                    if s.rationale:
                        st.write(s.rationale)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Opportunities**")
                        for o in s.opportunities or ["—"]:
                            st.markdown(f"- {o}")
                    with col2:
                        st.markdown("**Restrictions / compliance**")
                        for r in s.restrictions or ["—"]:
                            st.markdown(f"- {r}")
        else:
            st.info("No sector impact analysis available for this document.")
        if analysis.stakeholders:
            st.markdown("**Stakeholders:** " + " · ".join(analysis.stakeholders))

    # --- History & timeline ----------------------------------------------
    with tabs[2]:
        if analysis.policy_background:
            st.markdown("#### Why this bill exists")
            st.write(analysis.policy_background)
        fig = timeline_chart(analysis)
        if fig is not None:
            st.markdown("#### Policy evolution")
            st.plotly_chart(fig, width="stretch")
        elif analysis.timeline:
            st.markdown("#### Policy evolution")
            for ev in analysis.timeline:
                st.markdown(f"**{ev.year}** — {ev.label}. {ev.description}")
        if analysis.related_legislation:
            st.markdown("#### Related acts and amendments")
            df = pd.DataFrame(
                [
                    {
                        "Legislation": r.name,
                        "Year": r.year or "",
                        "Relationship": r.relationship,
                        "Note": r.note,
                        "Confidence": r.confidence,
                    }
                    for r in analysis.related_legislation
                ]
            )
            st.dataframe(df, width="stretch", hide_index=True)
            st.caption(
                "Confidence *high* = cited in the bill text; *medium/low* = drawn from the "
                "model's general knowledge — verify before quoting."
            )
        if not (analysis.policy_background or analysis.timeline or analysis.related_legislation):
            st.info("No chronology available for this document.")

    # --- Impact forecast --------------------------------------------------
    with tabs[3]:
        horizons = [
            ("first year after passage", "Short term", analysis.term_impacts.short_term),
            ("1–5 years out", "Medium term", analysis.term_impacts.medium_term),
            ("beyond 5 years", "Long term", analysis.term_impacts.long_term),
        ]
        cols = st.columns(3)
        for col, (hint, label, horizon) in zip(cols, horizons):
            with col, st.container(border=True):
                st.markdown(f"**{label}**")
                st.caption(hint)
                for e in horizon.effects or ["—"]:
                    st.markdown(f"- {e}")
                if horizon.risks:
                    st.markdown("*Risks:*")
                    for r in horizon.risks:
                        st.markdown(f"- ⚠️ {r}")

    # --- Risks ------------------------------------------------------------
    with tabs[4]:
        if analysis.risk_flags:
            st.markdown(
                f"Overall controversy level: **{analysis.controversy_level.capitalize()}**"
            )
            for flag in sorted(
                analysis.risk_flags,
                key=lambda f: {"high": 0, "medium": 1, "low": 2}[f.severity],
            ):
                with st.container(border=True):
                    st.markdown(f"{SEVERITY_BADGE[flag.severity]} **{flag.topic}** — {flag.severity} severity")
                    st.write(flag.concern)
                    if flag.affected_groups:
                        st.caption("Affects: " + ", ".join(flag.affected_groups))
        else:
            st.info("No specific risk flags were raised for this bill.")

    # --- Export & data ----------------------------------------------------
    with tabs[5]:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download JSON report",
                data=analysis.model_dump_json(indent=2),
                file_name="ppiia_analysis.json",
                mime="application/json",
                width="stretch",
            )
        with col2:
            st.download_button(
                "⬇️ Download PDF summary",
                data=build_pdf_report(analysis),
                file_name="ppiia_report.pdf",
                mime="application/pdf",
                width="stretch",
            )
        if m:
            st.markdown("#### Evaluation metrics")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Metric": "Engine", "Value": f"{m.engine} ({m.model})"},
                        {"Metric": "Processing time (s)", "Value": m.processing_seconds},
                        {"Metric": "Flesch reading ease (higher = easier)", "Value": m.flesch_reading_ease},
                        {"Metric": "Flesch-Kincaid grade (target ≤ 8)", "Value": m.readability_grade},
                        {"Metric": "Input words", "Value": m.input_words},
                        {"Metric": "Summary words", "Value": m.summary_words},
                        {"Metric": "Compression ratio", "Value": m.compression_ratio},
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        st.markdown("#### Full structured output")
        st.json(analysis.model_dump(), expanded=False)


def render_landing() -> None:
    st.markdown(
        """
##### Turn any government bill into a briefing every citizen can read.

1. Pick an input in the sidebar — upload a **PDF**, paste a **URL** from
   [sansad.in](https://sansad.in/ls/knowledge-centre/government-bills), paste raw **text**,
   or use the built-in **sample bill**.
2. Choose an engine — **Claude AI** for the full analysis, or the free **local models**.
3. Click **Analyze bill**.

No API key handy? Click **Load sample demo** in the sidebar to explore a precomputed
analysis instantly.
"""
    )
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("**📖 Plain-language summary**")
        st.caption("10–20 lines at an 8th-grade reading level, plus FAQ and per-persona impacts.")
    with c2, st.container(border=True):
        st.markdown("**🏭 Sector impact & forecast**")
        st.caption("Which industries win or lose, and what changes over 1, 5, and 10 years.")
    with c3, st.container(border=True):
        st.markdown("**🕰️ History & risk flags**")
        st.caption("Predecessor acts, policy timeline, and clauses likely to cause friction.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🏛️ Public Policy Insight & Impact Analyzer")
st.caption("Simplifying government bills for every citizen — CivicTech / Public Policy Analytics")

with st.sidebar:
    st.markdown("### ⚙️ Engine")
    engine_name = st.radio(
        "Analysis engine",
        ["Claude AI (recommended)", "Local models (offline, free)"],
        help="Claude gives the full structured analysis. Local mode runs Hugging Face "
        "models on your CPU with no API key, but produces a shallower report.",
    )

    api_key: str | None = None
    model_id = ppiia_ai.DEFAULT_MODEL
    if engine_name.startswith("Claude"):
        model_label = st.selectbox("Claude model", list(MODEL_CHOICES))
        model_id = MODEL_CHOICES[model_label]
        env_key = ppiia_ai.get_api_key()
        if env_key:
            st.success("API key loaded from environment / .env")
            api_key = env_key
        else:
            api_key = st.text_input(
                "Anthropic API key",
                type="password",
                help="Stored only for this browser session. Or put it in a .env file "
                "(see .env.example).",
            ) or None
    else:
        if not local_models_available():
            st.warning(
                "Local models not installed — run "
                "`pip install -r requirements-local.txt` first (~700 MB)."
            )

    st.markdown("### 📥 Input")
    input_mode = st.radio(
        "Bill source",
        ["📄 Upload PDF", "🔗 Fetch from URL", "📝 Paste text", "⭐ Sample bill"],
    )
    uploaded = url_value = pasted = None
    if input_mode == "📄 Upload PDF":
        uploaded = st.file_uploader("Bill PDF", type=["pdf"])
    elif input_mode == "🔗 Fetch from URL":
        url_value = st.text_input(
            "Bill URL", placeholder="https://sansad.in/…/bill.pdf",
            help="Direct PDF links work best; JavaScript-heavy portal pages may not.",
        )
    elif input_mode == "📝 Paste text":
        pasted = st.text_area("Bill text", height=200)
    else:
        st.caption(f"Uses the bundled `{SAMPLE_PDF_PATH.name}`.")

    analyze_clicked = st.button("🔍 Analyze bill", type="primary", width="stretch")
    demo_clicked = st.button(
        "⚡ Load sample demo (no key needed)",
        width="stretch",
        help="Loads a precomputed Claude analysis of the sample bill.",
    )
    if st.session_state.get("analysis") is not None and st.button("🧹 Clear results", width="stretch"):
        st.session_state.pop("analysis", None)
        st.rerun()

if demo_clicked:
    try:
        st.session_state["analysis"] = ppiia_ai.load_sample_analysis()
    except FileNotFoundError:
        st.error("Sample analysis file is missing (data/sample_analysis.json).")

if analyze_clicked:
    run_analysis(engine_name, input_mode, uploaded, url_value or "", pasted or "", api_key, model_id)

analysis: BillAnalysis | None = st.session_state.get("analysis")
if analysis is None:
    render_landing()
else:
    render_dashboard(analysis)

st.divider()
st.caption(
    "PPIIA is an AI-powered civic education tool, not legal advice. Always verify against "
    "official sources: [sansad.in](https://sansad.in) · [prsindia.org](https://prsindia.org) · "
    "[egazette.gov.in](https://egazette.gov.in)"
)
