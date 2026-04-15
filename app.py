"""
PPIIA - Streamlit Dashboard
===========================
Interactive web app that wraps `ppiia.analyze(...)`.

LOCAL AI ONLY — runs Hugging Face transformers on CPU, no API keys.

Features:
    - Upload a Bill PDF, paste a URL, or paste raw text
    - Tabbed view: Summary / Impacts / Chronology / Sectors / Raw JSON
    - Download the report as JSON

Run:
    streamlit run app.py
"""

import json
import tempfile
import streamlit as st

from ppiia import analyze, BillReport

st.set_page_config(page_title="PPIIA - Bill Analyzer",
                   page_icon="🏛️", layout="wide")

st.title("🏛️ Public Policy Insight & Impact Analyzer")
st.caption("Simplifying government Bills for every citizen.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.success("**Mode:** Local AI\n\n"
                   "Runs Hugging Face transformers on your CPU — "
                   "no API keys, no cloud, 100% offline after the first "
                   "model download.")

st.sidebar.markdown("### How to use")
st.sidebar.markdown(
    "1. Upload a PDF **or** paste a URL **or** paste text\n"
    "2. Click **Analyze**\n"
    "3. Read each tab\n"
    "4. Download the JSON report"
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
input_mode = st.radio("Choose input type",
                      ["📄 PDF Upload", "🔗 URL", "📝 Paste Text"],
                      horizontal=True)

source = None
if input_mode == "📄 PDF Upload":
    up = st.file_uploader("Upload a Bill PDF", type=["pdf"])
    if up:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(up.read())
            source = tmp.name
        st.success(f"Saved: {up.name} ({up.size // 1024} KB)")
elif input_mode == "🔗 URL":
    source = st.text_input(
        "Bill URL",
        placeholder="https://sansad.in/ls/knowledge-centre/government-bills/..."
    )
else:
    raw = st.text_area("Paste Bill text here", height=280)
    source = raw if raw and raw.strip() else None


if st.button("Analyze", type="primary", disabled=source is None):
    with st.spinner("Analyzing — CPU inference, this may take 1–3 minutes ..."):
        try:
            report: BillReport = analyze(source)
            st.session_state["report"] = report.to_dict()
        except Exception as e:
            st.error(f"Analysis failed: {e}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
report = st.session_state.get("report")
if not report:
    st.info("Upload / paste a Bill and click **Analyze** to see results.")
    st.stop()

st.divider()
st.subheader(report.get("title") or "Bill Analysis")
if report.get("ministry"):
    st.caption(f"**Ministry:** {report['ministry']}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📝 Summary",
    "📊 Impacts",
    "🕑 Chronology",
    "🏭 Sectors",
    "⚠️ Risks",
    "🗂️ Raw JSON",
])

with tab1:
    if report.get("objective"):
        st.markdown(f"**Objective:** {report['objective']}")
    st.markdown("### Citizen-friendly summary")
    st.write(report.get("summary") or "*No summary generated.*")

with tab2:
    st.markdown("### Short-term impact (0–1 year)")
    st.write(report.get("impacts", {}).get("short_term") or "—")
    st.markdown("### Medium-term impact (1–5 years)")
    st.write(report.get("impacts", {}).get("medium_term") or "—")
    st.markdown("### Long-term impact (> 5 years)")
    st.write(report.get("impacts", {}).get("long_term") or "—")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ✅ Positives")
        for p in report.get("positives", []) or ["—"]:
            st.markdown(f"- {p}")
    with col2:
        st.markdown("#### ❌ Negatives")
        for n in report.get("negatives", []) or ["—"]:
            st.markdown(f"- {n}")

with tab3:
    chrono = report.get("chronology") or []
    if chrono:
        for idx, item in enumerate(chrono, 1):
            st.markdown(f"**{idx}.** {item}")
    else:
        st.info("No explicit predecessor acts detected.")
    ents = report.get("key_entities", {})
    if ents:
        st.markdown("##### Years mentioned")
        st.write(", ".join(ents.get("years", [])) or "—")
        st.markdown("##### Ministries / bodies")
        st.write(", ".join(ents.get("ministries", [])) or "—")

with tab4:
    sectors = report.get("sectors") or []
    if sectors:
        st.markdown("#### Top affected sectors")
        for s in sectors:
            st.markdown(f"- **{s}**")
    else:
        st.info("No sector classification available.")

with tab5:
    risks = report.get("risks") or []
    if risks:
        for r in risks:
            st.markdown(f"- ⚠️ {r}")
    else:
        st.info("No explicit risks detected by the local model.")

with tab6:
    st.json(report)
    st.download_button(
        "Download JSON report",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="bill_report.json",
        mime="application/json",
    )

st.divider()
st.caption("Built with Hugging Face transformers (local CPU) · Streamlit · Python")
