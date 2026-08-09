# 🏛️ Public Policy Insight & Impact Analyzer (PPIIA)

**Simplifying government Bills for every citizen.**

Give PPIIA any Indian government Bill — a PDF, a [sansad.in](https://sansad.in/ls/knowledge-centre/government-bills) URL, or pasted text — and it produces a structured, plain-English briefing:

- 📖 **Citizen-friendly summary** — 10–20 lines at an 8th-grade reading level, plus an FAQ and "what changes for you" persona cards
- 🎯 **Objective, ministry & key provisions** with clause references
- 🏭 **Sector & industry impact** — direction, magnitude, opportunities and restrictions per sector, with a chart
- 🕰️ **Chronology** — predecessor acts, amendments, and a policy-evolution timeline
- 🔮 **Impact forecast** — short-term (0–1 y), medium-term (1–5 y), long-term (5+ y), each with risks
- ⚠️ **Risk & controversy detection** — clauses likely to raise costs, affect rights, or cause friction
- ⬇️ **Exports** — the full JSON report and a downloadable PDF briefing

## Two analysis engines, one dashboard

| | 🤖 Claude AI engine (`ppiia_ai.py`) | 💻 Local engine (`ppiia.py`) |
|---|---|---|
| Needs | `ANTHROPIC_API_KEY` | Nothing (offline after ~700 MB model download) |
| Models | Claude Opus 5 (default), Sonnet 5, or Haiku 4.5 | `distilbart-cnn` + `distilbart-mnli` on CPU |
| Output | Full structured analysis (all features above), schema-validated JSON | Summary, sectors, entities, heuristic impacts |
| Quality | Genuinely 8th-grade friendly, deep analysis | Decent extractive feel |

The dual-mode design means the project runs for **anyone** — no key required — while an API key unlocks the complete analysis. A **precomputed demo** of the bundled sample bill loads with one click, needing neither.

## Quick start

```bash
git clone https://github.com/ANTONY-21/PPIIA.git
cd PPIIA
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**Claude engine (recommended):** copy `.env.example` to `.env` and set your `ANTHROPIC_API_KEY` (or paste the key in the app sidebar).

**Local engine (optional, offline):** `pip install -r requirements-local.txt`

Then launch the dashboard:

```bash
streamlit run app.py
```

No key handy? Click **⚡ Load sample demo** in the sidebar to explore a precomputed analysis of the bundled sample bill (*The Andhra Pradesh Reorganisation (Amendment) Bill, 2026*).

### CLI

```bash
# Claude engine — JSON + PDF report
python ppiia_ai.py sample --out outputs/analysis.json --pdf outputs/report.pdf
python ppiia_ai.py path/to/bill.pdf
python ppiia_ai.py https://sansad.in/.../bill.pdf

# Local engine
python ppiia.py path/to/bill.pdf --out outputs/report.json
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

21 tests cover schema validation/coercion, text preparation, the four-stage pipeline (API mocked), the bundled demo data, and PDF generation. No test calls the network.

## How the Claude pipeline works

```
PDF / URL / text
      │  ingestion (pdfplumber · requests + BeautifulSoup)   ← shared with local engine
      ▼
prepare_text()          keeps line/clause structure, strips page numbers,
      │                 joins hyphenated line-wraps, caps input size
      ▼
┌─ Stage 1  Bill topic classifier ─ title, ministry, category, objective,
│                                   key provisions, stakeholders
├─ Stage 2  Chronology builder ──── policy background, related acts
│                                   (with confidence levels), timeline
├─ Stage 3  Impact analyzer ─────── per-sector impacts, short/medium/long-term
│                                   forecast, positives/negatives, risk flags
└─ Stage 4  Citizen summarizer ──── 8th-grade summary, FAQ, persona impacts
      │
      ▼
BillAnalysis (pydantic)  →  dashboard · JSON export · PDF report
```

Implementation notes:

- **Structured outputs** — every stage uses `client.messages.parse()` with a strict pydantic schema, so the model's JSON is validated at the API layer; no regex JSON-repair needed.
- **Prompt caching** — the bill text is sent as an identical cached block in all four calls, so stages 2–4 read it from Anthropic's prompt cache instead of re-billing full input.
- **Hallucination guardrails** — every related act carries a `confidence` field: `high` = cited in the bill text, `medium/low` = from the model's general knowledge; the UI says so.
- **Graceful failure** — refusals, rate limits, auth errors and truncated outputs surface as clear, actionable messages in the UI.

The dashboard renders both engines through one schema: `BillAnalysis.from_local()` adapts the local engine's flat report, so tabs, charts and exports work in either mode.

## Project structure

```
PPIIA/
├── app.py                   # Streamlit dashboard (both engines + demo mode)
├── ppiia.py                 # local engine: ingestion, cleaning, HF models, CLI
├── ppiia_ai.py              # Claude engine: 4-stage pipeline, metrics, CLI
├── schemas.py               # pydantic models: BillAnalysis + LLM stage schemas
├── report_pdf.py            # PDF briefing generator (fpdf2)
├── data/sample_analysis.json# precomputed demo of the sample bill
├── sample_bill.pdf          # AP Reorganisation (Amendment) Bill, 2026
├── tests/                   # 21 offline tests
├── docs/langflow_design.md  # the pipeline as a Langflow graph
├── ppiia_colab.ipynb        # Colab notebook + Gradio demo (local engine)
├── requirements.txt         # core (Claude engine + dashboard)
├── requirements-local.txt   # + torch/transformers for offline mode
├── requirements-dev.txt     # + pytest
└── INTERVIEW_GUIDE.md       # live-evaluation study guide
```

## Evaluation metrics

Shown in the app under **🗂️ Export & data** for every analysis:

| Project metric | How PPIIA measures it |
|---|---|
| Quality of summarization / readability | Flesch reading ease + Flesch-Kincaid grade of the summary (target ≤ 8) |
| Topic classification accuracy | Closed category vocabulary + reviewable category/sector labels per bill |
| Chronology correctness | Per-act `confidence` labels separate text-cited facts from model knowledge |
| Relevance of industry impact | Per-sector rationale, opportunities and restrictions are all inspectable |
| Processing time | Wall-clock seconds per document, shown in the metrics row |

## Deployment

- **Streamlit Community Cloud** — push to GitHub, create an app pointing at `app.py`, and add `ANTHROPIC_API_KEY` in the app's *Secrets*.
- **Hugging Face Spaces** — create a Streamlit Space, upload the repo (keep `requirements.txt` as-is; skip `requirements-local.txt` to stay in the free tier), and set `ANTHROPIC_API_KEY` as a Space secret.
- **Colab / Gradio** — `ppiia_colab.ipynb` runs the local engine with a public Gradio URL.

## Limitations & disclaimer

- Scanned (image-only) PDFs need OCR first — PPIIA reads embedded text.
- JavaScript-heavy portal pages may not fetch; direct PDF links work best.
- Historical context beyond the bill text is model knowledge — check the confidence labels and verify before quoting.

PPIIA is an AI-powered **civic education tool, not legal advice**. Always verify against official sources: [sansad.in](https://sansad.in) · [prsindia.org](https://prsindia.org) · [egazette.gov.in](https://egazette.gov.in).

## Tech tags

`Python` `NLP` `Legal NLP` `Claude API` `Structured Outputs` `Prompt Caching` `Transformers` `Zero-Shot Classification` `Summarization` `Streamlit` `Plotly` `Pydantic` `Regulation Analytics`
