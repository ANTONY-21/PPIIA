# Public Policy Insight & Impact Analyzer (PPIIA)

PPIIA helps users explore English government Bills from a selectable-text PDF, a direct document URL, or pasted text. It provides a source-based local analysis, an optional local AI draft, and a separate Claude-powered analysis. The dashboard can export JSON and PDF reports.

**This is an education and research aid, not legal advice.** Automatically selected passages and topic suggestions can be incomplete, incidental, or affected by PDF extraction errors. Check the complete Bill and consult a qualified expert before relying on an interpretation.

## Local source-based analysis

Local analysis is the default and does not require an API key or model download. It:

- Extracts text from a PDF, direct URL, or pasted text.
- Selects the Bill's stated purpose and potentially substantive passages using text patterns. The summary retains source wording and may remain technical.
- Suggests possible sectors from specific keyword matches, displaying the matched phrase and an excerpt. A match does **not** establish an actual impact, and the app may omit relevant sectors or abstain when it finds no clear match.
- Shows possible references to legislation and allows JSON and PDF export.

The local default does not assess controversy, determine legal status, or forecast future effects. Image-only PDFs require OCR before upload. English text-based documents work best; selection and extraction can still miss or distort details.

### Optional AI draft

Select **Generate an additional AI draft** in the sidebar to run DistilBART locally. This is slower and may generate inaccurate or incomplete text. The draft is displayed separately from the source-based summary and requires the optional dependencies and model downloads.

### Claude analysis

The Claude option uses an Anthropic API key and a four-stage structured pipeline for metadata, chronology, impact and risk, and a citizen summary. Its outputs are model-generated and require independent verification. Structured schema validation checks format, not truth. The model selections and API access must be available to the account used.

A precomputed sample analysis is also available via **Load sample demo**. It demonstrates the interface without running a new analysis.

## Setup

```bash
git clone https://github.com/ANTONY-21/PPIIA.git
cd PPIIA
python -m venv .venv
```

Activate the environment on Windows with `.venv\Scripts\activate` or on macOS/Linux with `source .venv/bin/activate`, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

For the optional local AI draft, also install:

```bash
pip install -r requirements-local.txt
```

The first AI run downloads model weights unless they are already cached. Set `HF_HOME` to change the Hugging Face cache location. Advanced users can set `PPIIA_MODELS_DIR` to a directory containing `summarizer` and `classifier` checkpoint folders.

For Claude, configure `ANTHROPIC_API_KEY` in the environment or a local `.env` based on `.env.example`, or enter it in the app. Keep keys private and never commit them. Test model availability and usage costs before a live demonstration.

## Using the dashboard

1. Select **Local analysis** or the Claude engine.
2. Upload a text-based PDF, provide a direct document URL, or paste text.
3. Select **Analyze bill**. For the saved example, select **Load sample demo** instead.
4. Review the source summary and the topic evidence. Compare excerpts with the full Bill.
5. Use **Export & data** to download the report.

A Parliament listing page is not itself a Bill. Open an individual Bill and use its direct PDF link, or download and upload the PDF. Some dynamic pages cannot be extracted.

## Command line

```bash
# Local source-based analysis to JSON
python ppiia.py path/to/bill.pdf --out outputs/report.json

# Claude analysis (requires an API key and an available model)
python ppiia_ai.py sample --out outputs/analysis.json --pdf outputs/report.pdf
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite covers source-evidence selection, schema adaptation, document preparation, sample data, PDF generation, and the Claude pipeline with a mocked API. The tests do not establish accuracy on every Bill or verify a live API call. A broader evaluation requires a diverse, expert-reviewed set of complete Bills.

## Project files

- `app.py` — Streamlit interface and input handling.
- `ppiia.py` — local ingestion, source-based analysis, and optional model draft.
- `evidence.py` — source passage and topic-evidence selection.
- `ppiia_ai.py` — Claude pipeline and readability metrics.
- `schemas.py` — shared Pydantic output models.
- `report_pdf.py` — PDF report generation.
- `data/sample_analysis.json` and `sample_bill.pdf` — precomputed demonstration.
- `tests/` — automated checks; `INTERVIEW_GUIDE.md` — evaluation notes.

## Deployment

Streamlit Community Cloud can deploy this repository using `app.py` as its entry point. The source-based mode works without an API key. A public deployment should be tested independently, including access and resource limits, before it is shared. See the [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Limitations

- The source-based summary is a selection of source text, not a complete or guaranteed-correct interpretation.
- Topic matches are evidence of words in a document, not proof of an affected sector.
- Scanned PDFs require OCR; extraction from complex layouts and dynamic sites can fail.
- AI drafts and generated chronology, impacts, or risk analysis may be wrong and should be checked against official sources.

Consult [Sansad](https://sansad.in), [PRS Legislative Research](https://prsindia.org), or the [e-Gazette](https://egazette.gov.in) for original documents and current legal status.
