# PPIIA live-evaluation guide

This guide describes the current local implementation. Explain the work in your own words and make only claims you can demonstrate. The project is a civic education and research aid, not legal advice.

## Suggested session

GUVI's evaluation guide describes a 30-minute session: approximately 3 minutes for an introduction, 12 for the project, 10 for questions, and 5 for feedback. Follow your evaluator's instructions if the timing differs.

## Introduction

Introduce yourself and your batch, then explain that PPIIA helps readers explore government Bills. The problem is that legal documents can be difficult to navigate. The application extracts document text, selects source passages, suggests possible topics with supporting evidence, and provides reports.

## Demonstration

1. Start the Streamlit app and select **Local analysis (no API key)**.
2. Upload an English, selectable-text Bill PDF or paste a direct document URL or text.
3. Select **Analyze bill**. Show the stated purpose and selected provision in the source-based summary.
4. Open **Sector impact**. Explain that each topic is suggested by matched words and an excerpt; it is not a verified impact assessment.
5. Compare a selected passage with the original PDF. Discuss extraction errors and important provisions the selection may have missed.
6. Show **Export & data** and download the JSON or PDF report.
7. Show the relevant code: `ppiia.py` for ingestion and orchestration, `evidence.py` for passage and topic selection, `schemas.py` for structured output, `app.py` for the interface, and `report_pdf.py` for the PDF.

The **Load sample demo** button loads a precomputed analysis. Identify it as precomputed; do not present it as a new model run. An optional checkbox generates a separate local AI draft, which can be inaccurate. The Claude engine is a separate API-based path that requires a key and an available model.

## Questions to practise

**How is the local summary created?**

The app uses text patterns to select the Bill's stated purpose and candidate provisions. It preserves source wording instead of claiming a fluent rewrite is necessarily accurate. It may miss a clause, and PDF extraction can disturb reading order.

**How are sectors identified?**

The local default matches a limited set of specific phrases and displays a supporting excerpt. It can abstain when there is no clear match. A word match does not prove a sector benefits, loses, or is even materially affected.

**How do you handle longer documents?**

The source-based default scans the available text. The optional model draft uses token-sized chunks and combines partial summaries; a generated draft may still omit or distort information. The Claude pipeline has its own input cap.

**What does Pydantic do?**

It validates the structure and types of the analysis. Valid structure does not mean the content or legal interpretation is correct.

**How do you test the project?**

The automated tests cover source selection, topic evidence, schema adaptation, preparation, PDF generation, and a mocked Claude pipeline. A passing test suite does not establish accuracy on all Bills or verify a live API request. A broader evaluation needs diverse, expert-reviewed Bills.

**What are the main limitations?**

Scanned PDFs need OCR. Complex PDF layouts and dynamic pages can extract incorrectly. The source selection can omit important clauses, sector matches can be incidental, and AI-generated drafts can make mistakes. Local mode does not assess risk or forecast outcomes.

**What would you improve?**

Add OCR, compare results with an expert-reviewed dataset, include source-page references, and improve coverage across Bill formats and languages.

## Before the call

- Practise on the same computer and network you will use for the evaluation.
- Have the GitHub repository, a Bill PDF, and a backup report ready.
- Verify any deployed demo and video link if required by your course.
- Keep API keys and private information off screen.
- Rehearse the distinction between source selection, the optional AI draft, and the precomputed sample.
