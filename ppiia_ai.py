"""
PPIIA — Claude AI analysis engine
=================================
The LLM-powered mode of the Public Policy Insight & Impact Analyzer.

Runs a four-stage analysis over a bill's text using the Anthropic Claude API
with structured outputs (every stage returns schema-validated JSON):

    1. Bill topic classifier + metadata     -> title, ministry, category,
                                               objective, key provisions
    2. Chronology builder                   -> policy background, related
                                               acts/amendments, timeline
    3. Impact analyzer + risk detection     -> sector impacts, short/medium/
                                               long-term forecast, risks
    4. Citizen summarizer                   -> 8th-grade summary, FAQ,
                                               persona impacts

Ingestion (PDF/URL/text) is shared with the local engine in `ppiia.py`.
Requires ANTHROPIC_API_KEY (in the environment or a `.env` file).

CLI:
    python ppiia_ai.py sample                        # analyze sample_bill.pdf
    python ppiia_ai.py path/to/bill.pdf --pdf report.pdf
    python ppiia_ai.py https://sansad.in/...bill.pdf --out analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import anthropic
import textstat
from dotenv import load_dotenv

import ppiia
from schemas import (
    AnalysisMetrics,
    BillAnalysis,
    LLMChronology,
    LLMImpact,
    LLMMetadata,
    LLMSummary,
    SourceInfo,
    TermImpacts,
)

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL = os.getenv("PPIIA_MODEL", "claude-opus-5")
MAX_OUTPUT_TOKENS = int(os.getenv("PPIIA_MAX_OUTPUT_TOKENS", "16000"))
MAX_INPUT_CHARS = int(os.getenv("PPIIA_MAX_INPUT_CHARS", "400000"))
REQUEST_TIMEOUT = float(os.getenv("PPIIA_TIMEOUT_SECONDS", "600"))

MODEL_CHOICES = {
    "Claude Opus 5 — best quality (default)": "claude-opus-5",
    "Claude Sonnet 5 — balanced": "claude-sonnet-5",
    "Claude Haiku 4.5 — fastest / cheapest": "claude-haiku-4-5",
}

CATEGORIES = [
    "Agriculture", "Finance", "Education", "Healthcare", "Technology",
    "Environment", "Defence", "Labour", "Energy", "Infrastructure",
    "Justice and Law", "Social Welfare", "Taxation", "Trade and Commerce",
    "Transport", "Governance", "Other",
]

SAMPLE_PDF_PATH = PROJECT_ROOT / "sample_bill.pdf"
SAMPLE_ANALYSIS_PATH = PROJECT_ROOT / "data" / "sample_analysis.json"


class AnalysisError(RuntimeError):
    """A user-facing failure in the AI analysis pipeline."""


def get_api_key() -> str | None:
    return os.getenv("ANTHROPIC_API_KEY") or None


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------
def prepare_text(raw: str) -> tuple[str, bool]:
    """Light cleanup that keeps line structure (clause numbers matter to the
    LLM, unlike the local summarizer which flattens whitespace).

    Returns (text, truncated_flag).
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)      # join hyphenated line-wraps
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)      # standalone page numbers
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    truncated = len(text) > MAX_INPUT_CHARS
    if truncated:
        text = text[:MAX_INPUT_CHARS]
    return text, truncated


def build_source_info(text: str, source_type: str, name: str, truncated: bool = False) -> SourceInfo:
    return SourceInfo(
        source_type=source_type,  # type: ignore[arg-type]
        name=name,
        char_count=len(text),
        word_count=len(text.split()),
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are PPIIA, an expert analyst of Indian parliamentary legislation. "
    "You explain government bills accurately and in plain language for ordinary "
    "citizens. Ground every claim in the bill text provided; where you draw on "
    "general knowledge of Indian law and policy, say so through the confidence "
    "fields, and never invent the names of acts, committees, or ministries."
)


def _metadata_instruction() -> str:
    return (
        "Read the bill above and extract its core metadata.\n\n"
        f"- category: the single best fit from this list: {', '.join(CATEGORIES)}.\n"
        "- secondary_categories: any other entries from the same list that clearly apply (may be empty).\n"
        "- ministry: the ministry or department responsible, if identifiable from the text; otherwise null.\n"
        "- objective: 2-3 plain-English sentences on what the bill is trying to achieve and why.\n"
        "- one_line_essence: one sentence a news ticker could run.\n"
        "- key_provisions: the 5-10 provisions that matter most, each with a short plain-English "
        "description and the clause/section reference when visible in the text.\n"
        "- stakeholders: the groups of people, institutions, and businesses the bill directly touches."
    )


def _chronology_instruction(title: str) -> str:
    return (
        f'Trace the legislative and policy history behind "{title}" using the bill text above.\n\n'
        "- policy_background: one paragraph on the history of the problem and why this bill exists now.\n"
        "- related_legislation: earlier Acts, bills, amendments, rules, or committee reports connected "
        "to this bill. Prefer legislation actually cited in the text (confidence \"high\"). You may add "
        "well-known Indian legislation or policy milestones from general knowledge with confidence "
        "\"medium\" or \"low\". Never invent names.\n"
        "- timeline: 4-10 chronological milestones (year, short label, one-line description) telling the "
        "policy evolution story that leads up to this bill."
    )


def _impact_instruction(title: str) -> str:
    return (
        f'Assess the practical impact of "{title}" using the bill text above.\n\n'
        f"- sector_impacts: the 3-8 sectors materially affected. Use sector names from this list where "
        f"possible: {', '.join(ppiia.SECTORS)}. For each give the direction for that sector "
        "(positive/negative/mixed/neutral), the magnitude (low/medium/high), a one-or-two-sentence "
        "rationale, concrete opportunities, and concrete restrictions or compliance burdens.\n"
        "- short_term (0-1 year), medium_term (1-5 years), long_term (more than 5 years): the most "
        "likely effects and the main risks in each window.\n"
        "- positives / negatives: 4-8 bullet points each, written for a general reader.\n"
        "- risk_flags: clauses or aspects that could raise costs, restrict rights, be hard to implement, "
        "or cause public or political friction — each with a severity and the groups affected.\n"
        "- controversy_level: your overall read (low/medium/high)."
    )


def _summary_instruction(title: str) -> str:
    return (
        f'Write the citizen-facing explanation of "{title}" using the bill text above.\n\n'
        "- plain_summary: 10-20 short sentences at an 8th-grade reading level, each sentence on its own "
        "line. No legal jargon; use everyday examples. Cover what the bill does, why it exists, who it "
        "affects, what changes for ordinary people, and when.\n"
        "- citizen_faq: 4-6 questions a normal person would actually ask (for example, \"Does this cost "
        "me money?\") with direct answers.\n"
        "- persona_impacts: pick the 3 everyday personas most relevant to this bill (for example, "
        "\"A farmer in Maharashtra\", \"A small business owner\", \"A college student\") and give one or "
        "two sentences on what changes for each."
    )


# ---------------------------------------------------------------------------
# Claude client
# ---------------------------------------------------------------------------
class ClaudeEngine:
    """Thin wrapper over the Anthropic SDK for schema-validated stage calls."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or get_api_key()
        if not key:
            raise AnalysisError(
                "No Anthropic API key found. Copy .env.example to .env and set "
                "ANTHROPIC_API_KEY, or paste a key in the app sidebar."
            )
        self.model = model or DEFAULT_MODEL
        self._client = anthropic.Anthropic(
            api_key=key, timeout=REQUEST_TIMEOUT, max_retries=3
        )

    def run_stage(self, bill_text: str, instruction: str, output_model: type):
        """One structured-output call. Returns a validated `output_model` instance."""
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"<bill>\n{bill_text}\n</bill>",
                                # identical first block across the four stages, so
                                # stages 2-4 read the bill text from the prompt cache
                                "cache_control": {"type": "ephemeral"},
                            },
                            {"type": "text", "text": instruction},
                        ],
                    }
                ],
                output_format=output_model,
            )
        except anthropic.AuthenticationError as e:
            raise AnalysisError(
                "The Anthropic API rejected the key. Check ANTHROPIC_API_KEY."
            ) from e
        except anthropic.RateLimitError as e:
            raise AnalysisError(
                "Rate-limited by the Anthropic API. Wait a minute and try again."
            ) from e
        except anthropic.APIConnectionError as e:
            raise AnalysisError(
                "Could not reach the Anthropic API — check your internet connection."
            ) from e
        except anthropic.APIStatusError as e:
            raise AnalysisError(f"Anthropic API error ({e.status_code}): {e.message}") from e

        if response.stop_reason == "refusal":
            raise AnalysisError(
                "The model declined to analyze this document. Verify it is a "
                "legislative text and try again."
            )
        if response.parsed_output is None:
            raise AnalysisError(
                f"The model response was incomplete (stop_reason={response.stop_reason}). "
                "Try again, or use a shorter document."
            )
        return response.parsed_output


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def analyze_with_claude(
    raw_text: str,
    source: SourceInfo | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    engine: ClaudeEngine | None = None,
    progress: Callable[[str], None] | None = None,
) -> BillAnalysis:
    """Run the full four-stage Claude analysis over raw bill text."""
    notify = progress or (lambda msg: None)
    engine = engine or ClaudeEngine(api_key=api_key, model=model)

    text, truncated = prepare_text(raw_text)
    if len(text) < 200:
        raise AnalysisError(
            "The extracted text is too short to analyze (under 200 characters). "
            "If this is a scanned PDF it needs OCR first."
        )
    if source is None:
        source = build_source_info(text, "text", "Pasted text")
    source.truncated = source.truncated or truncated

    started = time.perf_counter()

    notify("Stage 1/4 — Classifying the bill and extracting key provisions…")
    meta: LLMMetadata = engine.run_stage(text, _metadata_instruction(), LLMMetadata)

    notify("Stage 2/4 — Tracing legislative history and related acts…")
    chrono: LLMChronology = engine.run_stage(
        text, _chronology_instruction(meta.title), LLMChronology
    )

    notify("Stage 3/4 — Assessing sector impact, forecast, and risks…")
    impact: LLMImpact = engine.run_stage(text, _impact_instruction(meta.title), LLMImpact)

    notify("Stage 4/4 — Writing the citizen-friendly summary…")
    summary: LLMSummary = engine.run_stage(text, _summary_instruction(meta.title), LLMSummary)

    elapsed = time.perf_counter() - started
    notify(f"Done in {elapsed:.0f} s — assembling the report…")

    return _assemble(meta, chrono, impact, summary, source, engine.model, elapsed)


def _assemble(
    meta: LLMMetadata,
    chrono: LLMChronology,
    impact: LLMImpact,
    summary: LLMSummary,
    source: SourceInfo,
    model: str,
    elapsed: float,
) -> BillAnalysis:
    analysis = BillAnalysis(
        title=meta.title,
        ministry=meta.ministry,
        category=meta.category,
        secondary_categories=meta.secondary_categories,
        objective=meta.objective,
        one_line_essence=meta.one_line_essence,
        key_provisions=[p.model_dump() for p in meta.key_provisions],
        stakeholders=meta.stakeholders,
        policy_background=chrono.policy_background,
        related_legislation=[r.model_dump() for r in chrono.related_legislation],
        timeline=[t.model_dump() for t in chrono.timeline],
        sector_impacts=[s.model_dump() for s in impact.sector_impacts],
        term_impacts=TermImpacts(
            short_term=impact.short_term.model_dump(),
            medium_term=impact.medium_term.model_dump(),
            long_term=impact.long_term.model_dump(),
        ),
        positives=impact.positives,
        negatives=impact.negatives,
        risk_flags=[r.model_dump() for r in impact.risk_flags],
        controversy_level=impact.controversy_level,
        citizen_faq=[f.model_dump() for f in summary.citizen_faq],
        persona_impacts=[p.model_dump() for p in summary.persona_impacts],
        plain_summary=summary.plain_summary,
        source=source,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )
    analysis.metrics = compute_metrics(
        engine="claude",
        model=model,
        processing_seconds=elapsed,
        input_words=source.word_count,
        summary_text=analysis.plain_summary,
    )
    return analysis


def compute_metrics(
    *,
    engine: str,
    model: str,
    processing_seconds: float | None,
    input_words: int | None,
    summary_text: str,
) -> AnalysisMetrics:
    """Evaluation metrics: readability of the citizen summary + throughput."""
    ease = grade = None
    summary_words = len(summary_text.split()) if summary_text else 0
    if summary_words >= 10:
        try:
            ease = round(float(textstat.flesch_reading_ease(summary_text)), 1)
            grade = round(float(textstat.flesch_kincaid_grade(summary_text)), 1)
        except Exception:
            pass
    ratio = None
    if input_words and summary_words:
        ratio = round(input_words / summary_words, 1)
    return AnalysisMetrics(
        engine=engine,
        model=model,
        processing_seconds=round(processing_seconds, 2) if processing_seconds is not None else None,
        flesch_reading_ease=ease,
        readability_grade=grade,
        input_words=input_words,
        summary_words=summary_words or None,
        compression_ratio=ratio,
    )


# ---------------------------------------------------------------------------
# Precomputed demo
# ---------------------------------------------------------------------------
def load_sample_analysis() -> BillAnalysis:
    """Load the precomputed analysis of sample_bill.pdf (no API key needed)."""
    data = json.loads(SAMPLE_ANALYSIS_PATH.read_text(encoding="utf-8"))
    analysis = BillAnalysis.model_validate(data)
    if analysis.metrics is None:
        analysis.metrics = compute_metrics(
            engine="precomputed demo",
            model="bundled sample analysis",
            processing_seconds=None,
            input_words=analysis.source.word_count if analysis.source else None,
            summary_text=analysis.plain_summary,
        )
    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_source(src: str) -> tuple[str, SourceInfo]:
    """Resolve a CLI source argument to (raw_text, SourceInfo)."""
    if src.lower() == "sample":
        raw = ppiia.extract_text_from_pdf(str(SAMPLE_PDF_PATH))
        return raw, build_source_info(raw, "sample", SAMPLE_PDF_PATH.name)
    if src.lower().startswith(("http://", "https://")):
        raw = ppiia.extract_text_from_url(src)
        return raw, build_source_info(raw, "url", src)
    path = Path(src)
    if path.exists() and path.suffix.lower() == ".pdf":
        raw = ppiia.extract_text_from_pdf(str(path))
        return raw, build_source_info(raw, "pdf", path.name)
    if path.exists():
        raw = path.read_text(encoding="utf-8", errors="replace")
        return raw, build_source_info(raw, "text", path.name)
    return src, build_source_info(src, "text", "Command-line text")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPIIA Claude engine — analyze a bill from PDF/URL/text"
    )
    parser.add_argument("source", help='"sample", a PDF path, a URL, or a text file')
    parser.add_argument("--model", default=None, help=f"Claude model id (default {DEFAULT_MODEL})")
    parser.add_argument("--out", default="outputs/analysis.json", help="JSON output path")
    parser.add_argument("--pdf", default=None, help="Also write a PDF report to this path")
    args = parser.parse_args()

    raw, source = _load_source(args.source)
    analysis = analyze_with_claude(raw, source, model=args.model, progress=print)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nJSON report saved to {out}")

    if args.pdf:
        from report_pdf import build_pdf_report

        pdf_path = Path(args.pdf)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(build_pdf_report(analysis))
        print(f"PDF report saved to {pdf_path}")

    print(f"\n=== {analysis.title} ===")
    print(analysis.one_line_essence)
    print(f"\nCategory: {analysis.category} | Ministry: {analysis.ministry}")
    print(f"Readability grade: {analysis.metrics.readability_grade if analysis.metrics else '—'}")
    print(f"\n{analysis.plain_summary[:600]}")


if __name__ == "__main__":
    main()
