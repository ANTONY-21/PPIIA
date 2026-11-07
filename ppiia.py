"""
Public Policy Insight & Impact Analyzer (PPIIA)
===============================================
Takes a government Bill (PDF / URL / text) and produces a citizen-friendly,
structured analysis:

    {
        "title":        "<bill name>",
        "ministry":     "<owning ministry>",
        "objective":    "<one-paragraph purpose>",
        "summary":      "<8th-grade friendly 10-20 line summary>",
        "sectors":      [ ... top affected industries ... ],
        "key_entities": { "dates": [...], "acts_cited": [...], "orgs": [...] },
        "chronology":   [ ... previous acts / amendments found in the text ... ],
        "impacts": {
            "short_term":  "...",
            "medium_term": "...",
            "long_term":   "...",
        },
        "positives": [ ... ],
        "negatives": [ ... ],
        "risks":     [ ... ],
    }

LOCAL AI ONLY — no API keys, no paid services, no internet after first
model download. Uses Hugging Face transformers running on CPU:

    - sshleifer/distilbart-cnn-12-6     (abstractive summarization)
    - valhalla/distilbart-mnli-12-3     (zero-shot sector classification)

Run from terminal:
    pip install -r requirements.txt
    python ppiia.py path/to/bill.pdf
"""

from __future__ import annotations

import os
import re
import json
import argparse
from dataclasses import dataclass, asdict, field
import evidence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SECTORS = [
    "Agriculture", "Finance", "Education", "Healthcare",
    "Technology", "Environment", "Defence", "Labour",
    "Energy", "Infrastructure", "Justice and Law", "Social Welfare",
    "Taxation", "Trade and Commerce", "Transport",
]

# Chunk long documents to fit inside model context windows
CHUNK_TOKENS = 900          # ~words, rough proxy
SUMMARY_MAX_LEN = 120       # generated tokens per chunk
SUMMARY_MIN_LEN = 15


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
@dataclass
class BillReport:
    title: str = ""
    ministry: str = ""
    objective: str = ""
    summary: str = ""
    source_passages: list = field(default_factory=list)
    topic_evidence: list = field(default_factory=list)
    ai_draft: str = ""
    sectors: list = field(default_factory=list)
    key_entities: dict = field(default_factory=dict)
    chronology: list = field(default_factory=list)
    impacts: dict = field(default_factory=dict)
    positives: list = field(default_factory=list)
    negatives: list = field(default_factory=list)
    risks: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 1. Input ingestion
# ---------------------------------------------------------------------------
def extract_text_from_pdf(path: str) -> str:
    """Extract text in reading order, falling back to pdfplumber if needed."""
    print(f"[1] Extracting text from PDF: {path}")
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        text = ""
    if len(text.strip()) < 200:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)

    print(f"    -> {len(text):,} characters extracted")
    return text


def extract_text_from_url(url: str) -> str:
    """
    Fetch a bill from a URL. Supports both direct PDF links and HTML pages
    (e.g. Sansad portal). Uses requests + BeautifulSoup for HTML.
    """
    print(f"[1] Fetching from URL: {url}")
    import requests
    resp = requests.get(url, timeout=30, headers={"User-Agent": "PPIIA/1.0"})
    resp.raise_for_status()

    ctype = resp.headers.get("content-type", "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(resp.content)
            tmp = f.name
        try:
            return extract_text_from_pdf(tmp)
        finally:
            os.unlink(tmp)

    # Assume HTML
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    if "/knowledge-centre/government-bills" in url.lower():
        raise ValueError("This URL is a list of bills, not a bill document. Open a specific bill and copy its PDF link, or upload its PDF.")
    print(f"    -> {len(text):,} characters extracted")
    return text


# ---------------------------------------------------------------------------
# 2. Text cleaning + chunking
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
_PAGE_NUM_RE = re.compile(r"\n\s*\d+\s*\n")
_HEADER_RE = re.compile(r"(Page\s+\d+\s+of\s+\d+)", re.I)


def clean_text(text: str) -> str:
    """Normalise whitespace, strip page numbers and headers."""
    text = _PAGE_NUM_RE.sub("\n", text)
    text = _HEADER_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def chunk_text(text: str, max_words: int = CHUNK_TOKENS) -> list[str]:
    """
    Split text into chunks of ~max_words words each. Simple word-based
    splitter so we don't need a tokenizer. Keeps paragraph boundaries.
    """
    paragraphs = text.split(". ")
    chunks, current, current_len = [], [], 0
    for p in paragraphs:
        n = len(p.split())
        if current_len + n > max_words and current:
            chunks.append(". ".join(current) + ".")
            current, current_len = [], 0
        current.append(p)
        current_len += n
    if current:
        chunks.append(". ".join(current))
    return chunks


# ---------------------------------------------------------------------------
# 3. Entity extraction (regex)
# ---------------------------------------------------------------------------
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_ACT_RE = re.compile(r"\b([A-Z][A-Za-z,\s]+?(?:Act|Bill|Ordinance)(?:\s*,?\s*\d{4})?)\b")
_MINISTRY_RE = re.compile(r"Ministry of [A-Z][A-Za-z &]+?(?=[.,\n])")


def extract_entities(text: str) -> dict:
    """Pull out obvious entities with regex. Works without any ML model."""
    years = sorted(set(m.group() for m in _YEAR_RE.finditer(text)))
    acts = sorted({m.strip() for m in _ACT_RE.findall(text)})[:20]
    ministries = sorted({m.strip() for m in _MINISTRY_RE.findall(text)})
    return {
        "years": years[:15],
        "acts_cited": acts,
        "ministries": ministries,
    }


def extract_key_passages(text: str, limit: int = 4) -> list[str]:
    """Select source passages for review; extraction is not exhaustive."""
    return evidence.selected_passages(text, limit)


# ---------------------------------------------------------------------------
# 4. Title / ministry heuristics
# ---------------------------------------------------------------------------
def guess_title(text: str) -> str:
    """
    First reasonable-looking sentence that mentions 'Bill' or 'Act'.
    Falls back to the first non-empty line.
    """
    # Parliamentary PDFs often break the heading across several lines.
    match = re.search(
        r"\bTHE\s+([A-Z][A-Z0-9\s(),&'’\-]{5,180}?\bBILL\s*,?\s*20\d{2})\b",
        text[:12000], re.IGNORECASE,
    )
    if match:
        return "The " + " ".join(match.group(1).split()).title()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for ln in lines[:40]:
        if "bill" in ln.lower() or "act" in ln.lower():
            if 5 < len(ln.split()) < 25:
                return ln
    first = lines[0] if lines else ""
    return first if 5 <= len(first) <= 160 else "Untitled Bill"


def guess_ministry(text: str) -> str:
    m = _MINISTRY_RE.search(text)
    return m.group().strip() if m else ""


def llm_available() -> str:
    """Kept for backwards-compat with the Streamlit / Gradio UIs.
    This project is LOCAL-ONLY, so this always returns 'free'."""
    return "free"


# ---------------------------------------------------------------------------
# 5. Local-AI analysis (Hugging Face models, called directly)
# ---------------------------------------------------------------------------
# NOTE: We deliberately avoid transformers.pipeline() because some Colab
# environments ship a broken pipeline task registry. Calling the models
# directly via AutoTokenizer/AutoModel is more robust across versions.
_sum_tok = None
_sum_model = None
_mnli_tok = None
_mnli_model = None


def _load_hf_models():
    """Lazy-load HF models directly (first call downloads the weights)."""
    global _sum_tok, _sum_model, _mnli_tok, _mnli_model
    import torch  # noqa: F401
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
    )

    model_root = os.getenv("PPIIA_MODELS_DIR")
    local_summary = os.path.join(model_root, "summarizer") if model_root else None
    local_classifier = os.path.join(model_root, "classifier") if model_root else None
    if model_root and not all(os.path.isfile(os.path.join(p, "pytorch_model.bin")) for p in (local_summary, local_classifier)):
        raise RuntimeError("The model directory is incomplete. PPIIA_MODELS_DIR must contain summarizer and classifier checkpoints, or be unset to use the Hugging Face cache.")
    summary_source = local_summary if local_summary and os.path.isfile(os.path.join(local_summary, "pytorch_model.bin")) else "sshleifer/distilbart-cnn-12-6"
    classifier_source = local_classifier if local_classifier and os.path.isfile(os.path.join(local_classifier, "pytorch_model.bin")) else "valhalla/distilbart-mnli-12-3"

    if _sum_model is None:
        print("[HF] loading summarizer (sshleifer/distilbart-cnn-12-6) ...")
        _sum_tok = AutoTokenizer.from_pretrained(summary_source)
        _sum_model = AutoModelForSeq2SeqLM.from_pretrained(
            summary_source
        )
        _sum_model.eval()

    if _mnli_model is None:
        print("[HF] loading zero-shot NLI model (valhalla/distilbart-mnli-12-3) ...")
        _mnli_tok = AutoTokenizer.from_pretrained(classifier_source)
        _mnli_model = AutoModelForSequenceClassification.from_pretrained(
            classifier_source
        )
        _mnli_model.eval()

    return _sum_tok, _sum_model, _mnli_tok, _mnli_model


def _summarize_chunk(text: str, max_len: int, min_len: int) -> str:
    """Single-chunk summarization via direct model.generate()."""
    import torch
    tok, model, _, _ = _load_hf_models()
    inputs = tok(
        text, return_tensors="pt", truncation=True, max_length=1024
    )
    input_length = inputs["input_ids"].shape[-1]
    max_len = max(20, min(max_len, max(20, input_length // 2)))
    min_len = min(min_len, max_len - 1)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_length=max_len,
            min_length=min_len,
            num_beams=2,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    result = tok.decode(out_ids[0], skip_special_tokens=True).strip()
    # Generation can reach its token limit mid-sentence. Keep completed
    # sentences where possible rather than presenting a dangling fragment.
    if result and result[-1] not in ".!?":
        endings = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", result)]
        if endings:
            result = result[:endings[-1]].strip()
    return result


def summarize_long_text(text: str) -> str:
    """
    Map-reduce summarization:
      1. Split into chunks.
      2. Summarise each chunk.
      3. Join summaries and summarise again (reduce step).
    """
    tokenizer, _, _, _ = _load_hf_models()
    # Split by model tokens, not words, so legal terms and numbers are not
    # silently dropped when the tokenizer expands them into multiple tokens.
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    token_chunks = [ids[i:i + 850] for i in range(0, len(ids), 850)]
    chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in token_chunks]
    print(f"[HF] map-reduce summarization over {len(chunks)} chunks ...")

    partials = []
    for i, ch in enumerate(chunks):
        if not ch.strip():
            continue
        s = _summarize_chunk(ch, SUMMARY_MAX_LEN, SUMMARY_MIN_LEN)
        partials.append(s)
        print(f"    chunk {i+1}/{len(chunks)}  ->  {len(s)} chars")

    joined = " ".join(partials)
    if len(joined.split()) < 200:
        return joined

    # Recursively reduce instead of truncating a long combined summary.
    for _ in range(10):
        joined_ids = tokenizer.encode(joined, add_special_tokens=False, truncation=False)
        if len(joined_ids) <= 850:
            return _summarize_chunk(joined, 300, 60)
        groups = [joined_ids[i:i + 850] for i in range(0, len(joined_ids), 850)]
        reduced = [_summarize_chunk(tokenizer.decode(g, skip_special_tokens=True), 220, 40)
                   for g in groups]
        joined = " ".join(reduced)
    raise ValueError("The document is too large to summarize reliably in local mode.")


def classify_sectors_hf(text: str, top_k: int = 5) -> list:
    """
    Zero-shot classification via direct NLI scoring.
    For each candidate sector we build the hypothesis
    'This text is about {sector}.' and take the entailment probability.
    """
    import torch
    _, _, tok, model = _load_hf_models()

    premise = " ".join(text.split()[:500])
    scores = {}
    for sector in SECTORS:
        hypothesis = f"This text is about {sector}."
        enc = tok(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=512,
        )
        with torch.no_grad():
            logits = model(**enc).logits[0]
        # MNLI labels: 0=contradiction, 1=neutral, 2=entailment
        # Use sigmoid on entailment vs contradiction for multi-label scoring
        ent_contra = logits[[0, 2]]
        probs = torch.softmax(ent_contra, dim=-1)
        scores[sector] = float(probs[1])  # entailment probability

    pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [lbl for lbl, s in pairs[:top_k] if s > 0.5]


def _simple_impact_bullets(summary: str) -> dict:
    """
    Lightweight heuristic to produce short/medium/long-term strings from
    the summary. Splits the summary into thirds and labels them.
    """
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
    if not sents:
        return {"short_term": "", "medium_term": "", "long_term": ""}
    n = len(sents)
    return {
        "short_term":  " ".join(sents[: max(1, n // 3)]),
        "medium_term": " ".join(sents[n // 3: max(2, 2 * n // 3)]),
        "long_term":   " ".join(sents[2 * n // 3:]) or sents[-1],
    }


def analyse_with_hf(text: str) -> dict:
    """Local-mode analyser using only open HuggingFace models."""
    summary = summarize_long_text(text)
    sectors = classify_sectors_hf(text)
    return {
        "title": "",
        "ministry": "",
        "objective": "",
        "summary": summary,
        "sectors": sectors,
        "chronology": [],
        "impacts": {},
        "positives": [],
        "negatives": [],
        "risks": [],
    }


# ---------------------------------------------------------------------------
# 6. Top-level pipeline
# ---------------------------------------------------------------------------
def analyze(source: str, use_ai_draft: bool = False) -> BillReport:
    """
    Main entrypoint. `source` can be:
        - a path to a PDF
        - an http(s) URL
        - raw text (when neither of the above)
    Returns a BillReport dataclass.
    """
    if source.lower().startswith(("http://", "https://")):
        raw = extract_text_from_url(source)
    elif os.path.exists(source) and source.lower().endswith(".pdf"):
        raw = extract_text_from_pdf(source)
    else:
        raw = source  # assume raw text

    text = clean_text(raw)
    if len(text) < 200:
        raise ValueError("The document has too little readable text. If the PDF is scanned, run OCR first or paste the English text.")

    print(f"[2] Cleaned text length = {len(text):,} chars, "
          f"{len(text.split()):,} words")

    # Regex-based quick wins
    entities = extract_entities(text)
    title_guess = guess_title(raw)
    ministry_guess = guess_ministry(raw)

    print("[3] Selecting source-grounded passages and topic evidence")
    passages = extract_key_passages(raw)
    purpose = evidence.stated_purpose(raw)
    topic_matches = evidence.identify_topics(raw, title_guess, purpose)
    ai_draft = summarize_long_text(text) if use_ai_draft else ""

    report = BillReport(
        title=title_guess,
        ministry=ministry_guess,
        objective="",
        summary=evidence.build_source_summary(raw, passages),
        source_passages=passages,
        topic_evidence=[asdict(item) for item in topic_matches],
        ai_draft=ai_draft,
        sectors=[item.sector for item in topic_matches],
        key_entities=entities,
        chronology=entities.get("acts_cited", []),
        impacts={}, positives=[], negatives=[], risks=[],
    )
    return report


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Public Policy Insight & Impact Analyzer (local AI)")
    parser.add_argument("source",
                        help="Path to a PDF or URL or a raw text file")
    parser.add_argument("--out", default="outputs/report.json",
                        help="Where to write the JSON report")
    args = parser.parse_args()

    report = analyze(args.source)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"\nDONE - report saved to {args.out}")
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
