"""Source-grounded, conservative local summaries and topic evidence.

Extracted wording is not a legal interpretation and may contain PDF reading
order errors. The system deliberately abstains when evidence is insufficient.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TopicEvidence:
    sector: str
    matched_phrase: str
    excerpt: str


# Specific phrases are intentionally preferred to generic words such as
# "government", "business", "bank" or "law", which occur in many Bills.
TOPIC_TERMS = {
    "Data and technology": ("digital personal data", "personal data", "data fiduciary", "data protection", "cybersecurity", "cyber security", "telecommunications"),
    "Environment and water": ("water pollution", "pollution control", "control of pollution", "environmental protection", "water (prevention and control of pollution)", "effluent", "environmental impact"),
    "Social welfare and tribal affairs": ("scheduled tribes", "scheduled tribe", "scheduled castes", "social welfare", "social security", "tribal community", "disability benefits"),
    "Agriculture": ("agriculture", "agricultural land", "farmers", "crop production", "agricultural produce"),
    "Finance and insurance": ("banking regulation", "banking companies", "financial services", "securities market", "insurance business", "insurance companies"),
    "Health": ("public health", "health care", "healthcare", "hospitals", "medical practitioners", "pharmaceutical"),
    "Education": ("higher education", "school education", "universities", "educational institutions", "teachers"),
    "Employment and labour": ("labour rights", "labor rights", "employment conditions", "working conditions", "industrial disputes", "workers' rights", "employees"),
    "Energy": ("electricity", "power generation", "renewable energy", "petroleum", "oilfields", "natural gas"),
    "Transport": ("railways", "railway", "national highways", "road transport", "air carrier", "air transport"),
}


def normalize(text: str) -> str:
    # Preserve wording; only normalize whitespace and remove standalone
    # page/line numbers that commonly appear in parliamentary PDFs.
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stated_purpose(text: str) -> str:
    match = re.search(r"\bA\s+BILL\b\s*(.*?)(?=\bBE\s+it\s+enacted\b)", text,
                      flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    section = normalize(match.group(1))
    # Preserve the first sentence of the Bill's own purpose. Include one
    # substantive WHEREAS sentence when it explains why it is introduced.
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z])", section)
    if not pieces:
        return ""
    first = pieces[0].strip()
    if len(first) < 15 or len(first) > 1000:
        return ""
    reason = next((p.strip() for p in pieces[1:] if re.search(
        r"\bWHEREAS\b.*\b(?:necessary|purpose|decriminalis|rationalis)", p, re.I)), "")
    return (first + (" " + reason if reason and len(reason) < 700 else "")).strip()


def operative_body(text: str) -> str:
    start = re.search(r"\bBE\s+it\s+enacted\b", text, re.IGNORECASE)
    body = text[start.end():] if start else text
    end = re.search(r"\b(?:STATEMENT\s+OF\s+OBJECTS\s+AND\s+REASONS|ANNEXURE|FINANCIAL\s+MEMORANDUM)\b", body, re.IGNORECASE)
    return body[:end.start()] if end else body


def sentences(text: str) -> list[str]:
    normalized = normalize(text)
    # Legal Bills contain clauses separated by periods, semicolons and
    # numbered section starts. Long quotations are kept together when possible.
    parts = re.split(r"(?<=[.!?;])\s+(?=(?:[A-Z(“‘]|\d+\.\s))", normalized)
    return [p.strip() for p in parts if 45 <= len(p.strip()) <= 1500]


def without_illustrations(text: str) -> str:
    """Exclude labeled examples when choosing provisions or topic evidence."""
    return re.sub(
        r"(?ims)^\s*Illustrations?\s*\.\s*\n.*?(?=^\s*(?:\(\d+\)|\d+\.\s*\(\d+\)|CHAPTER\b)|\Z)",
        "\n", text,
    )


def selected_passages(text: str, limit: int = 4) -> list[str]:
    ranked = []
    for index, passage in enumerate(sentences(without_illustrations(operative_body(text)))):
        lower = passage.lower()
        score = sum(weight for phrase, weight in (
            ("shall be substituted", 10), ("shall be inserted", 10),
            ("shall be omitted", 9), ("shall be amended", 9),
            ("for the words", 7), ("following section", 6),
            ("personal data", 2), ("consent", 3), ("right to", 4),
            ("penalty", 3), ("prohibited", 4), ("shall be liable", 3),
            ("may process", 4), ("shall not", 3),
            ("exempt certain categories", 6), ("withdraw her consent", 4),
            ("right to access", 4), ("right to correction", 4),
        ) if phrase in lower)
        if re.search(r"\b(?:short title|shall come into force|commencement)\b", lower):
            score -= 5
        if re.search(r"(?:namely|following section).{0,20}\b\d+[a-z]?\.$", lower):
            continue
        if score > 0:
            if "exempt" in lower:
                kind = "exemption"
            elif "penalty" in lower:
                kind = "penalty"
            elif "right to" in lower or "withdraw her consent" in lower:
                kind = "rights"
            elif "notice" in lower:
                kind = "notice"
            elif "consent" in lower:
                kind = "consent"
            else:
                kind = "provision"
            ranked.append((score, index, passage, kind))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected, kinds = [], set()
    token_sets = []
    # First select distinct types of provision; then fill any remaining slots.
    for distinct in (True, False):
        for _, _, passage, kind in ranked:
            if passage in selected or (distinct and kind in kinds):
                continue
            words = set(re.findall(r"[a-z]{4,}", passage.lower()))
            if any(len(words & seen) / max(1, len(words | seen)) > 0.7 for seen in token_sets):
                continue
            selected.append(passage)
            token_sets.append(words)
            kinds.add(kind)
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    return selected


def _phrase_pattern(phrase: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?!\w)", re.IGNORECASE)


def identify_topics(text: str, title: str, purpose: str, limit: int = 4) -> list[TopicEvidence]:
    priority = normalize(title + ". " + purpose)
    body = normalize(without_illustrations(operative_body(text)))
    body_sentences = sentences(body)
    found = []
    for sector, phrases in TOPIC_TERMS.items():
        matches = []
        for phrase in phrases:
            pattern = _phrase_pattern(phrase)
            priority_match = pattern.search(priority)
            body_matches = list(pattern.finditer(body))
            if priority_match or len(body_matches) >= 2:
                score = (10 if priority_match else 0) + min(len(body_matches), 5) + min(len(phrase), 30) / 100
                matches.append((score, phrase, pattern, bool(priority_match)))
        if not matches:
            continue
        score, phrase, pattern, from_priority = max(matches, key=lambda item: item[0])
        # Cite an actual source sentence. A title/purpose is valid evidence of
        # a topic but not evidence that a sector benefits or loses.
        excerpt = next((s for s in body_sentences if pattern.search(s) and len(s) <= 600), "")
        if from_priority and pattern.search(title):
            excerpt = title
        elif from_priority and pattern.search(purpose):
            excerpt = purpose
        if not excerpt:
            match = pattern.search(body)
            if match:
                excerpt = body[max(0, match.start() - 120):min(len(body), match.end() + 220)].strip()
        if excerpt:
            found.append((score, TopicEvidence(sector=sector, matched_phrase=phrase, excerpt=excerpt)))
    return [item for _, item in sorted(found, key=lambda x: (-x[0], x[1].sector))[:limit]]


def build_source_summary(text: str, passages: list[str]) -> str:
    purpose = stated_purpose(text)
    parts = []
    if purpose:
        parts.append("Stated purpose (source wording): " + purpose)
    if passages:
        parts.append("Selected provision (source wording): " + passages[0])
    if not parts:
        return "No sufficiently clear purpose or provision was identified automatically. Read the original Bill."
    return "\n\n".join(parts)
