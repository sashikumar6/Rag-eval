"""
source_attribution.py — Measures whether the RAG response attributes facts to sources.

Applies to case_law and cross_source questions. For each sentence in the response
we check whether it contains an attribution signal (USC citation, CFR citation, or
case law citation). Score = attributed sentences / total sentences (excluding
boilerplate opening/closing sentences).

See DECISIONS.md ADR-017.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SourceAttributionResult:
    """Attribution result for a single question."""
    question_id: str
    score: float
    attributed: int
    total: int
    missing_attributions: list[str] = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Attribution signal patterns
# ---------------------------------------------------------------------------

# Federal statute: 18 U.S.C. § 1344, 42 USC 2000e
_FEDERAL_SIGNAL = re.compile(
    r"\d{1,2}\s*U\.?S\.?C\.?\s*§?\s*[\d\w\-]+",
    re.IGNORECASE,
)

# CFR: 26 C.F.R. § 1.401, 29 CFR 541
_CFR_SIGNAL = re.compile(
    r"\d{1,2}\s*C\.?F\.?R\.?\s*§?\s*[\d\w\-\.]+",
    re.IGNORECASE,
)

# Case law citation: "Name v. Name, NNN U.S. NNN" or reporter references
_CASE_LAW_SIGNAL = re.compile(
    r"""
    (
        \w[\w\s]+\s+v\.\s+\w[\w\s]+,\s*\d+     # Party v. Party, NNN
        |
        \d+\s+(?:U\.S\.|F\.\d+d|F\.\d+th|S\.\s*Ct\.|F\.\s*Supp\.)\s+\d+  # reporter citation
        |
        (?:the\s+)?[Cc]ourt\s+held              # "the Court held"
        |
        (?:the\s+)?[Cc]ourt\s+(?:ruled|found|stated|noted|observed|wrote)
        |
        [Jj]ustice\s+\w+\s+wrote
        |
        [Cc]hief\s+[Jj]ustice\s+\w+
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Any section symbol (§) used as a general legal reference
_SECTION_SIGNAL = re.compile(r"§\s*[\d\w\-]+")

# Boilerplate sentence starters to exclude from "fact" count
_BOILERPLATE_PREFIXES = (
    "in summary",
    "to summarize",
    "in conclusion",
    "in short",
    "therefore,",
    "thus,",
    "accordingly,",
    "as a result,",
    "based on the above",
    "based on the foregoing",
)


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for attribution analysis."""
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\(])", text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]


def _is_boilerplate(sentence: str) -> bool:
    low = sentence.lower().strip()
    return any(low.startswith(prefix) for prefix in _BOILERPLATE_PREFIXES)


def _has_attribution(sentence: str) -> bool:
    """Return True if the sentence contains any attribution signal."""
    return bool(
        _FEDERAL_SIGNAL.search(sentence)
        or _CFR_SIGNAL.search(sentence)
        or _CASE_LAW_SIGNAL.search(sentence)
        or _SECTION_SIGNAL.search(sentence)
    )


# ---------------------------------------------------------------------------
# Main metric
# ---------------------------------------------------------------------------

def compute_source_attribution(
    question_id: str,
    rag_response_text: str,
    source_type: str = "case_law",
) -> SourceAttributionResult:
    """
    Compute source attribution score for a single question.

    Args:
        question_id: ID from golden set
        rag_response_text: Full answer text from the RAG
        source_type: "case_law" | "cross_source" — affects expected attribution signals

    Returns:
        SourceAttributionResult with score in [0.0, 1.0]
    """
    if not rag_response_text or not rag_response_text.strip():
        return SourceAttributionResult(
            question_id=question_id,
            score=0.0,
            attributed=0,
            total=0,
            missing_attributions=["empty response"],
            breakdown={"reason": "empty response"},
        )

    sentences = _split_sentences(rag_response_text)
    # Filter out boilerplate
    fact_sentences = [s for s in sentences if not _is_boilerplate(s)]

    if not fact_sentences:
        return SourceAttributionResult(
            question_id=question_id,
            score=0.0,
            attributed=0,
            total=0,
            missing_attributions=["no substantive sentences found"],
            breakdown={"reason": "no substantive sentences found"},
        )

    attributed = [s for s in fact_sentences if _has_attribution(s)]
    unattributed = [s for s in fact_sentences if not _has_attribution(s)]

    total = len(fact_sentences)
    attr_count = len(attributed)
    score = round(attr_count / total, 4) if total > 0 else 0.0

    # Collect short excerpts of unattributed sentences for the report
    missing = [s[:120] + "..." if len(s) > 120 else s for s in unattributed[:5]]

    return SourceAttributionResult(
        question_id=question_id,
        score=score,
        attributed=attr_count,
        total=total,
        missing_attributions=missing,
        breakdown={
            "source_type": source_type,
            "total_sentences": total,
            "attributed_sentences": attr_count,
            "unattributed_count": len(unattributed),
        },
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate_source_attribution(results: list[SourceAttributionResult]) -> dict:
    """
    Aggregate source attribution scores across questions that ran this metric.

    Returns:
        {
            "score": float,          # mean score
            "total_evaluated": int,
            "per_question": [...]
        }
    """
    if not results:
        return {"score": 0.0, "total_evaluated": 0, "per_question": []}

    scores = [r.score for r in results]
    mean_score = sum(scores) / len(scores)

    return {
        "score": round(mean_score, 4),
        "total_evaluated": len(results),
        "per_question": [
            {
                "question_id": r.question_id,
                "score": r.score,
                "attributed": r.attributed,
                "total": r.total,
                "missing_attributions": r.missing_attributions,
            }
            for r in results
        ],
    }
