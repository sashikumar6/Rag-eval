"""
__init__.py — Metrics package for legal-rag-eval.
"""

from metrics.citation_accuracy import (
    CitationAccuracyResult,
    CitationMatchResult,
    ParsedCitation,
    aggregate_citation_accuracy,
    compute_citation_accuracy,
    extract_citations_from_text,
)
from metrics.hallucination import (
    HallucinationResult,
    aggregate_hallucination,
    compute_hallucination,
)
from metrics.answer_relevancy import (
    AnswerRelevancyResult,
    aggregate_answer_relevancy,
    compute_answer_relevancy,
)

__all__ = [
    # Citation accuracy
    "ParsedCitation",
    "CitationMatchResult",
    "CitationAccuracyResult",
    "extract_citations_from_text",
    "compute_citation_accuracy",
    "aggregate_citation_accuracy",
    # Hallucination
    "HallucinationResult",
    "compute_hallucination",
    "aggregate_hallucination",
    # Answer relevancy
    "AnswerRelevancyResult",
    "compute_answer_relevancy",
    "aggregate_answer_relevancy",
]
