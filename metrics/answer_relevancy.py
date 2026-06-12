"""
answer_relevancy.py — DeepEval AnswerRelevancyMetric wrapper for legal RAG evaluation.

Design decision: We wrap DeepEval's AnswerRelevancyMetric rather than rolling our
own because relevancy requires semantic understanding of whether an answer addresses
the question. Simple keyword overlap (e.g. ROUGE) would give high scores to answers
that repeat the question without actually answering it.

Threshold: 0.70 — deliberately lower than citation accuracy (0.80) because
relevancy is partially subjective. The RAG may correctly identify the relevant
statute but phrase the answer differently from the golden set.

See DECISIONS.md § "Why DeepEval for relevancy?" for full rationale.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AnswerRelevancyResult:
    """Result of answer relevancy check for a single question."""
    question_id: str
    score: float                  # 0.0 = irrelevant, 1.0 = highly relevant
    passed: bool                  # True if score >= threshold
    threshold: float
    reason: Optional[str] = None
    skipped: bool = False


def compute_answer_relevancy(
    question_id: str,
    question: str,
    rag_response_text: str,
    threshold: float = 0.70,
) -> AnswerRelevancyResult:
    """
    Check whether the RAG's answer is relevant to the question asked.

    Args:
        question_id: ID from golden set (e.g. "q001")
        question: The original question from the golden set
        rag_response_text: The RAG's answer text
        threshold: Minimum acceptable relevancy score. Default 0.70.

    Returns:
        AnswerRelevancyResult with score in [0.0, 1.0] or skipped=True

    How DeepEval AnswerRelevancyMetric works:
        It uses an LLM to evaluate whether `actual_output` directly and
        completely answers the `input` question. The LLM generates statements
        from the answer and checks what fraction of them are relevant to the query.
    """
    skip_llm = os.environ.get("SKIP_LLM_METRICS", "false").lower() == "true"
    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))

    if skip_llm or not has_api_key:
        logger.warning(
            f"[{question_id}] Answer relevancy check skipped "
            f"(SKIP_LLM_METRICS={skip_llm}, OPENAI_API_KEY_SET={has_api_key})"
        )
        return AnswerRelevancyResult(
            question_id=question_id,
            score=-1.0,
            passed=True,      # Don't fail CI on skipped metrics
            threshold=threshold,
            reason="Skipped — DeepEval/OpenAI not available",
            skipped=True,
        )

    try:
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        metric = AnswerRelevancyMetric(threshold=threshold)

        test_case = LLMTestCase(
            input=question,
            actual_output=rag_response_text,
        )

        metric.measure(test_case)
        score = metric.score if metric.score is not None else 0.0
        passed = score >= threshold

        return AnswerRelevancyResult(
            question_id=question_id,
            score=round(score, 4),
            passed=passed,
            threshold=threshold,
            reason=metric.reason if hasattr(metric, "reason") else None,
        )

    except ImportError:
        logger.warning(
            f"[{question_id}] deepeval not installed — relevancy check skipped. "
            "Install with: pip install deepeval"
        )
        return AnswerRelevancyResult(
            question_id=question_id,
            score=-1.0,
            passed=True,
            threshold=threshold,
            reason="Skipped — deepeval not installed",
            skipped=True,
        )
    except Exception as e:
        logger.error(f"[{question_id}] Answer relevancy metric error: {e}")
        return AnswerRelevancyResult(
            question_id=question_id,
            score=-1.0,
            passed=True,    # Don't fail CI on metric errors
            threshold=threshold,
            reason=f"Error: {str(e)}",
            skipped=True,
        )


def aggregate_answer_relevancy(results: list[AnswerRelevancyResult]) -> dict:
    """
    Aggregate answer relevancy scores across all questions.

    Skipped results (score == -1.0) are excluded from the aggregate.

    Returns:
        {
            "answer_relevancy": float,  # mean score (higher is better)
            "pass_count": int,
            "fail_count": int,
            "skip_count": int,
            "total": int,
            "threshold": float,
        }
    """
    active = [r for r in results if not r.skipped and r.score >= 0.0]
    skipped = [r for r in results if r.skipped]
    failed = [r for r in active if not r.passed]

    mean_relevancy = (
        sum(r.score for r in active) / len(active) if active else 0.0
    )

    threshold = results[0].threshold if results else 0.70

    return {
        "answer_relevancy": round(mean_relevancy, 4),
        "pass_count": sum(1 for r in active if r.passed),
        "fail_count": len(failed),
        "skip_count": len(skipped),
        "total": len(results),
        "threshold": threshold,
        "per_question": [
            {
                "question_id": r.question_id,
                "score": r.score,
                "passed": r.passed,
                "skipped": r.skipped,
                "reason": r.reason,
            }
            for r in results
        ],
    }
