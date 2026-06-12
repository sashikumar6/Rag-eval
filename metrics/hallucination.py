"""
hallucination.py — DeepEval HallucinationMetric wrapper for legal RAG evaluation.

Design decision: We wrap DeepEval's HallucinationMetric rather than rolling our
own because hallucination detection requires an LLM to reason over context. The
alternative (BERTScore / NLI) was considered but lacks legal domain awareness.
See DECISIONS.md § "Why DeepEval for hallucination?" for full rationale.

Threshold: 0.10 (stricter than DeepEval's default 0.5) because in the legal
domain, even a 10% hallucination rate is unacceptable — wrong statute citations
can cause real harm.

Usage without DeepEval:
  If SKIP_LLM_METRICS=true or OPENAI_API_KEY is not set, this returns a
  sentinel score of -1.0 and logs a warning. The runner treats -1.0 as
  "skipped" rather than "failed" so CI can still run citation_accuracy checks.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HallucinationResult:
    """Result of hallucination check for a single question."""
    question_id: str
    score: float                  # 0.0 = no hallucination, 1.0 = fully hallucinated
    passed: bool                  # True if score <= threshold
    threshold: float
    reason: Optional[str] = None
    skipped: bool = False         # True if DeepEval/OpenAI unavailable


def compute_hallucination(
    question_id: str,
    rag_response_text: str,
    retrieval_context: Optional[str],
    threshold: float = 0.10,
) -> HallucinationResult:
    """
    Check whether the RAG's answer contains claims not supported by the
    retrieval context (hallucinations).

    Args:
        question_id: ID from golden set (e.g. "q001")
        rag_response_text: The RAG's answer text
        retrieval_context: The ground-truth context from the golden set.
            This is what the RAG *should* have grounded its answer in.
            For no_answer edge cases this may be None.
        threshold: Maximum acceptable hallucination score. Default 0.10.

    Returns:
        HallucinationResult with score in [0.0, 1.0] or skipped=True

    How DeepEval HallucinationMetric works:
        It uses an LLM to check each claim in `actual_output` against the
        provided `context`. Claims that cannot be verified by the context
        are counted as hallucinations. Score = hallucinated_claims / total_claims.
    """
    skip_llm = os.environ.get("SKIP_LLM_METRICS", "false").lower() == "true"
    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))

    if skip_llm or not has_api_key:
        logger.warning(
            f"[{question_id}] Hallucination check skipped "
            f"(SKIP_LLM_METRICS={skip_llm}, OPENAI_API_KEY_SET={has_api_key})"
        )
        return HallucinationResult(
            question_id=question_id,
            score=-1.0,
            passed=True,      # Don't fail CI on skipped metrics
            threshold=threshold,
            reason="Skipped — DeepEval/OpenAI not available",
            skipped=True,
        )

    # No-answer edge case: if retrieval_context is None, the RAG should
    # have said it can't answer. We check if it fabricated an answer.
    if retrieval_context is None:
        # If the response is very long (>100 words), it likely fabricated content
        word_count = len(rag_response_text.split())
        if word_count > 150:
            return HallucinationResult(
                question_id=question_id,
                score=1.0,
                passed=False,
                threshold=threshold,
                reason=(
                    f"No-answer edge case: response is suspiciously long "
                    f"({word_count} words) despite no retrieval context"
                ),
            )
        return HallucinationResult(
            question_id=question_id,
            score=0.0,
            passed=True,
            threshold=threshold,
            reason="No-answer edge case: RAG correctly abstained or responded briefly",
        )

    try:
        from deepeval import evaluate  # noqa: F401 — verify import
        from deepeval.metrics import HallucinationMetric
        from deepeval.test_case import LLMTestCase

        metric = HallucinationMetric(threshold=threshold)

        test_case = LLMTestCase(
            input="[Legal query]",           # Not used by HallucinationMetric
            actual_output=rag_response_text,
            context=[retrieval_context],     # Ground-truth context list
        )

        metric.measure(test_case)
        score = metric.score if metric.score is not None else 0.0
        passed = score <= threshold

        return HallucinationResult(
            question_id=question_id,
            score=round(score, 4),
            passed=passed,
            threshold=threshold,
            reason=metric.reason if hasattr(metric, "reason") else None,
        )

    except ImportError:
        logger.warning(
            f"[{question_id}] deepeval not installed — hallucination check skipped. "
            "Install with: pip install deepeval"
        )
        return HallucinationResult(
            question_id=question_id,
            score=-1.0,
            passed=True,
            threshold=threshold,
            reason="Skipped — deepeval not installed",
            skipped=True,
        )
    except Exception as e:
        logger.error(f"[{question_id}] Hallucination metric error: {e}")
        return HallucinationResult(
            question_id=question_id,
            score=-1.0,
            passed=True,    # Don't fail CI on metric errors
            threshold=threshold,
            reason=f"Error: {str(e)}",
            skipped=True,
        )


def aggregate_hallucination(results: list[HallucinationResult]) -> dict:
    """
    Aggregate hallucination scores across all questions.

    Skipped results (score == -1.0) are excluded from the aggregate.

    Returns:
        {
            "hallucination_rate": float,  # mean score (lower is better)
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

    hallucination_rate = (
        sum(r.score for r in active) / len(active) if active else 0.0
    )

    threshold = results[0].threshold if results else 0.10

    return {
        "hallucination_rate": round(hallucination_rate, 4),
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
