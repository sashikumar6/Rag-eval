"""
conflict_detection.py — Detects whether the RAG correctly flags (or cleanly avoids)
conflicts between a statute's plain text and a court's narrowing/extending construction.

Applies only to cross_source questions that have an `expected_conflict` field.

Scoring:
- expected_conflict=True:  1.0 if a strong conflict signal is present, 0.0 if silent.
- expected_conflict=False: 1.0 if no conflict signal, 0.5 if minor hedging only.

See DECISIONS.md ADR-018 and ADR-019.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConflictDetectionResult:
    """Conflict detection result for a single question."""
    question_id: str
    score: float
    conflict_flagged: bool
    expected: bool
    reasoning: str


# ---------------------------------------------------------------------------
# Conflict signal patterns
# ---------------------------------------------------------------------------

# Strong conflict signals — the response explicitly acknowledges divergence
_STRONG_CONFLICT_SIGNALS = [
    r"conflict",
    r"tension",
    r"diverge[sd]?",
    r"divergence",
    r"textual divergence",
    r"diverges from",
    r"inconsistent",
    r"at odds with",
    r"narrows? the statute",
    r"narrow(?:ed|ing) construction",
    r"limit(?:ed|s|ing) the statute",
    r"restrict(?:ed|s|ing) the statute",
    r"depart(?:s|ed|ure) from the (plain |statutory )?text",
    r"beyond the (plain |statutory )?text",
    r"despite the statutory (text|language)",
    r"despite the (plain|statutory) (text|language)",
    r"the statute.*?but.*?court",
    r"court.*?restrict",
    r"court.*?narrow",
    r"court.*?limit",
    r"plain (text|language).*?(does not|doesn't|cannot)",
    r"statutory text.*?(broader|narrower)",
    r"genuine tension",
    r"direct textual divergence",
    r"court.*?gloss",
    r"limiting.*?construction",
    r"constru(?:ed|ing).*?narrowly",
    r"vagueness",
]

# Minor hedging signals — "however", "but", "although" without specific conflict vocab
_MINOR_HEDGING_SIGNALS = [
    r"\bhowever\b",
    r"\bnevertheless\b",
    r"\byet\b",
    r"\bnotwithstanding\b",
    r"\balthou?gh\b",
    r"\bwhile\b.*?court",
    r"\bwhile\b.*?statute",
    r"\bdespite\b",
    r"\beven though\b",
    r"\bbut\b.*?the court",
]

_STRONG_RE = re.compile(
    "|".join(f"(?:{p})" for p in _STRONG_CONFLICT_SIGNALS),
    re.IGNORECASE | re.DOTALL,
)

_HEDGE_RE = re.compile(
    "|".join(f"(?:{p})" for p in _MINOR_HEDGING_SIGNALS),
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Main metric
# ---------------------------------------------------------------------------

def compute_conflict_detection(
    question_id: str,
    rag_response_text: str,
    expected_conflict: bool,
) -> ConflictDetectionResult:
    """
    Compute conflict detection score for a single cross-source question.

    Args:
        question_id: ID from golden set
        rag_response_text: Full answer text from the RAG
        expected_conflict: True if the statute and case law genuinely conflict,
                           False if they are consistent

    Returns:
        ConflictDetectionResult with score in {0.0, 0.5, 1.0}
    """
    if not rag_response_text or not rag_response_text.strip():
        return ConflictDetectionResult(
            question_id=question_id,
            score=0.0 if expected_conflict else 1.0,
            conflict_flagged=False,
            expected=expected_conflict,
            reasoning="empty response — no conflict signal possible",
        )

    has_strong_signal = bool(_STRONG_RE.search(rag_response_text))
    has_hedging = bool(_HEDGE_RE.search(rag_response_text))

    conflict_flagged = has_strong_signal

    if expected_conflict:
        if has_strong_signal:
            score = 1.0
            reasoning = "Strong conflict signal detected — correctly flags divergence."
        else:
            score = 0.0
            reasoning = (
                "No conflict signal found. Expected the response to flag divergence "
                "between the statute's plain text and the court's narrowing construction."
            )
    else:
        # expected_conflict=False: statute and case are consistent
        if has_strong_signal:
            score = 0.0
            reasoning = (
                "Strong conflict signal present, but statute and case are consistent. "
                "The response incorrectly implies divergence."
            )
        elif has_hedging:
            score = 0.5
            reasoning = (
                "Minor hedging language detected (e.g. 'however', 'but'). "
                "The statute and case are consistent; hedging is unnecessary but not wrong."
            )
        else:
            score = 1.0
            reasoning = "Clean response — correctly presents statute and case as consistent."

    return ConflictDetectionResult(
        question_id=question_id,
        score=score,
        conflict_flagged=conflict_flagged,
        expected=expected_conflict,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate_conflict_detection(results: list[ConflictDetectionResult]) -> dict:
    """
    Aggregate conflict detection scores across questions that ran this metric.

    Returns:
        {
            "score": float,
            "total_evaluated": int,
            "correct_conflict_flags": int,
            "correct_clean": int,
            "per_question": [...]
        }
    """
    if not results:
        return {"score": 0.0, "total_evaluated": 0, "per_question": []}

    scores = [r.score for r in results]
    mean_score = sum(scores) / len(scores)

    correct_flags = sum(
        1 for r in results if r.expected and r.conflict_flagged
    )
    correct_clean = sum(
        1 for r in results if not r.expected and not r.conflict_flagged
    )

    return {
        "score": round(mean_score, 4),
        "total_evaluated": len(results),
        "correct_conflict_flags": correct_flags,
        "correct_clean": correct_clean,
        "per_question": [
            {
                "question_id": r.question_id,
                "score": r.score,
                "conflict_flagged": r.conflict_flagged,
                "expected_conflict": r.expected,
                "reasoning": r.reasoning,
            }
            for r in results
        ],
    }
