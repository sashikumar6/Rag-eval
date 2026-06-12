"""
citation_accuracy.py — Custom citation accuracy metric for legal RAG evaluation.

Design decision: We wrote this from scratch rather than using DeepEval's built-in
string match because legal citations require:
  1. Parsing structured citation syntax (e.g. "18 U.S.C. § 1344")
  2. Partial matching — right title, wrong section → 0.5 score, not 0.0
  3. Handling citation variants (§ vs. Section, U.S.C. vs USC, etc.)

See DECISIONS.md § "Why custom citation_accuracy?" for full rationale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParsedCitation:
    """A structured legal citation parsed from text."""
    raw: str
    title: Optional[int] = None          # e.g. 18
    section: Optional[str] = None        # e.g. "1344", "2000e-2"
    subsection: Optional[str] = None     # e.g. "(a)(1)"
    source_type: str = "usc"             # "usc" | "cfr" | "case_law"


@dataclass
class CitationMatchResult:
    """Result of comparing a single actual citation against expected."""
    expected: str
    actual_citations: list[str]
    matched: bool = False
    partial_match: bool = False          # Right title, wrong/missing section
    score: float = 0.0
    reason: str = ""


@dataclass
class CitationAccuracyResult:
    """Aggregate result for a single golden set entry."""
    question_id: str
    expected_citation: Optional[str]
    extracted_citations: list[str]
    match_result: Optional[CitationMatchResult]
    score: float                         # 0.0 – 1.0
    breakdown: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

# Matches: 18 U.S.C. § 1344, 42 U.S.C. § 2000e-2, 26 USC 401, 8 U.S.C. 1101
_USC_PATTERN = re.compile(
    r"""
    (\d{1,2})          # title number (1-2 digits)
    \s*
    U\.?S\.?C\.?       # USC / U.S.C. / U.S.C
    \s*
    §?\s*              # optional section symbol
    ([\d\w\-\.]+       # section number: digits, letters, hyphens, dots
    (?:\([a-zA-Z0-9\-]+\))* # optional subsection(s): (a), (1), (k), (a)(1)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Matches: 26 C.F.R. § 1.401(a)-1, 29 CFR 541.100
_CFR_PATTERN = re.compile(
    r"""
    (\d{1,2})          # title number
    \s*
    C\.?F\.?R\.?       # CFR / C.F.R.
    \s*
    §?\s*
    ([\d\w\-\.\(\)]+)  # part.section format
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Matches inline [Citation: 18 U.S.C. § 1344] format used by the RAG's prompts
_INLINE_CITATION_PATTERN = re.compile(
    r"\[Citation:\s*(.+?)\]",
    re.IGNORECASE,
)


def extract_citations_from_text(text: str) -> list[ParsedCitation]:
    """
    Extract all legal citations from a block of text.

    Handles:
    - Standard USC format: "18 U.S.C. § 1344"
    - CFR format: "26 C.F.R. § 1.401(a)-1"
    - RAG inline format: "[Citation: 18 U.S.C. § 1344]"
    - Variant spellings (USC, U.S.C., U.S.C.)

    Returns a deduplicated list of ParsedCitation objects.
    """
    citations: list[ParsedCitation] = []
    seen: set[str] = set()

    # Extract from [Citation: ...] tags first
    for inline_match in _INLINE_CITATION_PATTERN.finditer(text):
        inner = inline_match.group(1).strip()
        # Re-parse the inner text for structure
        usc = _USC_PATTERN.search(inner)
        if usc:
            raw = usc.group(0)
            if raw not in seen:
                seen.add(raw)
                citations.append(ParsedCitation(
                    raw=raw,
                    title=int(usc.group(1)),
                    section=usc.group(2),
                    source_type="usc",
                ))

    # Extract USC citations from full text
    for m in _USC_PATTERN.finditer(text):
        raw = m.group(0).strip()
        # Normalize for deduplication
        key = f"usc:{m.group(1)}:{m.group(2)}"
        if key not in seen:
            seen.add(key)
            citations.append(ParsedCitation(
                raw=raw,
                title=int(m.group(1)),
                section=m.group(2),
                source_type="usc",
            ))

    # Extract CFR citations
    for m in _CFR_PATTERN.finditer(text):
        raw = m.group(0).strip()
        key = f"cfr:{m.group(1)}:{m.group(2)}"
        if key not in seen:
            seen.add(key)
            citations.append(ParsedCitation(
                raw=raw,
                title=int(m.group(1)),
                section=m.group(2),
                source_type="cfr",
            ))

    return citations


# ---------------------------------------------------------------------------
# Citation parsing (for expected citations in golden set)
# ---------------------------------------------------------------------------

def parse_expected_citation(citation_str: str) -> Optional[ParsedCitation]:
    """Parse a golden set expected_citation string into a ParsedCitation."""
    if not citation_str:
        return None

    usc_match = _USC_PATTERN.search(citation_str)
    if usc_match:
        return ParsedCitation(
            raw=citation_str,
            title=int(usc_match.group(1)),
            section=usc_match.group(2),
            source_type="usc",
        )

    cfr_match = _CFR_PATTERN.search(citation_str)
    if cfr_match:
        return ParsedCitation(
            raw=citation_str,
            title=int(cfr_match.group(1)),
            section=cfr_match.group(2),
            source_type="cfr",
        )

    return None


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def _normalize_section(section: str) -> str:
    """Normalize a section number for comparison."""
    # Remove leading zeros, lowercase
    section = section.strip().lower()
    # Strip trailing subsection refs for base comparison
    # "1344(a)" → "1344"
    base = re.split(r"[\(\[]", section)[0]
    return base


def match_citation(
    expected: ParsedCitation,
    actuals: list[ParsedCitation],
) -> CitationMatchResult:
    """
    Compare an expected citation against a list of extracted citations.

    Scoring:
    - Full match (same title + same section base): 1.0
    - Partial match (same title, wrong/missing section): 0.5
    - No match (wrong title or no citations at all): 0.0

    Why partial scoring?
    Legal research sometimes retrieves the right statute title but a related
    section. A score of 0.5 acknowledges partial correctness rather than
    treating it as a total failure — important for aggregate CI thresholds.
    """
    raw_actuals = [c.raw for c in actuals]

    if not actuals:
        return CitationMatchResult(
            expected=expected.raw,
            actual_citations=raw_actuals,
            matched=False,
            partial_match=False,
            score=0.0,
            reason="No citations found in response",
        )

    expected_title = expected.title
    expected_section_base = _normalize_section(expected.section or "")

    title_matches = [c for c in actuals if c.title == expected_title]

    if not title_matches:
        return CitationMatchResult(
            expected=expected.raw,
            actual_citations=raw_actuals,
            matched=False,
            partial_match=False,
            score=0.0,
            reason=f"Expected title {expected_title} not found in response citations",
        )

    # Title matched — check section
    for actual in title_matches:
        actual_section_base = _normalize_section(actual.section or "")
        if actual_section_base == expected_section_base:
            return CitationMatchResult(
                expected=expected.raw,
                actual_citations=raw_actuals,
                matched=True,
                partial_match=False,
                score=1.0,
                reason=f"Full match: {actual.raw}",
            )

    # Title matched but section didn't — partial credit
    return CitationMatchResult(
        expected=expected.raw,
        actual_citations=raw_actuals,
        matched=False,
        partial_match=True,
        score=0.5,
        reason=(
            f"Partial match: title {expected_title} found but section "
            f"'{expected_section_base}' not matched. Found sections: "
            f"{[_normalize_section(c.section or '') for c in title_matches]}"
        ),
    )


# ---------------------------------------------------------------------------
# Main metric function
# ---------------------------------------------------------------------------

def compute_citation_accuracy(
    question_id: str,
    expected_citation_str: Optional[str],
    rag_response_text: str,
) -> CitationAccuracyResult:
    """
    Compute citation accuracy for a single question.

    Args:
        question_id: ID from golden set (e.g. "q001")
        expected_citation_str: The expected citation from golden set (may be None)
        rag_response_text: Full text of the RAG's /chat response answer

    Returns:
        CitationAccuracyResult with score in [0.0, 1.0]

    Special cases:
    - expected_citation is None (no-answer edge case): score is 1.0 if the
      RAG response contains no confident legal citation (correct behavior is
      to say "I don't know"), 0.0 if it fabricates a citation.
    """
    extracted = extract_citations_from_text(rag_response_text)
    raw_extracted = [c.raw for c in extracted]

    # Edge case: expected_citation is None (no_answer edge case)
    if expected_citation_str is None:
        # Good behavior: RAG says it can't answer and cites nothing
        # Bad behavior: RAG fabricates a citation
        has_fabricated = len(extracted) > 0
        score = 0.0 if has_fabricated else 1.0
        return CitationAccuracyResult(
            question_id=question_id,
            expected_citation=None,
            extracted_citations=raw_extracted,
            match_result=None,
            score=score,
            breakdown={
                "score": score,
                "extracted": raw_extracted,
                "expected": None,
                "match_type": "none",
            },
        )

    expected_parsed = parse_expected_citation(expected_citation_str)
    if expected_parsed is None:
        # Couldn't parse expected — skip scoring gracefully
        return CitationAccuracyResult(
            question_id=question_id,
            expected_citation=expected_citation_str,
            extracted_citations=raw_extracted,
            match_result=None,
            score=0.0,
            breakdown={
                "score": 0.0,
                "extracted": raw_extracted,
                "expected": expected_citation_str,
                "match_type": "none",
            },
        )

    match_result = match_citation(expected_parsed, extracted)

    return CitationAccuracyResult(
        question_id=question_id,
        expected_citation=expected_citation_str,
        extracted_citations=raw_extracted,
        match_result=match_result,
        score=match_result.score,
        breakdown={
            "score": match_result.score,
            "extracted": raw_extracted,
            "expected": expected_citation_str,
            "match_type": "exact" if match_result.matched else "partial" if match_result.partial_match else "none",
        },
    )


# ---------------------------------------------------------------------------
# Aggregate scorer
# ---------------------------------------------------------------------------

def aggregate_citation_accuracy(results: list[CitationAccuracyResult]) -> dict:
    """
    Aggregate citation accuracy across all questions.

    Returns:
        {
            "score": float,          # mean score across all questions
            "full_matches": int,
            "partial_matches": int,
            "no_matches": int,
            "total": int,
            "per_question": [...]
        }
    """
    if not results:
        return {"score": 0.0, "total": 0}

    total = len(results)
    full_matches = sum(1 for r in results if r.match_result and r.match_result.matched)
    partial_matches = sum(1 for r in results if r.match_result and r.match_result.partial_match)
    no_matches = total - full_matches - partial_matches
    mean_score = sum(r.score for r in results) / total

    return {
        "score": round(mean_score, 4),
        "full_matches": full_matches,
        "partial_matches": partial_matches,
        "no_matches": no_matches,
        "total": total,
        "per_question": [
            {
                "question_id": r.question_id,
                "score": r.score,
                "expected": r.expected_citation,
                "extracted": r.extracted_citations,
                "match_type": r.breakdown.get("match_type", "none"),
            }
            for r in results
        ],
    }
