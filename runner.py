"""
runner.py — Main eval harness orchestrator for legal-rag-eval.

Reads the golden set, calls the legal RAG's /chat endpoint for each question,
runs all three metrics, and writes a structured JSON report.

Usage:
    python runner.py                          # Runs full eval, writes baseline_report.json
    python runner.py --report after_cfr       # Writes after_cfr_report.json
    python runner.py --ids q001 q002          # Runs specific questions only
    python runner.py --dry-run                # Validates dataset + config, no HTTP calls
    python runner.py --skip-llm-metrics       # Skip DeepEval (citation only, no API key needed)

Exit codes:
    0 — All CI thresholds passed
    1 — One or more thresholds breached (CI will fail the PR)
    2 — Dataset or config error (not a threshold failure)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

# Adjust import path so this works when called from repo root
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from metrics.citation_accuracy import compute_citation_accuracy, aggregate_citation_accuracy
from metrics.hallucination import compute_hallucination, aggregate_hallucination
from metrics.answer_relevancy import compute_answer_relevancy, aggregate_answer_relevancy
from metrics.source_attribution import compute_source_attribution, aggregate_source_attribution
from metrics.conflict_detection import compute_conflict_detection, aggregate_conflict_detection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAG client
# ---------------------------------------------------------------------------

def call_chat_endpoint(
    question: str,
    base_url: Optional[str] = None,
    timeout: float = cfg.HTTP_TIMEOUT,
    retries: int = cfg.HTTP_MAX_RETRIES,
) -> Optional[dict]:
    """
    Call the legal RAG's /chat endpoint with exponential backoff retries.

    The endpoint signature is:
        POST /api/v1/chat
        Body: {"query": str, "mode": "auto"}
        Response: {"answer": str, "mode": str, "confidence": str,
                   "citations": [...], "session_id": str, ...}

    Returns the parsed JSON response dict, or None on failure.
    """
    endpoint = base_url or cfg.CHAT_ENDPOINT
    payload = {
        "query": question,
        "mode": "auto",
    }

    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException:
            logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}")
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP {e.response.status_code} on attempt {attempt + 1}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calling /chat: {e}")

        if attempt < retries:
            sleep_secs = 2 ** attempt  # Exponential backoff: 1s, 2s
            logger.info(f"Retrying in {sleep_secs}s...")
            time.sleep(sleep_secs)

    return None


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def evaluate_question(entry: dict, dry_run: bool = False) -> dict:
    """
    Run all three metrics for a single golden set entry.

    Args:
        entry: One entry from golden_set.json
        dry_run: If True, skip the HTTP call and use a mock response

    Returns:
        A structured result dict for inclusion in the report.
    """
    question_id = entry["id"]
    question = entry["question"]
    expected_answer = entry.get("expected_answer", "")
    expected_citation = entry.get("expected_citation")
    retrieval_context = entry.get("retrieval_context")
    source_type = entry.get("source_type", "federal")
    expected_conflict = entry.get("expected_conflict")

    logger.info(f"[{question_id}] Evaluating: {question[:80]}...")

    # --------------- Call the RAG endpoint ---------------
    if dry_run:
        rag_answer = (
            f"[DRY RUN] Mock answer for {question_id}. "
            "In production this would call the /chat endpoint."
        )
        rag_raw = {"answer": rag_answer, "mode": "federal", "confidence": "low"}
    else:
        start = time.perf_counter()
        rag_raw = call_chat_endpoint(question)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        if rag_raw is None:
            logger.error(f"[{question_id}] Failed to get response from /chat")
            return {
                "question_id": question_id,
                "question": question,
                "error": "Failed to get response from /chat endpoint",
                "citation_accuracy": None,
                "hallucination": None,
                "answer_relevancy": None,
                "source_attribution": None,
                "conflict_detection": None,
                "overall_pass": False,
                "latency_ms": None,
                "rag_response": None,
            }

        rag_answer = rag_raw.get("answer", "")
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

    # --------------- Run metrics ---------------
    citation_result = compute_citation_accuracy(
        question_id=question_id,
        expected_citation_str=expected_citation,
        rag_response_text=rag_answer,
    )

    hallucination_result = compute_hallucination(
        question_id=question_id,
        rag_response_text=rag_answer,
        retrieval_context=retrieval_context,
        threshold=cfg.HALLUCINATION_MAX,
    )

    relevancy_result = compute_answer_relevancy(
        question_id=question_id,
        question=question,
        rag_response_text=rag_answer,
        threshold=cfg.ANSWER_RELEVANCY_MIN,
    )

    source_attr_result = None
    if source_type in ("case_law", "cross_source"):
        source_attr_result = compute_source_attribution(
            question_id=question_id,
            rag_response_text=rag_answer,
            source_type=source_type,
        )

    conflict_result = None
    if source_type == "cross_source" and expected_conflict is not None:
        conflict_result = compute_conflict_detection(
            question_id=question_id,
            rag_response_text=rag_answer,
            expected_conflict=expected_conflict,
        )

    # --------------- CI pass/fail per question ---------------
    citation_pass = citation_result.score >= cfg.CITATION_ACCURACY_MIN
    hallucination_pass = (
        hallucination_result.passed or hallucination_result.skipped
    )
    relevancy_pass = (
        relevancy_result.passed or relevancy_result.skipped
    )
    source_attr_pass = (
        source_attr_result is None
        or source_attr_result.score >= cfg.SOURCE_ATTRIBUTION_MIN
    )
    conflict_pass = (
        conflict_result is None
        or conflict_result.score >= cfg.CONFLICT_DETECTION_MIN
    )
    overall_pass = (
        citation_pass
        and hallucination_pass
        and relevancy_pass
        and source_attr_pass
        and conflict_pass
    )

    result = {
        "question_id": question_id,
        "title_number": entry.get("title_number"),
        "difficulty": entry.get("difficulty"),
        "edge_case": entry.get("edge_case"),
        "question": question,
        "expected_citation": expected_citation,
        "rag_answer_preview": rag_answer[:400] if rag_answer else None,
        "metrics": {
            "citation_accuracy": {
                "score": citation_result.score,
                "passed": citation_pass,
                "threshold": cfg.CITATION_ACCURACY_MIN,
                "extracted_citations": citation_result.extracted_citations,
                "breakdown": citation_result.breakdown,
            },
            "hallucination": {
                "score": hallucination_result.score,
                "passed": hallucination_pass,
                "threshold": cfg.HALLUCINATION_MAX,
                "skipped": hallucination_result.skipped,
                "reason": hallucination_result.reason,
            },
            "answer_relevancy": {
                "score": relevancy_result.score,
                "passed": relevancy_pass,
                "threshold": cfg.ANSWER_RELEVANCY_MIN,
                "skipped": relevancy_result.skipped,
                "reason": relevancy_result.reason,
            },
            "source_attribution": {
                "score": source_attr_result.score,
                "passed": source_attr_pass,
                "threshold": cfg.SOURCE_ATTRIBUTION_MIN,
                "attributed": source_attr_result.attributed,
                "total": source_attr_result.total,
                "missing_attributions": source_attr_result.missing_attributions,
            } if source_attr_result is not None else None,
            "conflict_detection": {
                "score": conflict_result.score,
                "passed": conflict_pass,
                "threshold": cfg.CONFLICT_DETECTION_MIN,
                "conflict_flagged": conflict_result.conflict_flagged,
                "expected_conflict": expected_conflict,
                "reasoning": conflict_result.reasoning,
            } if conflict_result is not None else None,
        },
        "overall_pass": overall_pass,
        "latency_ms": latency_ms if not dry_run else None,
        "rag_mode": rag_raw.get("mode") if rag_raw else None,
        "rag_confidence": rag_raw.get("confidence") if rag_raw else None,
    }

    status = "✅ PASS" if overall_pass else "❌ FAIL"
    extra = ""
    if source_attr_result is not None:
        extra += f" src_attr={source_attr_result.score:.2f}"
    if conflict_result is not None:
        extra += f" conflict={conflict_result.score:.2f}"
    logger.info(
        f"[{question_id}] {status} | "
        f"citation={citation_result.score:.2f} "
        f"halluc={hallucination_result.score:.2f} "
        f"relev={relevancy_result.score:.2f}"
        f"{extra}"
    )

    return result


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    results: list[dict],
    report_name: str = "baseline",
    dry_run: bool = False,
) -> dict:
    """
    Build the full structured report from per-question results.

    Report schema:
    {
        "metadata": { timestamp, report_name, config, dry_run },
        "summary": {
            "total_questions": int,
            "overall_pass": bool,
            "thresholds_breached": [...],
            "aggregate": {
                "hallucination_rate": float,
                "citation_accuracy": float,
                "answer_relevancy": float,
            }
        },
        "ci": {
            "passed": bool,
            "breaches": [...],
            "thresholds": { hallucination_max, citation_min, relevancy_min }
        },
        "by_difficulty": { easy: {...}, medium: {...}, hard: {...} },
        "by_title": { 8: {...}, 11: {...}, ... },
        "edge_cases": [...],
        "per_question": [...]
    }
    """
    from metrics.citation_accuracy import aggregate_citation_accuracy, CitationAccuracyResult
    from metrics.hallucination import aggregate_hallucination, HallucinationResult
    from metrics.answer_relevancy import aggregate_answer_relevancy, AnswerRelevancyResult
    from metrics.source_attribution import SourceAttributionResult
    from metrics.conflict_detection import ConflictDetectionResult

    # Re-build typed result objects from result dicts for aggregators
    citation_results = []
    hallucination_results = []
    relevancy_results = []

    for r in results:
        if r.get("error"):
            continue

        m = r.get("metrics", {})

        # Citation
        ca = m.get("citation_accuracy", {})
        citation_results.append(CitationAccuracyResult(
            question_id=r["question_id"],
            expected_citation=r.get("expected_citation"),
            extracted_citations=ca.get("extracted_citations", []),
            match_result=None,
            score=ca.get("score", 0.0),
            breakdown=ca.get("breakdown", {}),
        ))

        # Hallucination
        hall = m.get("hallucination", {})
        hallucination_results.append(HallucinationResult(
            question_id=r["question_id"],
            score=hall.get("score", -1.0),
            passed=hall.get("passed", True),
            threshold=hall.get("threshold", cfg.HALLUCINATION_MAX),
            reason=hall.get("reason"),
            skipped=hall.get("skipped", False),
        ))

        # Relevancy
        rel = m.get("answer_relevancy", {})
        relevancy_results.append(AnswerRelevancyResult(
            question_id=r["question_id"],
            score=rel.get("score", -1.0),
            passed=rel.get("passed", True),
            threshold=rel.get("threshold", cfg.ANSWER_RELEVANCY_MIN),
            reason=rel.get("reason"),
            skipped=rel.get("skipped", False),
        ))

    source_attr_results = []
    conflict_results = []
    for r in results:
        if r.get("error"):
            continue
        m = r.get("metrics", {})

        sa = m.get("source_attribution")
        if sa is not None:
            source_attr_results.append(SourceAttributionResult(
                question_id=r["question_id"],
                score=sa.get("score", 0.0),
                attributed=sa.get("attributed", 0),
                total=sa.get("total", 0),
                missing_attributions=sa.get("missing_attributions", []),
            ))

        cd = m.get("conflict_detection")
        if cd is not None:
            conflict_results.append(ConflictDetectionResult(
                question_id=r["question_id"],
                score=cd.get("score", 0.0),
                conflict_flagged=cd.get("conflict_flagged", False),
                expected=cd.get("expected_conflict", False),
                reasoning=cd.get("reasoning", ""),
            ))

    agg_citation = aggregate_citation_accuracy(citation_results)
    agg_halluc = aggregate_hallucination(hallucination_results)
    agg_relev = aggregate_answer_relevancy(relevancy_results)
    agg_source_attr = aggregate_source_attribution(source_attr_results)
    agg_conflict = aggregate_conflict_detection(conflict_results)

    # CI threshold checks
    breaches = []
    if agg_citation["score"] < cfg.CITATION_ACCURACY_MIN:
        breaches.append(
            f"citation_accuracy {agg_citation['score']:.3f} < threshold {cfg.CITATION_ACCURACY_MIN}"
        )
    if agg_halluc["hallucination_rate"] > cfg.HALLUCINATION_MAX and agg_halluc["skip_count"] < len(results):
        breaches.append(
            f"hallucination_rate {agg_halluc['hallucination_rate']:.3f} > threshold {cfg.HALLUCINATION_MAX}"
        )
    if agg_relev["answer_relevancy"] < cfg.ANSWER_RELEVANCY_MIN and agg_relev["skip_count"] < len(results):
        breaches.append(
            f"answer_relevancy {agg_relev['answer_relevancy']:.3f} < threshold {cfg.ANSWER_RELEVANCY_MIN}"
        )
    if agg_source_attr["total_evaluated"] > 0 and agg_source_attr["score"] < cfg.SOURCE_ATTRIBUTION_MIN:
        breaches.append(
            f"source_attribution {agg_source_attr['score']:.3f} < threshold {cfg.SOURCE_ATTRIBUTION_MIN}"
        )
    if agg_conflict["total_evaluated"] > 0 and agg_conflict["score"] < cfg.CONFLICT_DETECTION_MIN:
        breaches.append(
            f"conflict_detection {agg_conflict['score']:.3f} < threshold {cfg.CONFLICT_DETECTION_MIN}"
        )

    ci_passed = len(breaches) == 0

    # Break down by difficulty
    by_difficulty: dict = {}
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r.get("difficulty") == diff and not r.get("error")]
        if subset:
            by_difficulty[diff] = {
                "count": len(subset),
                "pass_count": sum(1 for r in subset if r.get("overall_pass")),
                "mean_citation": round(
                    sum(r["metrics"]["citation_accuracy"]["score"] for r in subset) / len(subset), 4
                ),
            }

    # Break down by title
    by_title: dict = {}
    for r in results:
        if r.get("error"):
            continue
        t = str(r.get("title_number", "unknown"))
        if t not in by_title:
            by_title[t] = {"count": 0, "pass_count": 0, "citation_scores": []}
        by_title[t]["count"] += 1
        if r.get("overall_pass"):
            by_title[t]["pass_count"] += 1
        by_title[t]["citation_scores"].append(
            r["metrics"]["citation_accuracy"]["score"]
        )
    for t, data in by_title.items():
        scores = data.pop("citation_scores")
        data["mean_citation"] = round(sum(scores) / len(scores), 4) if scores else 0.0

    # Edge case results
    edge_cases = [r for r in results if r.get("edge_case") and r.get("edge_case") is not False]

    report = {
        "metadata": {
            "report_name": report_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "config": {
                "rag_base_url": cfg.RAG_BASE_URL,
                "chat_endpoint": cfg.CHAT_ENDPOINT,
                "hallucination_max": cfg.HALLUCINATION_MAX,
                "citation_min": cfg.CITATION_ACCURACY_MIN,
                "relevancy_min": cfg.ANSWER_RELEVANCY_MIN,
                "source_attribution_min": cfg.SOURCE_ATTRIBUTION_MIN,
                "conflict_detection_min": cfg.CONFLICT_DETECTION_MIN,
            },
        },
        "summary": {
            "total_questions": len(results),
            "errored": sum(1 for r in results if r.get("error")),
            "overall_pass": ci_passed,
            "aggregate": {
                "hallucination_rate": agg_halluc["hallucination_rate"],
                "citation_accuracy": agg_citation["score"],
                "answer_relevancy": agg_relev["answer_relevancy"],
                "source_attribution_avg": agg_source_attr["score"],
                "conflict_detection_avg": agg_conflict["score"],
            },
        },
        "ci": {
            "passed": ci_passed,
            "breaches": breaches,
            "thresholds": {
                "hallucination_max": cfg.HALLUCINATION_MAX,
                "citation_min": cfg.CITATION_ACCURACY_MIN,
                "relevancy_min": cfg.ANSWER_RELEVANCY_MIN,
                "source_attribution_min": cfg.SOURCE_ATTRIBUTION_MIN,
                "conflict_detection_min": cfg.CONFLICT_DETECTION_MIN,
            },
        },
        "by_difficulty": by_difficulty,
        "by_title": by_title,
        "edge_cases": [
            {
                "question_id": r["question_id"],
                "edge_case_type": r.get("edge_case"),
                "overall_pass": r.get("overall_pass"),
                "citation_score": r["metrics"]["citation_accuracy"]["score"],
            }
            for r in edge_cases
        ],
        "detailed": {
            "citation_accuracy": agg_citation,
            "hallucination": agg_halluc,
            "answer_relevancy": agg_relev,
            "source_attribution": agg_source_attr,
            "conflict_detection": agg_conflict,
        },
        "per_question": results,
    }

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Legal RAG Eval Runner — evaluates /chat endpoint against golden set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--report",
        default="baseline",
        help="Report name (used as filename prefix). Default: 'baseline' → baseline_report.json",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Run only specific question IDs (e.g. --ids q001 q002)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset without calling the RAG endpoint",
    )
    parser.add_argument(
        "--skip-llm-metrics",
        action="store_true",
        help="Skip DeepEval metrics (no OPENAI_API_KEY needed)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"Override RAG base URL. Default: {cfg.RAG_BASE_URL}",
    )
    parser.add_argument(
        "--dataset",
        default=str(cfg.DATASET_PATH),
        help=f"Path to golden_set.json. Default: {cfg.DATASET_PATH}",
    )

    args = parser.parse_args()

    # Apply CLI overrides
    if args.skip_llm_metrics:
        import os
        os.environ["SKIP_LLM_METRICS"] = "true"

    if args.base_url:
        cfg.RAG_BASE_URL = args.base_url
        cfg.CHAT_ENDPOINT = f"{args.base_url}/api/v1/chat"

    # --------------- Load dataset ---------------
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return 2

    with dataset_path.open() as f:
        golden_set = json.load(f)

    logger.info(f"Loaded {len(golden_set)} questions from {dataset_path}")

    # Filter by IDs if specified
    if args.ids:
        golden_set = [e for e in golden_set if e["id"] in args.ids]
        logger.info(f"Filtered to {len(golden_set)} questions: {args.ids}")

    if not golden_set:
        logger.error("No questions to evaluate after filtering")
        return 2

    # --------------- Health check ---------------
    if not args.dry_run:
        logger.info(f"Checking RAG health at {cfg.RAG_BASE_URL}/health ...")
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{cfg.RAG_BASE_URL}/health")
                if resp.status_code == 200:
                    logger.info("✅ RAG is healthy")
                else:
                    logger.warning(f"⚠️  RAG health check returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"⚠️  Could not reach RAG health endpoint: {e}")
            logger.warning("Proceeding anyway — individual question calls may fail")

    # --------------- Run evaluation ---------------
    logger.info(f"{'[DRY RUN] ' if args.dry_run else ''}Evaluating {len(golden_set)} questions...")
    results = []

    for i, entry in enumerate(golden_set, 1):
        logger.info(f"--- Question {i}/{len(golden_set)} ---")
        result = evaluate_question(entry, dry_run=args.dry_run)
        results.append(result)

    # --------------- Build report ---------------
    report = build_report(results, report_name=args.report, dry_run=args.dry_run)

    # --------------- Write report ---------------
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = cfg.REPORT_DIR / f"{args.report}_report.json"

    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"📄 Report written to: {report_path}")

    # --------------- Print summary ---------------
    summary = report["summary"]
    ci = report["ci"]

    print("\n" + "=" * 60)
    print(f"  EVAL REPORT: {args.report}")
    print("=" * 60)
    print(f"  Questions evaluated : {summary['total_questions']}")
    print(f"  Errors              : {summary['errored']}")
    print()
    print(f"  Citation Accuracy   : {summary['aggregate']['citation_accuracy']:.3f}  (min: {cfg.CITATION_ACCURACY_MIN})")
    print(f"  Hallucination Rate  : {summary['aggregate']['hallucination_rate']:.3f}  (max: {cfg.HALLUCINATION_MAX})")
    print(f"  Answer Relevancy    : {summary['aggregate']['answer_relevancy']:.3f}  (min: {cfg.ANSWER_RELEVANCY_MIN})")
    sa_avg = summary['aggregate'].get('source_attribution_avg', 0.0)
    cd_avg = summary['aggregate'].get('conflict_detection_avg', 0.0)
    if report.get("detailed", {}).get("source_attribution", {}).get("total_evaluated", 0) > 0:
        print(f"  Source Attribution  : {sa_avg:.3f}  (min: {cfg.SOURCE_ATTRIBUTION_MIN})")
    if report.get("detailed", {}).get("conflict_detection", {}).get("total_evaluated", 0) > 0:
        print(f"  Conflict Detection  : {cd_avg:.3f}  (min: {cfg.CONFLICT_DETECTION_MIN})")
    print()

    if ci["passed"]:
        print("  CI RESULT: ✅ ALL THRESHOLDS PASSED")
    else:
        print("  CI RESULT: ❌ THRESHOLD BREACHES:")
        for breach in ci["breaches"]:
            print(f"    • {breach}")

    print("=" * 60 + "\n")

    # Exit code 1 if CI thresholds breached (for GitHub Actions)
    return 0 if ci["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
