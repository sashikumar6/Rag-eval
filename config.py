"""
eval harness configuration

All thresholds and endpoint settings live here.
CI reads these — do not hardcode thresholds elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# RAG endpoint
# ---------------------------------------------------------------------------

RAG_BASE_URL: str = os.environ.get("RAG_BASE_URL", "http://localhost:8000")
CHAT_ENDPOINT: str = f"{RAG_BASE_URL}/api/v1/chat"

# ---------------------------------------------------------------------------
# CI thresholds
# ---------------------------------------------------------------------------

# Maximum allowable hallucination rate (0.0 = none, 1.0 = all hallucinated)
HALLUCINATION_MAX: float = float(os.environ.get("HALLUCINATION_MAX", "0.10"))

# Minimum required citation accuracy (partial + full matches)
CITATION_ACCURACY_MIN: float = float(os.environ.get("CITATION_ACCURACY_MIN", "0.80"))

# Minimum required answer relevancy
ANSWER_RELEVANCY_MIN: float = float(os.environ.get("ANSWER_RELEVANCY_MIN", "0.70"))

# Minimum required source attribution (applies to case_law and cross_source questions)
SOURCE_ATTRIBUTION_MIN: float = float(os.environ.get("SOURCE_ATTRIBUTION_MIN", "0.75"))

# Minimum required conflict detection accuracy (applies to cross_source questions)
CONFLICT_DETECTION_MIN: float = float(os.environ.get("CONFLICT_DETECTION_MIN", "0.80"))

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).parent
DATASET_PATH: Path = REPO_ROOT / "eval_dataset" / "golden_set.json"
REPORT_DIR: Path = REPO_ROOT / "reports"

# ---------------------------------------------------------------------------
# HTTP client settings
# ---------------------------------------------------------------------------

# Timeout for each /chat call (seconds)
HTTP_TIMEOUT: float = float(os.environ.get("HTTP_TIMEOUT", "120.0"))

# Number of retries on transient HTTP errors
HTTP_MAX_RETRIES: int = int(os.environ.get("HTTP_MAX_RETRIES", "2"))

# ---------------------------------------------------------------------------
# LLM settings (used by DeepEval wrappers)
# ---------------------------------------------------------------------------

# DeepEval uses OPENAI_API_KEY from environment automatically.
# Set OPENAI_API_KEY in the environment; do not hardcode here.
EVAL_MODEL: str = os.environ.get("EVAL_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Runner settings
# ---------------------------------------------------------------------------

# If True, skip DeepEval metrics (useful for offline / CI without API key)
SKIP_LLM_METRICS: bool = os.environ.get("SKIP_LLM_METRICS", "false").lower() == "true"

# Edge-case question IDs — evaluated separately in report
EDGE_CASE_IDS: list[str] = ["q015", "q016", "q014", "q027", "q028"]

# Titles covered by the golden set
COVERED_TITLES: list[int] = [8, 11, 15, 18, 26, 28, 29, 42]

# CFR-specific question IDs (added in Week 2)
CFR_QUESTION_IDS: list[str] = [
    "q017", "q018", "q019", "q020", "q021",
    "q022", "q023", "q024", "q025", "q026",
]

# Case law question IDs (added in Week 3)
CASE_LAW_QUESTION_IDS: list[str] = [
    "q029", "q030", "q031", "q032", "q033",
    "q034", "q035", "q036", "q037", "q038",
]

# Cross-source question IDs (statute + CFR/case law together)
CROSS_SOURCE_IDS: list[str] = [
    "q027", "q028",
    "q039", "q040", "q041", "q042", "q043",
]
