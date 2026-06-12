# legal-rag-eval

## What this is

A standalone black-box evaluation harness for a Federal Law RAG system. Given a golden dataset of hand-curated legal Q&A pairs, it calls the RAG's `/chat` endpoint over HTTP, scores each response on five metrics (citation accuracy, hallucination rate, answer relevancy, source attribution, and conflict detection), and produces a structured JSON report with CI pass/fail thresholds. The harness has zero coupling to the RAG codebase — it needs only an HTTP URL.

The eval lives in a separate repository by design. Decoupling the eval from the RAG means it can regression-test any backend that speaks the same API contract (`POST /api/v1/chat` → `{"answer": str, "citations": [...], "mode": str}`). This lets the RAG be refactored, migrated, or replaced without touching the eval infrastructure. The RAG being tested lives at [github.com/yourusername/Legalchatbot](https://github.com/yourusername/Legalchatbot).

## Quick Start (Local)

**Prerequisites:** Python 3.11+, an OpenAI API key, and the legal-rag running locally.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env: set OPENAI_API_KEY and optionally RAG_BASE_URL

# 3. Run the full eval (all 43 questions, all 5 metrics)
python runner.py

# Skip DeepEval metrics (no API key needed — citation-only mode)
python runner.py --skip-llm-metrics

# Dry run: validates config and dataset without HTTP calls
python runner.py --dry-run

# Run specific questions
python runner.py --ids q029 q030 q039

# Custom report name (writes reports/after_cfr_report.json)
python runner.py --report after_cfr
```

## Quick Start (Against Deployed RAG)

```bash
RAG_BASE_URL=https://your-deployed-url.com python runner.py
```

Or with a named report:

```bash
python runner.py --base-url https://your-deployed-url.com --report production_baseline
```

## Thresholds

| Metric | Threshold | Why this number |
|--------|-----------|-----------------|
| Hallucination Rate | < 10% | Legal accuracy is non-negotiable — one fabricated statute can cause real harm |
| Citation Accuracy | > 80% | Statute numbers must be verifiable; wrong section = wrong law |
| Answer Relevancy | > 70% | Answers must address the actual question, not adjacent topics |
| Source Attribution | > 75% | Users need to know whether a fact came from statute, regulation, or case law |
| Conflict Detection | > 80% | Conflicts between statute and case law must never be silently ignored |

All thresholds are configurable via environment variables (`HALLUCINATION_MAX`, `CITATION_ACCURACY_MIN`, etc.) without touching code. See `config.py`.

## Golden Dataset

| Source | Questions | Edge Cases |
|--------|-----------|------------|
| U.S. Code Title 8 | 2 | — |
| U.S. Code Title 11 | 2 | — |
| U.S. Code Title 15 | 2 | — |
| U.S. Code Title 18 | 2 | 1 multi_statute |
| U.S. Code Title 26 | 2 | — |
| U.S. Code Title 28 | 1 | 1 no_answer |
| U.S. Code Title 29 | 2 | — |
| U.S. Code Title 42 | 2 | 1 ambiguous |
| CFR Title 26 + 29 | 10 | — |
| Cross-source | 7 | 2 expected_conflict |
| Case Law | 10 | — |
| **Total** | **43** | **5** |

Every entry includes: `question`, `expected_answer`, `expected_citation`, `retrieval_context`, `difficulty` (easy/medium/hard), and `source_type`. Cross-source entries add `expected_sources` and `expected_conflict`. Case law entries add `expected_court`.

The dataset is hand-curated using real verifiable citations — no LLM-generated ground truth. See [ADR-014](DECISIONS.md) for why.

## Metrics

### `citation_accuracy` — Custom

**Type:** Custom regex-based scorer, no LLM required.

**What it measures:** Whether the RAG's response cites the correct statute or case. Extracts `18 U.S.C. § 1344`-style and `Party v. Party, NNN U.S. NNN`-style citations from the response text and compares against the golden set's `expected_citation`.

**Scoring:** Full match (right title + right section) = 1.0. Partial match (right title, wrong section) = 0.5. No match = 0.0. For `no_answer` edge cases: 1.0 if the RAG correctly abstains, 0.0 if it fabricates a citation.

**Why it matters for legal RAG:** Wrong section numbers are not a formatting problem — `18 U.S.C. § 1343` (wire fraud) and `18 U.S.C. § 1344` (bank fraud) carry different sentences and elements. Partial credit acknowledges that citing the right title reflects genuine retrieval signal even when the section is off.

---

### `hallucination` — DeepEval wrapper

**Type:** DeepEval `HallucinationMetric` wrapper. Requires `OPENAI_API_KEY`; set `SKIP_LLM_METRICS=true` to skip without failing CI.

**What it measures:** The fraction of claims in the RAG's answer that cannot be grounded in the `retrieval_context` from the golden set. Score = hallucinated claims / total claims; lower is better.

**Why it matters for legal RAG:** Citation accuracy checks the citation string, not the surrounding explanation. A RAG can cite `18 U.S.C. § 1344` correctly while fabricating the penalty, the elements, or the mens rea. Hallucination detection checks the prose.

---

### `answer_relevancy` — DeepEval wrapper

**Type:** DeepEval `AnswerRelevancyMetric` wrapper. Requires `OPENAI_API_KEY`; graceful degradation as above.

**What it measures:** Whether each statement in the answer is relevant to the question asked. Score in [0.0, 1.0]; higher is better.

**Why it matters for legal RAG:** A legally accurate answer to the wrong question is still wrong. This metric catches mode routing failures — if the RAG answers a contract question with bankruptcy law, citation accuracy may still be 0.0 but relevancy makes the failure mode explicit.

---

### `source_attribution` — Custom

**Type:** Custom regex-based sentence-level scorer, no LLM required.

**What it measures:** For case law and cross-source questions, scores whether each substantive sentence in the response contains an attribution signal — a USC citation, a CFR citation, or a case law citation (`Party v. Party`, reporter citation, "the Court held"). Score = attributed sentences / total substantive sentences.

**Why it matters for legal RAG:** Multi-source answers are dangerous without provenance. If a response says "the penalty is 30 years" without attributing it to either the statute or the case, the user cannot verify which source that claim came from, or whether the sources agree.

---

### `conflict_detection` — Custom

**Type:** Custom keyword-signal scorer, no LLM required. Applies only to cross-source questions with an `expected_conflict` field.

**What it measures:** Whether the RAG correctly flags (or cleanly avoids flagging) divergence between a statute's plain text and a court's narrowing or extending construction. Score is 1.0/0.5/0.0: 1.0 if expected and detected, 1.0 if not expected and absent, 0.5 for minor unwarranted hedging, 0.0 if conflict is expected but response is silent.

**Why it matters for legal RAG:** A RAG that silently presents the narrowed Skilling construction of `18 U.S.C. § 1346` as the plain statutory text is making a legal judgment it has no authority to make. Conflicts between statute and case law must be surfaced, not resolved silently.

## Sample Report Output

```json
{
  "timestamp": "2026-01-15T10:30:00Z",
  "rag_url": "http://localhost:8000",
  "total_questions": 43,
  "results_by_source": {
    "federal": {"citation_accuracy": 0.87, "hallucination": 0.03},
    "cfr": {"citation_accuracy": 0.84, "hallucination": 0.05},
    "case_law": {"citation_accuracy": 0.81, "source_attribution": 0.79},
    "cross_source": {"conflict_detection": 0.83}
  },
  "aggregate": {
    "hallucination_rate": 0.04,
    "citation_accuracy": 0.87,
    "answer_relevancy": 0.81,
    "source_attribution": 0.79,
    "conflict_detection": 0.83
  },
  "overall": "PASS",
  "failed_questions": []
}
```

Reports are written to `reports/<name>_report.json` and include per-question breakdowns, difficulty-stratified scores, and CI breach details for every metric.

## Why Legal RAG Specifically

Hallucination in a general-purpose RAG is an inconvenience — a user gets a wrong fact and checks another source. Hallucination in a legal RAG can cause direct, irreversible harm: a defendant advised of the wrong statute of limitations misses their window; a business structured around a misquoted tax regulation faces penalties; a tenant told the wrong eviction procedure loses housing. Legal citations are binary: `18 U.S.C. § 1344` either exists and says what the RAG claims, or it doesn't. That determinism makes citation accuracy a concrete, defensible metric rather than a matter of subjective judgment — and it makes the "why eval matters" story immediately obvious to anyone who has ever needed to cite a source in a legal proceeding.

## CI Integration

The eval runs nightly at 2am UTC via `.github/workflows/eval_ci.yml` and can be triggered manually or by a `repository_dispatch` event from the legal-rag repo. Exit code 0 = all thresholds passed; exit code 1 = at least one threshold breached (blocks CI).

To trigger from the RAG's CI:

```yaml
- name: Trigger eval harness
  run: |
    curl -X POST https://api.github.com/repos/YOUR_ORG/legal-rag-eval/dispatches \
      -H "Authorization: token ${{ secrets.EVAL_REPO_PAT }}" \
      -H "Content-Type: application/json" \
      -d '{"event_type":"legal-rag-push","client_payload":{"sha":"${{ github.sha }}"}}'
```

Required secrets: `OPENAI_API_KEY` (optional; set `SKIP_LLM_METRICS=true` if absent), `EVAL_REPO_PAT` (for cross-repo dispatch).

## Repo Structure

```
legal-rag-eval/
├── eval_dataset/
│   └── golden_set.json          ← 43 hand-curated Q&A pairs
├── metrics/
│   ├── citation_accuracy.py     ← Custom regex scorer (no LLM)
│   ├── hallucination.py         ← DeepEval wrapper
│   ├── answer_relevancy.py      ← DeepEval wrapper
│   ├── source_attribution.py    ← Custom sentence-level attribution scorer
│   └── conflict_detection.py   ← Custom conflict signal detector
├── reports/                     ← Written by runner.py (gitignored)
├── .github/
│   └── workflows/
│       └── eval_ci.yml          ← Nightly + cross-repo CI trigger
├── runner.py                    ← Main orchestrator
├── config.py                    ← All thresholds + endpoint config
├── requirements.txt
├── .env.example                 ← Environment variable template
└── DECISIONS.md                 ← 20 architectural decisions with alternatives
```

## See Also

- [DECISIONS.md](DECISIONS.md) — 20 architecture decisions, each with alternatives considered and trade-offs accepted. Written for interview prep.
- Legal RAG repo: [github.com/yourusername/Legalchatbot](https://github.com/yourusername/Legalchatbot)
