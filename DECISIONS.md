# DECISIONS.md — Architectural Decision Record

This file documents every significant design decision made in this project,
including what alternatives were considered and why each choice was made.
Written for interview prep: every decision has a **"Why not X?"** section
and a **"1-liner for quick recall"** at the end.

---

## Decision 1: Custom `citation_accuracy` Metric Instead of Generic String Match

**Choice**: A regex-based citation extractor with partial matching (right title, wrong section = 0.5 score).

**Alternatives Considered**:
- `ROUGE-L` score between expected_answer and actual_answer
- Exact string match of `expected_citation` in the response text
- Embedding cosine similarity between expected and actual answers
- DeepEval's built-in `GEvalMetric` with a custom rubric

**Why this**:
Legal citations have structure: `18 U.S.C. § 1344`. A pure string match fails on:
- Spacing variants: `18 U.S.C. §1344` vs `18 U.S.C. § 1344`
- USC/U.S.C. variants
- The RAG may correctly cite title 18 but cite § 1343 (wire fraud) instead of § 1344 (bank fraud) — related, not wrong. Partial credit (0.5) reflects this.
- ROUGE would score a verbose wrong answer higher than a concise correct one.

**Why not exact string match**: Fails on citation formatting variants. A response citing `18 USC 1344` should not score 0.0 against `18 U.S.C. § 1344`.

**Why not ROUGE**: ROUGE measures n-gram overlap between entire answer texts. A RAG that copies the question into the answer scores high. It doesn't specifically validate *citations*.

**Why not embedding similarity**: Embeddings conflate "18 U.S.C. § 1344" and "18 U.S.C. § 1343" as very similar (they're in the same region of embedding space) — so a hallucinated adjacent section could score 0.95.

**Why not GEval**: Requires an LLM call per question, adds cost and latency. Citation checking is a deterministic problem — use deterministic code.

**Trade-offs**: Our regex doesn't handle case law citations like `Smith v. Jones, 123 F.3d 456` — added in Week 3's `source_attribution.py`.

**Interview 1-liner**: *"Legal citations have structure — I parse them structurally and apply partial credit because adjacent-section citations are partially correct, not totally wrong."*

---

## Decision 2: DeepEval for Hallucination and Relevancy (Not Rolling Our Own)

**Choice**: Wrap `deepeval.metrics.HallucinationMetric` and `AnswerRelevancyMetric`.

**Alternatives Considered**:
- Natural Language Inference (NLI) models (e.g., `cross-encoder/nli-deberta-v3-base`)
- BERTScore between actual and expected answers
- Custom GPT-4 prompt to evaluate claims
- Trusting citation accuracy alone (skip hallucination entirely)

**Why this**:
Hallucination detection requires reasoning over whether a claim is supported by a context passage. This is fundamentally a reading comprehension task — LLMs do it well. DeepEval gives us:
- A tested, maintained implementation
- `HallucinationMetric` that takes a `context` list (our `retrieval_context`) and checks each claim in `actual_output`
- Structured `reason` output (useful for debugging which claim hallucinated)

**Why not NLI models**: Require GPU for reasonable speed; hallmark of legal domain is long contexts (whole statute sections). DeBERTa has a 512-token limit.

**Why not BERTScore**: Measures semantic similarity between two texts, not whether claims are grounded in a *third* text (the context). High BERTScore doesn't mean no hallucination.

**Why not custom GPT-4 prompt**: DeepEval's prompt is already well-tuned. Writing our own risks prompt drift, inconsistent output formats, and maintenance burden.

**Why not skip hallucination**: Citation accuracy only checks if the *citation string* appears; a RAG could fabricate the surrounding explanation while citing the right statute. We need to check the prose, not just the citation.

**Trade-offs**: Requires `OPENAI_API_KEY` at runtime. Mitigated by `SKIP_LLM_METRICS=true` flag and graceful degradation (skipped ≠ failed in CI).

**Interview 1-liner**: *"Hallucination is a reading comprehension problem — LLMs solve it better than heuristics. DeepEval is battle-tested; I add legal-domain configuration (0.10 threshold, retrieval_context as the ground truth)."*

---

## Decision 3: Strict Mode Isolation Between Data Sources

**Choice**: Each data source (federal corpus, CFR, case law) has its own Qdrant collection and its own retriever class. The agent never crosses retrieval modes silently.

**Alternatives Considered**:
- Single Qdrant collection with a `source_type` metadata filter
- Unified retriever with source weighting
- Merging all sources at ingestion time

**Why this**:
Legal accuracy requires knowing *where* a fact came from:
- A statute (U.S. Code) states what the law is
- A CFR regulation states how an agency implements it
- A case says how a court interpreted it in a specific context
Mixing these silently is dangerous — a case holding may be overturned, a regulation may be superseded by a newer statute. Isolation makes attribution mandatory, not optional.

**Why not single collection with filters**: Qdrant filters on metadata are fast, but Qdrant's dense vector search doesn't isolate semantic space — a query for "tax fraud" could return CFR results ranked above U.S. Code results because the CFR text happens to be more similar. Separate collections give complete isolation.

**Why not unified retriever with source weighting**: Weights are hard to tune and leak through. If a legal document asks "what does 18 U.S.C. § 1344 say?" you want only the statute, not a blend with a circuit court opinion.

**Why not merging at ingestion**: Source attribution becomes impossible after ingestion. If a chunk says "fraud carries 30 years" you cannot tell if that's from the statute or from a court's dicta.

**Trade-offs**: More Qdrant collections = more RAM, more embedding passes. Acceptable trade-off given legal accuracy requirements.

**Interview 1-liner**: *"In legal research, provenance isn't optional — a statute, a regulation, and a case are three different things with different precedential weight. Mixing them silently is a correctness bug, not a recall improvement."*

---

## Decision 4: LangGraph for the Agent Workflow

**Choice**: LangGraph `StateGraph` with typed `GraphState` and named nodes.

**Alternatives Considered**:
- Plain Python: sequential function calls with if/else routing
- Celery task chains for async pipeline
- LangChain `AgentExecutor` (ReAct pattern)
- Prefect / Airflow for workflow orchestration

**Why this**:
The 13-node workflow has conditional branching (retry on bad retrieval, skip clarification if mode is clear, verify before finalizing). LangGraph gives:
- A visual, inspectable graph of the workflow
- Typed state (`GraphState` as `TypedDict`) that flows between nodes
- Conditional edges that make routing logic explicit and testable
- Easy extension: adding a new node (e.g., `retrieve_cfr_context`) is one `workflow.add_node()` call

**Why not plain Python**: Conditional retry logic with multiple retry counts, verification loops, and mode-dependent routing becomes spaghetti. No way to visualize or test the workflow structure.

**Why not Celery**: Celery is for distributed async work (fire and forget). The RAG's workflow is synchronous and stateful within a single request. Celery adds Redis dependency with no benefit.

**Why not LangChain AgentExecutor**: ReAct pattern is for open-ended tool use. Legal RAG has a *fixed* workflow — the routing decisions are deterministic (is there an upload_id? is the query about tax?). A fixed workflow outperforms an LLM deciding which tool to call next.

**Why not Prefect/Airflow**: These are data pipeline schedulers. They add massive infrastructure overhead for what is fundamentally a per-request workflow.

**Trade-offs**: LangGraph is a newer library with breaking changes between versions. Pinned to `langgraph==0.2.60` in requirements.

**Interview 1-liner**: *"LangGraph makes branching workflows explicit and testable. The alternative is nested if/else that nobody can reason about — LangGraph turns your workflow into a diagram you can inspect and extend node by node."*

---

## Decision 5: Qdrant Collection Namespacing (Not Single Collection + Filters)

**Choice**: Separate Qdrant collections per corpus: `federal_corpus`, `uploaded_documents`, `cfr_corpus`, `case_law_corpus`.

*(See Decision 3 for the retriever isolation rationale — this decision focuses on the Qdrant storage layer specifically.)*

**Alternatives Considered**:
- Single `legal_corpus` collection with `source_type` payload filter
- Two collections: `federal` + `user_documents`
- One collection per U.S. Code title (8 collections)

**Why this**:
- Each corpus has different payloads: U.S. Code has `title_number`, `section_number`, `canonical_citation`; documents have `upload_id`, `page_number`; CFR has `part`, `section_identifier`. A single collection's payload schema would be a union of all fields with many null values.
- Embedding spaces may differ: if we later switch to a domain-specific legal embedding model for case law but keep `text-embedding-3-small` for statutes, separate collections allow per-collection model configuration.
- Collection-level access control is simpler than payload-level filtering for multi-tenant uploads.

**Why not one collection per title**: 8 federal title collections × future CFR titles × case law courts = unmanageable proliferation. The title filter is a payload filter within `federal_corpus`, not a collection split.

**Why not single collection**: Payload schema drift (different fields per source type), impossible to change vector size per corpus, collection-level metrics become meaningless.

**Trade-offs**: Each collection requires its own Qdrant index, consuming ~100MB RAM per 1M vectors. For the legal domain (~500k chunks for 8 titles), this is fine.

**Interview 1-liner**: *"Collections give you schema isolation, independent scaling, and the ability to use different embedding models per corpus. A `source_type` filter in a single collection saves RAM but sacrifices everything else."*

---

## Decision 6: USLM XML Parsing Over Plain Text

**Choice**: Parse the official USLM (United States Legislative Markup) XML format from `uscode.house.gov/download/`.

**Alternatives Considered**:
- Download PDF versions and use PyMuPDF
- Scrape `uscode.house.gov` HTML
- Use the GPO FDsys text files (`.txt` format)
- Use a third-party legal data API (CourtListener, Law.gov)

**Why this**:
USLM XML preserves the document hierarchy: title → chapter → subchapter → part → section → subsection. This hierarchy is critical for:
- Generating canonical citations: `18 U.S.C. § 1344` requires knowing title=18, section=1344
- Chunking at section boundaries without losing structural context
- Storing `chapter`, `subchapter` in metadata for context-aware retrieval

**Why not PDF**: PDFs lose hierarchy. PyMuPDF can extract text, but identifying chapter/section boundaries requires fragile heuristics (font size, indentation). Citation extraction from PDFs is error-prone.

**Why not HTML scraping**: HTML structure changes with site redesigns. The USLM XML is a stable, versioned, machine-readable format maintained by the House Office of the Law Revision Counsel.

**Why not plain text**: Same as PDF — no hierarchy, no reliable section boundaries.

**Why not third-party API**: CourtListener provides case law (used in Week 3). For statutes, the official source (USLM) is authoritative and free.

**Trade-offs**: USLM XML files are large (usc26.xml is ~300MB). Parsing requires `lxml` and careful namespace handling.

**Interview 1-liner**: *"The official USLM XML is the only format that preserves the hierarchy needed to generate legally correct citations. PDFs and text files are unstructured — you can't reliably tell where Title 18 Chapter 1 ends and Chapter 2 begins."*

---

## Decision 7: CourtListener API for Case Law (Not Scraping)

**Choice**: Use the CourtListener REST API v4 (`courtlistener.com/api/rest/v4/opinions/`).

**Alternatives Considered**:
- Scrape Google Scholar case law pages
- Scrape `scholar.google.com/scholar_case?case=...`
- Download bulk data from CourtListener (full dataset)
- Use Westlaw/LexisNexis API (paid)

**Why this**:
- CourtListener is a 501(c)(3) nonprofit with explicit API terms of service that allow programmatic access
- The API returns structured JSON: `case_name`, `citation`, `court`, `date_filed`, `precedential_status` — exactly what we need for citation formatting
- Filtering by `precedential_status=Published`, `court__jurisdiction=F` (federal), and date range is built into the API
- Respects rate limits: 5,000 requests/day on free tier, with `Retry-After` headers
- `tenacity` backoff handles transient errors without violating TOS

**Why not Google Scholar scraping**: Google explicitly prohibits scraping in their TOS. Scraped HTML structure changes frequently. No `precedential_status` or structured citation data.

**Why not CourtListener bulk data**: Bulk downloads are gigabytes of JSON. For 20 years of federal opinions across 8 title areas, the filtered API approach downloads only what's relevant (~50k-100k opinions).

**Why not Westlaw/LexisNexis**: They charge $100-300/month for API access. CourtListener is free for non-commercial use.

**Trade-offs**: CourtListener doesn't have all opinions (mainly PACER-accessible federal opinions). State court opinions may be incomplete. Acceptable for our use case (federal law only).

**Interview 1-liner**: *"CourtListener has an explicit API, structured metadata, and ToS that allow our use case. Scraping Google Scholar would give us more coverage but is legally and technically fragile."*

---

## Decision 8: Conflict Detection in Source Merger

**Choice**: When `source_merger.py` detects that a statute says X and a case law holding says something contradictory, the system *annotates* the conflict rather than silently picking the higher-scoring source.

**Alternatives Considered**:
- Rank by relevance score and return the top result regardless of source
- Let the LLM decide which source to trust in the prompt
- Return both results without annotation and let the user figure it out

**Why this**:
In the legal domain, silence is dangerous. If 18 U.S.C. § 1344 says "30 years maximum" but a circuit court opinion says "the statute was interpreted as 20 years in practice," a legal professional needs to know *both* exist and that they appear contradictory. Silently picking the higher-scoring chunk could:
- Return a circuit-specific interpretation as if it's universal law
- Miss that the statute was amended after the case was decided

Annotation (`[CONFLICT DETECTED: statute says X, case law says Y]`) surfaces the ambiguity to the user, who can then consult a licensed attorney.

**Why not let the LLM decide**: The LLM has no knowledge of which circuit's interpretation binds the user, which opinion is more recent, or whether the statute was amended. These are facts, not reasoning problems.

**Why not return both without annotation**: The user would not know why two contradictory facts appear in the answer. The annotation makes the system's reasoning transparent.

**Trade-offs**: Conflict detection requires comparing claims across chunks — currently implemented as keyword/negation matching, which has false positives. Week 3 implementation uses a simple NLI check.

**Interview 1-liner**: *"In law, surfacing ambiguity is a feature, not a bug. A system that silently picks one source when statute and case law conflict is making a legal judgment it has no authority to make."*

---

## Decision 9: `pydantic-settings` for Configuration

**Choice**: `class Settings(BaseSettings)` with env file at `infra/.env`.

**Alternatives Considered**:
- `os.environ.get()` calls scattered throughout the codebase
- `python-dotenv` with manual `os.getenv()` + type casting
- Dynaconf (multi-environment config library)
- Hardcoded values in source code

**Why this**:
`BaseSettings` gives:
- Type validation (e.g., `port: int = 8000` automatically casts `"8000"` from env to `int`)
- Field validators (e.g., `parse_federal_titles` parses JSON arrays from env strings)
- A single source of truth — all settings are fields of one class
- IDE autocomplete for `settings.openai_api_key`
- Pydantic's error messages when invalid values are provided

**Why not scattered `os.environ.get()`**: No type casting, no validation, no documentation of what env vars exist. Settings spread across files become impossible to audit.

**Why not Dynaconf**: Overkill for a single-environment backend. Dynaconf adds a dependency and its own config file format without material benefit over `pydantic-settings`.

**Trade-offs**: Pydantic v2 + pydantic-settings v2 have breaking changes from v1. Pinned to specific versions in requirements.txt.

**Interview 1-liner**: *"pydantic-settings gives me type safety, validation, and documentation of every env var in one class — scattered `os.getenv()` calls are a maintenance trap that breaks silently when you forget to cast types."*

---

## Decision 10: Retriever Injection via GraphState (Not Singletons)

**Choice**: `FederalRetriever` and `DocumentRetriever` instances are created per-request and injected into `GraphState` as fields (`federal_retriever`, `document_retriever`).

**Alternatives Considered**:
- Global singleton retrievers initialized at startup
- Thread-local storage per request
- Factory function called inside each graph node
- Dependency injection via FastAPI `Depends()`

**Why this**:
- **Testability**: In unit tests, you can inject a mock retriever without monkey-patching globals: `state["federal_retriever"] = MockRetriever()`
- **Thread safety**: Each request gets its own retriever instance — no shared mutable state
- **Visibility**: The graph state explicitly documents which retrievers are available. A new developer reading `GraphState` immediately sees all injected dependencies.
- **Future flexibility**: When we add `cfr_retriever` in Week 2, we add one field to `GraphState` and one injection in the service layer — no changes to existing nodes.

**Why not singletons**: A single shared `FederalRetriever` with a `qdrant_client` that's not thread-safe would cause race conditions. Even if Qdrant client is thread-safe, singleton retrievers make testing harder (you can't substitute a mock without patching the global).

**Why not FastAPI `Depends()`**: The LangGraph workflow runs inside a service method, not directly in a FastAPI route handler. FastAPI's dependency injection doesn't reach into LangGraph node functions.

**Trade-offs**: Creating a new retriever instance per request adds marginal overhead (object allocation, not network calls). The Qdrant client is reused via the module-level `_qdrant_client` in routes.py.

**Interview 1-liner**: *"Injection via GraphState makes dependencies explicit and mock-able. Singletons hide dependencies and make tests depend on global state — which breaks when you run tests in parallel."*

---

## Decision 11: Black-Box Eval Harness (Only Calls `/chat` Endpoint)

**Choice**: The eval harness only calls `POST /api/v1/chat` — it never imports or calls RAG code directly.

**Alternatives Considered**:
- Import RAG Python modules and call the agent graph directly (white-box testing)
- Use RAG's `/retrieval` endpoint to also test retrieval quality separately
- Test individual LangGraph nodes in isolation

**Why this**:
- **Portability**: The eval harness works against any deployment (local, staging, production, a competitor's API) without needing the RAG codebase.
- **True integration test**: Calling `/chat` validates the entire stack — FastAPI routing, LangGraph workflow, Qdrant retrieval, OpenAI generation, verification, response schema. White-box tests of individual nodes don't catch integration bugs.
- **Decoupling**: The eval harness can evolve independently. When we refactor the agent graph in Week 2, the harness doesn't need to change — only the `/chat` response quality changes.
- **CI realism**: Production CI should test what users actually experience.

**Why not white-box testing**: White-box tests duplicate the unit tests already in `backend/app/tests/`. The eval harness is a *quality* check, not a *correctness* check.

**Why not the `/retrieval` endpoint**: Retrieval quality is tested implicitly by citation accuracy — if the retriever returns the wrong chunks, the LLM will cite wrong statutes.

**Trade-offs**: We can't inspect why a question failed — did retrieval return bad chunks, or did the LLM hallucinate? The `rag_mode` and `rag_confidence` fields in the report provide some signal.

**Interview 1-liner**: *"Black-box evals test what users experience, not what developers wrote. Any white-box test of the LangGraph nodes is a unit test — the eval harness measures emergent quality of the full pipeline."*

---

## Decision 12: govinfo.gov Bulk XML for CFR Data (Not the eCFR API)

**Choice**: Download CFR XML volume files from `govinfo.gov/bulkdata/CFR/{year}/title-{num}/` rather than querying the eCFR REST API.

**Alternatives Considered**:
- eCFR API at `ecfr.gov/api/versioner/v1/` (paginated, real-time)
- Regulations.gov API (Federal Register / proposed rules focus)
- Web scraping `ecfr.gov` HTML
- Commercial legal data providers (Fastcase, Casetext, vLex)

**Why this**:
govinfo.gov bulk XML gives us the full title in one file per volume. Key advantages:
- **Completeness**: One download per volume captures every part and section in a title. The eCFR API requires paginated calls across potentially thousands of sections (Title 26 alone has ~7,000 sections).
- **Stability**: govinfo.gov is operated by the Government Publishing Office — the same authority that publishes the official CFR. The XML is the canonical machine-readable form.
- **Offline capability**: Downloaded files are cached locally (`cfr_data/`), so ingestion can run without internet after the initial download. The eCFR API would require live connectivity every time we re-ingest.
- **Format consistency**: govinfo.gov uses the same XML-derived tag structure across all titles. The eCFR API returns JSON, which loses the hierarchical part/subpart/section structure that we need for citation extraction.

**Why not eCFR API**: Rate-limited to ~1,000 requests/hour on the free tier. Getting all of Title 26 (Internal Revenue) via the eCFR API's `/full` endpoint would require hundreds of calls and ~20 minutes, versus a single 50MB file download from govinfo.gov.

**Why not Regulations.gov**: Regulations.gov focuses on the Federal Register (proposed rules, public comments). It does not serve the consolidated CFR as a searchable corpus.

**Why not scraping**: govinfo.gov's HTML listing is parseable, but their bulk data endpoint is designed for machine consumption — using it as intended is more stable than scraping.

**Why not commercial providers**: Fastcase and Casetext have excellent CFR data, but they charge per-call API fees. govinfo.gov is free, authoritative, and explicitly designed for bulk data access.

**Trade-offs**: govinfo.gov XML uses a tag-soup format (`<SECTION>`, `<SECTNO>`, `<P>`) inherited from SGML that differs from the clean USLM XML used for the U.S. Code. Our CFR parser handles this differently from the federal ingestion parser.

**Interview 1-liner**: *"govinfo.gov gives us the full CFR in one file per volume, offline-cacheable and from the official GPO source — the eCFR API would need hundreds of paginated calls and breaks if you lose connectivity mid-ingest."*

---

## Decision 13: CFR Titles 26 (Tax) and 29 (Labor) First

**Choice**: Start CFR ingestion with titles 26 and 29, not all 50 CFR titles.

**Alternatives Considered**:
- Ingest all 50 CFR titles (full Code of Federal Regulations)
- Start with a single title to minimize initial scope
- Let users configure any CFR title via env var
- Ingest only the parts directly referenced in our U.S. Code golden questions

**Why this**:
Titles 26 and 29 are the highest-value CFR titles for our existing corpus:
- **Title 26 (Treasury/IRS)**: Implements the Internal Revenue Code (26 U.S.C.). Our golden set already has 4 questions about 26 U.S.C. — the Treasury Regulations are what actually govern day-to-day tax compliance. A question about 401(k) plans is incomplete without 26 C.F.R. § 1.401(k)-2.
- **Title 29 (Labor/DOL)**: Implements FLSA, FMLA, ERISA, and OSHA (29 U.S.C.). The statute says "exempt executive employees" — the regulation (29 C.F.R. § 541.100) defines exactly what that means.

These two titles create immediate cross-source value: for every U.S. Code question in those domains, there is a corresponding CFR question that answers "but *how* does it work in practice?"

**Why not all 50 titles**: Title 26 alone is ~300MB XML across 22 volumes. Ingesting all 50 titles would require ~15GB of downloads and weeks of indexing time. Incremental expansion is the right approach.

**Why not a single title**: Two titles lets us test both the Tax domain (dense, technically complex) and the Labor domain (enforcement-heavy, different regulatory style). One title would not validate that the parser works across different CFR agencies.

**Trade-offs**: Titles 26 and 29 are two of the largest CFR titles. A more performance-conscious first iteration might have chosen smaller titles. We accepted this trade-off because size correlates with importance for our use cases.

**Interview 1-liner**: *"26 and 29 complement the U.S. Code titles we already have — every IRC statute question has a corresponding Treasury Regulation, and every FLSA question has a DOL regulation that defines the terms the statute left undefined."*

---

## Decision 14: `CFR_REGULATION` as a Distinct `QueryMode` (Not a Flag on `FEDERAL`)

**Choice**: Add `CFR_REGULATION = "cfr_regulation"` and `CROSS_SOURCE = "cross_source"` to the `QueryMode` enum as first-class modes, rather than extending `FEDERAL` with a `source_filter` parameter.

**Alternatives Considered**:
- Add `source_filter: list[str]` to `ChatRequest` (e.g., `source_filter=["cfr"]`)
- Add a `use_cfr: bool` flag to `ChatRequest`
- Route all regulation questions through FEDERAL mode with a Qdrant filter on `corpus = "cfr"`
- Single `MULTI_SOURCE` mode that always searches every corpus

**Why this**:
Mode is a *routing decision*, not a *filter parameter*. The distinction matters because:
- The LLM system prompt is different for each mode. A CFR answer needs `[Citation: X C.F.R. § Y.Z]` format and agency context ("this is an implementing regulation issued by IRS"). A federal statute answer needs `[Citation: X U.S.C. § Y]` format. Mixing these in a single mode would require convoluted prompt logic.
- The retriever is different. `CfrRetriever` queries `cfr_corpus`; `FederalRetriever` queries `federal_corpus`. Conflating them forces the retriever to decide which collection to query based on a flag — that's routing logic in the wrong layer.
- The `classify_mode` node already makes routing decisions based on query text. Adding CFR as a distinct mode means the router's `cfr_score` signal drives a clean branch, not a flag that leaks through all downstream nodes.
- Mode isolation (Decision 3) is expressed as a `QueryMode` enum value. Adding CFR as a mode keeps mode isolation checkable with a simple enum comparison.

**Why not `source_filter` param**: It puts routing logic in the API caller, requiring clients to understand which source to query. The router exists precisely to make that decision from the query text — clients should not need to know whether their tax question goes to `federal_corpus` or `cfr_corpus`.

**Why not a single `MULTI_SOURCE` mode**: Searching all corpora for every query is expensive (3× embedding calls) and adds noise for questions that clearly target one source. A user asking "what does 18 U.S.C. § 1344 say?" should not get CFR results mixed in.

**Trade-offs**: Each new mode adds a branch to `classify_mode`, `retrieve_context`, `generate_answer`, and `verify_answer`. The overhead is justified by the clarity of separation, but the agent file grows with each mode addition.

**Interview 1-liner**: *"Mode is a routing decision that determines the system prompt, the retriever, and the citation format. Collapsing it into a boolean flag would scatter routing logic across every node in the graph."*

---

## Decision 15: `CROSS_SOURCE` Mode Merges by Score After Parallel Retrieval

**Choice**: In `CROSS_SOURCE` mode, retrieve from both `federal_corpus` and `cfr_corpus` independently, then merge by relevance score and truncate to `top_k`.

**Alternatives Considered**:
- Weighted merge: multiply federal scores by α and CFR scores by β
- Sequential retrieval: fetch federal first, then use top federal result to query CFR
- Separate retrieval in two graph nodes with a dedicated merge node
- Always return a fixed split (e.g., 5 federal + 5 CFR regardless of scores)

**Why this**:
Score-based merge respects relevance signals. If the query is more closely described by CFR text than statute text (e.g., "what is the ADP safe harbor for 401(k)?"), the merge will naturally surface more CFR results — because the CFR section that answers the question will score higher than the statute sections that merely authorize the safe harbor to exist.

The alternative — a fixed split — would always return 5 statute chunks and 5 CFR chunks even when 8 of the top 10 most relevant chunks are from one source. That would bury high-signal evidence behind low-signal filler from the other corpus.

**Why not weighted merge**: Weights would need to be tuned per domain. Tax questions might need higher federal weight (the statute is more authoritative); FMLA questions might need higher CFR weight (the regulation defines everything). Without a principled way to set weights, score-based merge is the correct default.

**Why not sequential/chained retrieval**: Using the first retrieval result to query the second corpus (RAG-fusion style) can work but introduces ordering bias — the CFR retrieval would be anchored to whatever federal chunks happened to rank first, not to the original query intent.

**Why not a separate merge node**: The current graph is already 13 nodes. For the first implementation, inline merging inside `retrieve_context` keeps the graph topology unchanged. A dedicated `source_merger` node is designed for Week 3 when conflict detection is added.

**Trade-offs**: Score-normalization across two different collections is imperfect — `cfr_corpus` and `federal_corpus` are trained on the same embedding model (`text-embedding-3-small`) so scores are comparable, but there's no guarantee the score distributions are identical across collections.

**Interview 1-liner**: *"Score-based merge lets relevance drive the result — if the CFR text actually answers the question better than the statute, the merge surfaces it. A fixed split would bury the most relevant evidence behind arbitrary quota balancing."*

---

## Decision 16: CFR XML Tag-Soup Parser with Multiple Fallback Strategies

**Choice**: Write a custom `CfrXmlParser` using `lxml` with three fallback levels: structured `<SECTION>`/`<SECTNO>` parsing → `<DIV8 TYPE="SECTION">` parsing → paragraph-boundary splitting → character-boundary splitting.

**Alternatives Considered**:
- Parse only against a strict schema and skip non-conforming files
- Use `BeautifulSoup` with `html.parser` for lenient tag-soup parsing
- Convert XML to plain text first, then apply regex section detection
- Use the eCFR API JSON (no XML parsing needed)

**Why this**:
CFR XML from govinfo.gov is derived from a legacy SGML format converted to XML. Different volumes of the same title may use different tag conventions — some use `<SECTION><SECTNO>§ 1.401(a)-1</SECTNO>` while others use `<DIV8 N="§ 1.401(a)-1" TYPE="SECTION">`. A strict parser fails on any volume that uses a non-primary tag convention; a fallback-based parser degrades gracefully.

The three-level fallback mirrors real-world practice with government XML:
1. **Structured parse** (`<SECTION>` + `<SECTNO>`): Ideal; preserves section number and heading.
2. **Paragraph split**: When a section parses but is too large for one chunk.
3. **Character split**: Last resort for sections with no parseable paragraph structure.

**Why not strict-schema-only parsing**: Would fail silently on the volumes that use the `<DIV8>` tag convention, returning zero chunks for those volumes. We'd ingest Title 26 Vol 1 but miss Vols 3–22.

**Why not BeautifulSoup**: `lxml` with `recover=True` handles malformed XML as well as BeautifulSoup does, but lxml is already in our dependencies (`lxml==5.3.0` for USLM parsing). Adding BeautifulSoup would be an extra dependency for equivalent functionality.

**Why not plain text + regex**: Regular expressions for section detection in CFR text are fragile. The section number format `§ 1.401(a)-1` is hard to distinguish from cross-references (`see § 1.401(k)-2`) in plain text. The XML structure encodes exactly which text is a section number vs. body text.

**Trade-offs**: The multi-level fallback means some chunks may lack a parsed section number (fallback to character-boundary splitting assigns no `canonical_citation`). These chunks still get indexed but are harder to cite. In practice this affects <2% of sections in Titles 26 and 29.

**Interview 1-liner**: *"Government XML is real-world messy — different volumes of the same CFR title use different tag conventions. A strict parser fails half the corpus silently; fallback-based parsing is how you build for production data, not ideal data."*

---

## ADR-013: Separate repo for eval harness
**Decision:** legal-rag-eval lives in its own GitHub repository,
not as a subfolder of the legal-rag project
**Alternatives considered:**
- Subfolder inside legal-rag (e.g. legal-rag/eval/)
- Monorepo with shared CI
- Inline pytest tests inside the RAG backend
**Reasoning:** A separate repo makes the eval harness independently
usable — it can evaluate any RAG system that speaks the same API
contract, not just this one. It also gives the eval its own version
history, its own CI pipeline, and its own README, making it a
standalone portfolio artifact.
**Tradeoff accepted:** Requires both repos to be cloned locally for
development. Documented clearly in README.

## ADR-014: Hand-curated golden dataset over synthetic generation
**Decision:** 28 Q&A pairs manually written and verified
**Alternatives considered:**
- LLM-generated dataset (fast, scalable)
- Existing legal benchmarks: CUAD, LegalBench, ContractNLI
- User feedback collection from deployed system
**Reasoning:** LLM-generated ground truth defeats the purpose —
if the same model family generates both the dataset and the answers,
hallucinations in the dataset go undetected. Legal citations are
especially dangerous to auto-generate because LLMs confidently
produce plausible-sounding but non-existent statute numbers.
CUAD and LegalBench test contract review and NLI tasks, not
federal statute Q&A. User feedback requires a deployed system
with real users.
**Tradeoff accepted:** Small dataset (28 questions). Manual effort
to expand. Acceptable for baseline — dataset grows with the corpus.

## ADR-015: GPT-4o-mini as the eval model
**Decision:** EVAL_MODEL = "gpt-4o-mini" for DeepEval metric judgments
**Alternatives considered:**
- GPT-4o: higher judgment accuracy but ~10x cost per run
- Claude Sonnet: strong reasoning but adds a second API vendor dependency
- Local model via Ollama: free but inconsistent judgment quality and
  slow in CI environments
- No LLM judge (pure heuristics only): misses semantic hallucination
**Reasoning:** Eval runs on every nightly CI job across 28+ questions.
GPT-4o-mini keeps cost under $0.10 per full run while providing
sufficient judgment quality for hallucination and relevancy scoring.
The custom citation_accuracy metric is deterministic and does not
use an LLM judge at all, which is where legal accuracy matters most.
**Tradeoff accepted:** Slightly lower judgment accuracy than GPT-4o
for hallucination detection. Acceptable because citation_accuracy
(the highest-stakes metric) is deterministic.

## ADR-016: Nightly CI schedule with manual trigger
**Decision:** Eval runs nightly at 2am UTC and on manual dispatch
**Alternatives considered:**
- Every PR: maximum regression safety but expensive and slow
- Every merge to main: catches regressions faster but still costly
- Manual only: no automation, easy to skip
- Weekly: too infrequent for active development
**Reasoning:** Legal RAG changes slowly — ingestion runs are
infrequent and prompt changes are deliberate. Nightly is sufficient
to catch regressions introduced during a day of development without
making every PR expensive. Manual dispatch allows on-demand runs
before releases or after major changes.
**Tradeoff accepted:** Regressions introduced and fixed within the
same day won't appear in CI history. Acceptable given the development
cadence of this project.

## ADR-017: Regex-based source attribution metric
**Decision:** Use regex signal detection to score whether the RAG
response attributes each factual sentence to a legal source.
Three signal classes are checked per sentence: Federal (USC pattern,
§ symbol), CFR (C.F.R. pattern), and Case Law (Party v. Party,
reporter citations, "the Court held" phrases). Score = attributed
sentences / total substantive sentences.
**Alternatives considered:**
- NLP dependency parsing: higher precision but adds spaCy/stanza
  dependency and is slow in CI
- Embedding similarity to retrieval_context: measures relevance not
  attribution; a response can be relevant but uncited
- Manual rubric via LLM judge: too expensive to run per-sentence for
  every question
- Exact citation match per sentence: too strict — the RAG may
  paraphrase a citation correctly without exact formatting
**Reasoning:** Attribution is a structural property (does the sentence
reference a source?) more than a semantic one. Regex on well-known
legal citation patterns is deterministic, fast, zero-cost, and
precise enough for the sentence-level granularity we need. The metric
applies only to case_law and cross_source questions where multi-source
attribution is the primary evaluation concern.
**Tradeoff accepted:** Regex misses non-standard citation styles and
cannot detect attributions phrased as "as noted above" or "per the
above statute." These edge cases are rare in a properly prompted RAG.

## ADR-018: Binary conflict detection with partial credit for hedging
**Decision:** Score conflict detection as 1.0/0.5/0.0 rather than a
continuous score. For expected_conflict=true: 1.0 if a strong conflict
signal is present, 0.0 otherwise. For expected_conflict=false: 1.0 if
clean, 0.5 if minor hedging words appear without specific conflict
vocabulary.
**Alternatives considered:**
- Continuous scoring via keyword density: over-rewards verbose responses
  that happen to use conflict vocabulary in passing
- LLM judge: more nuanced but adds cost and non-determinism
- Simple binary (1.0 / 0.0): ignores the hedging case; a response that
  says "however" without flagging the actual conflict is not fully wrong
- Sentence-level scoring: granularity not warranted for a binary task
**Reasoning:** Conflict detection is a yes/no question — did the RAG
notice the divergence between statute text and judicial construction?
The partial-credit tier (0.5) for minor hedging acknowledges that
"however" is not a false positive but is also not the explicit flagging
we want for expected_conflict=true questions. The 0.5 score pushes
the aggregate below the 0.80 threshold only when multiple questions
are hedged, which is the right sensitivity level.
**Tradeoff accepted:** The strong-signal regex list must be maintained
as new cross-source question types are added. A conflict signal
vocabulary that works for statute-vs-case-law may not generalize to
regulation-vs-regulation conflicts without extension.

## ADR-019: Hand-labeling `expected_conflict` in golden set
**Decision:** The expected_conflict field in cross_source golden set
entries is hand-labeled based on legal analysis of each pair, rather
than derived algorithmically.
**Alternatives considered:**
- Algorithmic detection via citation overlap: cannot detect narrowing
  constructions that apply the statute's text selectively
- Automated via LLM at dataset-build time: introduces noise; the golden
  set should be ground truth, not LLM opinion
- Infer from expected_answer wording: circular — the expected_answer
  itself is written based on the hand-labeled conflict determination
**Reasoning:** Legal conflicts between statute and case law require
jurisprudential judgment that no algorithm reliably performs. The five
cross_source questions in q039-q043 were labeled by analyzing whether
the Supreme Court's holding narrowed or extended beyond the statute's
plain text: q039 (§ 1346 vs. Skilling: true — Court narrowed) and
q043 (§ 12113 vs. Echazabal: true — EEOC/Court extended beyond text)
are conflicts; q040, q041, q042 are consistent interpretations.
**Tradeoff accepted:** Hand-labeling does not scale beyond ~50 entries
without significant legal review effort. For a 43-entry golden set this
is acceptable; at larger scale the labeling process should be formalized
with a legal SME sign-off checklist.

## ADR-001: Custom citation_accuracy metric instead of string match
**Decision:** Regex-based citation extractor with partial-credit scoring
**Alternatives considered:**
- Exact string match of expected_citation in response text
- ROUGE-L score between expected and actual answer
- Embedding cosine similarity between citations
- DeepEval GEvalMetric with custom rubric
**Reasoning:** Legal citations have structure. Exact match fails on
spacing/punctuation variants (18 USC 1344 vs 18 U.S.C. § 1344).
Partial credit (0.5) for right title, wrong section reflects genuine
retrieval signal — adjacent sections are not total misses. ROUGE and
embeddings measure the wrong thing (text overlap vs. citation precision).
**Tradeoff accepted:** Regex does not handle non-standard citation styles;
extends to case law citations only in source_attribution.py.

## ADR-002: DeepEval for hallucination and answer relevancy
**Decision:** Wrap DeepEval HallucinationMetric and AnswerRelevancyMetric
**Alternatives considered:**
- NLI models (DeBERTa): 512-token limit; GPU required; too slow in CI
- BERTScore: measures semantic similarity, not grounding in a third text
- Custom GPT-4 prompt: maintenance burden; DeepEval's prompt already tuned
- Skip hallucination entirely: citation accuracy cannot catch prose fabrication
**Reasoning:** Hallucination detection is a reading comprehension task —
LLMs solve it better than heuristics. DeepEval gives a maintained
implementation with structured output (which claim failed) useful for
debugging. SKIP_LLM_METRICS=true enables offline runs without failing CI.
**Tradeoff accepted:** Requires OPENAI_API_KEY at runtime.

## ADR-003: Strict retrieval mode isolation per data source
**Decision:** Each source (federal, CFR, case law) has its own Qdrant
collection and retriever; the agent never crosses retrieval modes silently
**Alternatives considered:**
- Single collection with source_type payload filter
- Unified retriever with source weighting
- Merging all sources at ingestion time
**Reasoning:** Legal provenance is not optional. Statute, regulation, and
case law have different precedential weight and different update cycles.
Silent mixing can surface a superseded regulation over an amended statute.
Separate collections guarantee semantic space isolation.
**Tradeoff accepted:** More Qdrant collections = more RAM. Acceptable
given that legal accuracy > infrastructure cost in this domain.

## ADR-004: LangGraph StateGraph for agent workflow
**Decision:** 13-node LangGraph StateGraph with typed GraphState TypedDict
**Alternatives considered:**
- Plain Python sequential functions with if/else routing
- LangChain AgentExecutor (ReAct pattern)
- Celery task chains for async pipeline
- Prefect/Airflow for orchestration
**Reasoning:** The workflow has conditional branching (retry on bad
retrieval, mode-dependent routing, verification loops). LangGraph makes
routing explicit and testable as conditional edges rather than nested
if/else. ReAct is for open-ended tool use; this workflow is deterministic.
**Tradeoff accepted:** LangGraph is a newer library with breaking changes.
Pinned to specific version in requirements.txt.

## ADR-005: Separate Qdrant collections per corpus
**Decision:** federal_corpus, uploaded_documents, cfr_corpus, case_law_corpus
**Alternatives considered:**
- Single legal_corpus collection with source_type metadata filter
- Two collections (federal + user_documents)
- One collection per U.S. Code title
**Reasoning:** Each corpus has different payload schemas (USC has
section_number; documents have upload_id; CFR has part identifier).
A union schema with many null values would be unmanageable. Collection-
level isolation also enables per-corpus embedding model changes later.
**Tradeoff accepted:** Each collection requires its own Qdrant index.
Acceptable at the scale of this project (~500k chunks for 8 titles).

## ADR-006: USLM XML parsing for U.S. Code ingestion
**Decision:** Parse official USLM XML from uscode.house.gov
**Alternatives considered:**
- PDF via PyMuPDF: loses hierarchy; section boundaries are fragile
- HTML scraping of uscode.house.gov: breaks on site redesigns
- GPO plain-text .txt files: no hierarchy, no reliable section detection
- Third-party APIs (CourtListener, Law.gov): CourtListener is for case law
**Reasoning:** USLM XML preserves the full hierarchy needed to generate
canonical citations (title → chapter → subchapter → section). PDFs and
plain text have no machine-readable structure — section boundaries require
heuristics that fail on government document formatting conventions.
**Tradeoff accepted:** USLM files are large (usc26.xml ~300MB). Parsing
requires lxml and namespace-aware traversal.

## ADR-007: CourtListener REST API for case law ingestion
**Decision:** CourtListener API v4 for federal opinions
**Alternatives considered:**
- Scrape Google Scholar: prohibited by ToS; no structured citation metadata
- CourtListener bulk download: gigabytes; we only need filtered subsets
- Westlaw/LexisNexis API: $100-300/month; CourtListener is free
**Reasoning:** CourtListener is a 501(c)(3) with explicit API terms
allowing programmatic access. The API returns structured JSON with
case_name, citation, court, and precedential_status. Filtering by
Published status and federal jurisdiction gives us only authoritative
opinions without downloading the full dataset.
**Tradeoff accepted:** CourtListener coverage is incomplete for older
opinions and state courts. Acceptable for this federal-law-only corpus.

## ADR-008: Conflict annotation in source merger instead of silent ranking
**Decision:** When a statute and case law appear contradictory, annotate
the conflict rather than silently returning the higher-ranked chunk
**Alternatives considered:**
- Return highest-scoring chunk only (silence the conflict)
- Let the LLM decide which source to trust in the prompt
- Return both without annotation
**Reasoning:** In the legal domain silence is dangerous. A circuit-specific
interpretation returned as universal law, or a pre-amendment case returned
without noting the statute changed, can cause real harm. Annotation
([CONFLICT DETECTED]) surfaces the ambiguity to users who can then
consult counsel. The LLM has no knowledge of circuit geography or amendment
dates — these are factual, not reasoning, determinations.
**Tradeoff accepted:** Conflict detection uses keyword/negation matching
with false-positive risk. Week 3 adds the conflict_detection eval metric
to measure how well the RAG surfaces real conflicts.

## ADR-009: pydantic-settings BaseSettings for configuration
**Decision:** class Settings(BaseSettings) with env file at infra/.env
**Alternatives considered:**
- Scattered os.environ.get() calls throughout codebase
- python-dotenv with manual os.getenv() + type casting
- Dynaconf (multi-environment library)
- Hardcoded values in source code
**Reasoning:** BaseSettings gives typed fields, automatic env var
casting (str → int, str → list), field validators, and a single
auditable source for all configuration. Scattered os.getenv() calls
have no validation, no type safety, and no documentation of what
env vars the application expects.
**Tradeoff accepted:** pydantic v2 + pydantic-settings v2 have breaking
changes from v1. Both are pinned to specific versions.

## ADR-010: Retriever injection via GraphState (not singletons)
**Decision:** Retriever instances created per-request, injected as
GraphState fields (federal_retriever, document_retriever, cfr_retriever)
**Alternatives considered:**
- Global singleton retrievers initialized at startup
- Thread-local storage per request
- Factory function called inside each graph node
- FastAPI Depends() injection
**Reasoning:** State injection makes dependencies explicit and mockable.
Tests inject a MockRetriever without patching globals. Thread safety is
guaranteed because each request has its own state dict. New retrievers
(cfr_retriever in Week 2) are added as one new GraphState field and one
injection in the service layer — no changes to existing nodes.
**Tradeoff accepted:** Marginal per-request overhead for object allocation.
The Qdrant client is reused via module-level singleton.

## ADR-011: Black-box eval harness that only calls /chat
**Decision:** Eval harness makes only POST /api/v1/chat calls — no
direct imports or calls to RAG Python code
**Alternatives considered:**
- Import RAG modules and call agent graph directly (white-box)
- Use RAG's /retrieval endpoint to test retrieval separately
- Test individual LangGraph nodes in isolation
**Reasoning:** Black-box testing validates what users actually experience.
It works against any deployment (local, staging, production, a competitor)
without needing the RAG codebase. Calling /chat validates the entire
stack — routing, retrieval, generation, verification, response schema.
White-box tests of individual nodes don't catch integration failures.
**Tradeoff accepted:** Cannot inspect whether failures are retrieval or
generation failures. rag_mode and rag_confidence fields in the report
provide partial signal.

## ADR-012: govinfo.gov bulk XML for CFR ingestion (not eCFR API)
**Decision:** Download CFR volume files from govinfo.gov/bulkdata/CFR/
**Alternatives considered:**
- eCFR API at ecfr.gov: rate-limited; Title 26 would need hundreds of
  paginated calls vs one 50MB file download
- Regulations.gov API: focuses on Federal Register, not consolidated CFR
- Web scraping ecfr.gov HTML: fragile; bulk data endpoint is designed for
  machine use
- Commercial providers (Fastcase, Casetext): per-call fees
**Reasoning:** govinfo.gov is operated by the Government Publishing Office
— the same authority that publishes the official CFR. One file per volume
captures every section, is offline-cacheable after the initial download,
and uses a consistent XML structure across all titles. The eCFR API would
require live connectivity every re-ingest and hundreds of calls for
large titles.
**Tradeoff accepted:** govinfo.gov XML uses tag-soup SGML-derived format
that differs from clean USLM XML; requires a separate parser with
multi-level fallback strategies.

## ADR-020: Legal domain as eval target
**Decision:** Federal law as the domain for this eval harness
**Alternatives considered:**
- Medical RAG (high stakes but ground truth harder to verify)
- Financial RAG (good but citation format less standardized)
- General purpose RAG (low stakes, harder to show why eval matters)
**Reasoning:** Legal citations are deterministic and verifiable.
Either 18 U.S.C. § 1344 exists or it doesn't. This makes
citation_accuracy a concrete defensible metric rather than
subjective judgment. The high-stakes nature of legal hallucination
makes the "why eval matters" story immediately obvious to any
interviewer or hiring manager.
**Tradeoff accepted:** Requires domain knowledge to curate golden
dataset accurately. Smaller demo audience than general RAG.
