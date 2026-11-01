# Smart Product Search — Codex Implementation Iterations & Verification Gates

**Document type:** Implementation plan, not implementation.\
**Working agreement:** Codex implements **one iteration at a time**, runs verification, reports results, and stops at the gate. Avoid unrelated refactoring, speculative components, or extra model comparisons.\
**References:** Read `AGENTS.md`, `PROJECT_IDEA.md`, and `DATA_SETUP.md` before any code changes.

## 0. Working principles

01. Deliver **one working Smart Product Search product**. Do not turn the course assignment into a research benchmark or sprawling multi-agent platform.
02. Match the current repository engineering conventions (`uv`, typing, linting/tests, readable Python, no invented verification results). Follow `AGENTS.md`; when a proposed library cannot support Python 3.14, report the incompatibility and select a justified solution.
03. Keep each code edit small and independently verifiable. Make the simplest implementation that meets the current iteration acceptance gate.
04. Do not commit, push, make branches, deploy, or incur paid API charges unless explicitly instructed.
05. Do not use dataset `esci_label` as an inference feature. All metrics must be computed against hidden ground truth.
06. No automatic model training/fine-tuning, no multi-model experiments, no novel orchestration architecture, no LangChain/LangGraph unless absolutely necessary and approved.
07. Build a usable local UI **before** polishing evaluation charts or the report.
08. Keep external actions explicit: dataset/model downloads and LLM API calls may require internet. Respect API limits, secrets, and the user's compute budget.
09. Write short, factual iteration reports: files changed, commands executed, actual outputs, failures, next work.
10. If a gate is not met, repair the existing iteration before moving on; never claim an unexecuted check passed.

## Implementation overview

| Iteration | Goal                                   | Required observable result                                   |
| --------- | -------------------------------------- | ------------------------------------------------------------ |
| 0         | Repository / agentic development setup | Clear instructions and runnable test tooling                 |
| 1         | Data download and validation           | Full 601,354 rows downloaded, verified; catalog built        |
| 2         | Lexical retrieval + simple UI          | Users can search real product metadata via BM25              |
| 3         | Semantic reranking                     | MiniLM reranks real BM25 candidates locally                  |
| 4         | LLM ESCI classification                | One API call classifies top 10 in strict validated structure |
| 5         | Product UX and failure handling        | Grouped cards, caching, honest fallback behavior             |
| 6         | Evaluation + course hand-in            | Component metrics, CRISP-DM, diagram, documented demo        |

These are **deliverable milestones**, not instructions to run experiments comparing alternative systems.

______________________________________________________________________

## Iteration 0 — Project skeleton and agentic workflow

### Codex tasks

- Inspect the existing repository and `AGENTS.md`; reuse style and tooling rather than replacing them.
- Place these three documentation files under `docs/` (or the chosen existing docs location); reference them from the README.
- Initialize `pyproject.toml` and `uv.lock` **only if absent**; add minimal direct dependencies as their implementation stage requires them, not an expansive speculative stack.
- Define lightweight configuration keys: `data_dir`, `index_path`, `retrieval_k=100`, `rerank_k=50`, `classify_k=10`, `embedding_model`, `llm_model`, and `openrouter_api_key` (secret via environment / `.env`).
- Create agent skill instructions that materially help development, e.g. `.agents/skills/dataset-audit/SKILL.md` and `.agents/skills/component-evaluation/SKILL.md` and optionally `submission-readiness/SKILL.md`. Existing project skills should be reused or revised, not duplicated. Ensure skills provide actionable tasks, verification, and expected artifacts.
- Add `.gitignore` for raw/processed data, indexes, caches, `.env`, and generated evaluation outputs.
- Prepare an initial README with prerequisites and **planned** run commands; clearly mark commands unavailable until implemented.

### Verification / acceptance gate

- Configuration parse tests pass; repository tools can run from root.
- `AGENTS.md` remains intact except for explicitly requested NLP-specific edits.
- Skills exist and are referenced in workflow documentation.
- No large data, secrets, or generated binaries tracked by Git.
- Codex reports the actual commands it ran.

______________________________________________________________________

## Iteration 1 — Full Amazon ESCI download, audit, and indexable catalog

### Codex tasks

- Implement `scripts/download_data.py`, `scripts/prepare_data.py`, `scripts/validate_data.py` per `DATA_SETUP.md`.
- Download the **complete** English Small `train` and `test` splits from `thebajajra/amazon-esci-english-small`, pinned to a recorded revision.
- Check exact record counts, locale, small-version flag, labels, and schema. Reject silent source substitution.
- Prepare a standalone unique-product catalog and separate labeled query judgments.
- Strip unsafe/noisy HTML for text indexing and LLM input while preserving meaningful specifications, product IDs, and provenance.
- Record a compact data audit JSON/Markdown with counts, nulls, label distributions, distinct products, and any duplicate metadata conflicts.

### Verification / acceptance gate

- Full data count is **419,653 train + 181,701 test = 601,354 rows**.
- All original judgments are preserved in train/test artifacts.
- Product catalog is unique by `(product_locale, product_id)`; deduplication policy is documented and deterministic.
- No `esci_label` appears in a public catalog text field or serving-time product representation.
- Data pipeline can run a second time safely without changing results.

**Stop if counts unexpectedly differ.** Report source revision and discrepancy before proceeding.

______________________________________________________________________

## Iteration 2 — Working lexical search and basic Streamlit UI

### Codex tasks

- Implement a local SQLite database with FTS5 indexing from the prepared catalog.
- Include title/brand/bullets/description; configure simple field weights and consistent normalization.
- Implement `retrieve_products(query, k=100)` with stable product IDs, scores, and clear no-hit behavior.
- Implement a minimal Streamlit app: search input and list of real product cards with titles and optional metadata. This intermediate UI does **not** claim to classify products.
- Ensure search works without dataset file parsing every time the UI loads.

### Verification / acceptance gate

- Index contains the same number of unique products as the prepared catalog.
- Querying a distinctive product title token returns sensible nonempty candidates.
- Missing and empty queries do not crash the app.
- Product cards show only real stored metadata, not placeholder prices or images.
- BM25 retrieval unit tests cover ordering, index lookup, empty results, and stable IDs.
- A developer can start the app locally using a documented `uv run streamlit ...` command.

______________________________________________________________________

## Iteration 3 — Local pretrained semantic reranking

### Codex tasks

- Integrate exactly one pretrained embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Add a wrapper for query encoding and batch candidate encoding.
- Apply semantic ranking only to top **50 BM25** candidates by default.
- Fuse lexical/semantic rank deterministically (e.g. RRF) and select final top 10 for LLM. Document the rank fusion formula and configured parameters.
- Preserve result identity and original text. Add only minimal caching needed for acceptable interactivity.
- Keep the UI functional as search + semantic ranking when no OpenRouter key is provided.

### Verification / acceptance gate

- Model loads on CPU; a search returns top 10 (or fewer when insufficient candidates) without API access.
- Test vectors are consistent in shape; similarity computation is finite and deterministic for fixed inputs.
- Results never contain unknown/duplicate product IDs.
- The UI labels semantic-only results **Unclassified**; no fabricated E/S/C/I classes.
- Record one real smoke-test search and its observed latency, without asserting a performance SLA.

______________________________________________________________________

## Iteration 4 — OpenRouter ESCI classifier

### Codex tasks

- Implement a small OpenRouter adapter with configurable `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, request timeout, and capped retries.
- Select and document **one free model that can actually serve the intended structured output**. Do not assume every free endpoint supports JSON Schema. Verify selected model endpoint capabilities before using `response_format=json_schema`; otherwise request JSON and parse/validate locally.
- Send original query + up to 10 shortlisted product IDs and bounded listing texts in **one batched request**.
- Ask for E/S/C/I and one short evidence-grounded reason per product. Define structured input/output contracts, with strict checks for all supplied candidate IDs and valid labels.
- Treat model text and product text as untrusted inputs; prohibit any tool calls or off-platform actions from model instructions embedded in listings.
- On 429/timeout/invalid response, show **Unclassified** semantic results and an explicit status message; do not mislabel failed candidates `I`.
- Cache valid outputs based on query, selected candidate IDs, model ID, and prompt revision to minimize free API calls.

### Verification / acceptance gate

- A real successful API call completes and returns **valid labels for every supplied product**; log the model ID and request count (without secrets).
- Mocked tests cover normal response, duplicate/unknown IDs, invalid label, malformed JSON, 429, timeout, and missing API key.
- No paid fallback is configured by default.
- If chosen free model cannot consistently return valid structured output, stop and document the API limitation; switch free endpoint only after verifying compatibility, without introducing a large comparison experiment.
- Model selection is pinned for later reproducible evaluation.

**Free API note:** The OpenRouter `:free` tier is rate-limited, commonly **20 requests/min and 50 requests/day** without required credit purchases; actual provider limits may be tighter. Check current limits and plan evaluation calls accordingly. Do not burn the daily budget with unnecessary reruns.

______________________________________________________________________

## Iteration 5 — Final shopping product UX, consistency, and resilience

### Codex tasks

- Convert validated LLM outputs into Exact, Alternatives, and Accessories groups.
- Keep search order deterministic inside each group; do not rank by invented model confidence.
- Show product title, brand where present, concise real excerpt, predicted relation, and one explanation.
- Hide predicted Irrelevant items from recommendation groups; optionally show a count.
- Clearly indicate predicted categories, as distinct from dataset ground truth.
- Handle no-hit queries, no-exact queries, missing optional metadata, rate limits, invalid outputs, and offline mode.
- Provide optional small technical panel with stage timings, candidate counts, chosen model ID, and classification status for explanation during the demo.
- Do not add accounts, payments, price sliders, stock statuses, external shopping APIs, or long chat history.

### Verification / acceptance gate

- Full app walkthrough succeeds on a documented small set of *real English queries* available in the dataset.
- Category grouping follows exactly validated model labels; no product in multiple groups.
- API failure leaves a useful searchable result page marked unclassified.
- Repeated identical search can reuse valid cached predictions when caching is enabled.
- All UI content is escaped/sanitized and product text is not interpreted as trusted instructions.
- No false claims about availability, price, or Amazon sponsorship.

______________________________________________________________________

## Iteration 6 — Minimal evaluation, CRISP-DM, and submission package

**Scope constraint:** Do **not** add a second embedding model, run multiple architecture sweeps, or fine-tune an LLM. Measure the chosen architecture's existing components.

### 6.1 Tests and metrics

| Stage               | Required measurement                                                      | Setup                                                                                         |
| ------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Input normalization | Deterministic case-based test pass count                                  | Query fixtures                                                                                |
| Lexical retrieval   | Judged-relevant **Recall@100**                                            | Full test queries or documented reproducible subset; catalog contains products without labels |
| Semantic reranking  | **nDCG@10**                                                               | Test queries with judged candidate pools; label E=3/S=2/C=1/I=0 if using exact-item relevance |
| LLM classification  | **Macro-F1**, per-class precision/recall/support                          | Fixed held-out test pairs grouped by query; never send gold labels                            |
| Grouping/validation | Test pass/fail count                                                      | Deterministic unit tests                                                                      |
| Full system         | Successful search-to-UI smoke test, observed latency and failure behavior | Real local app and remote API when available                                                  |

**Evaluation validity notes:**

- ESCI has **incomplete catalog-wide judgments**. Retrieval metrics describe recovery of *known judged relevant* products, not recall of every relevant item in the full catalog.
- For semantic reranking, use a **judged-pool** assessment so nDCG is not silently computed by assuming every unjudged catalog product is irrelevant. Report exactly how candidate pools are formed. This conditional measurement is different from end-to-end search quality.
- For LLM classification, take approximately **20–25 fixed held-out test queries**, up to 10 judged products each (about 200–250 labeled pairs depending on availability) and run one batched request per query. Use fixed seed `2026`, report sample selection criteria, class supports, model ID, failed calls, and number of API requests. If some classes do not occur, explicitly note that their performance is not assessed; choose a documented class-coverage sampling strategy without tuning to outputs.
- When limited by the free tier, a smaller well-documented evaluation is acceptable as a **small-sample estimate**. Never report scores for unrun calls or treat errors as correct predictions.
- Include at least a few concrete failure examples with a plausible reason for each failure.

### 6.2 Final documentation

Prepare a short report or README section with the six CRISP-DM stages:

1. **Business understanding:** why shoppers need precise matches, alternatives, and accessories.
2. **Data understanding:** provenance, label frequencies, missing metadata, and constraints.
3. **Data preparation:** deduplication, cleaning, separation of labels and serving catalog.
4. **Modeling:** BM25, pretrained MiniLM, OpenRouter LLM, and product grouping business rules.
5. **Evaluation:** component measurements and practical failure modes; no fabricated results.
6. **Deployment:** local Streamlit prototype, setup commands, offline/online behavior, limits.

Include a high-level architecture diagram, input/output contracts, commands to reproduce data setup and evaluation, dependency/secret instructions, and code-agent context-management notes. Cite the original Amazon paper and dataset/repack sources.

### 6.3 Ten-minute demo outline

| Approx. segment | Content                                                      |
| --------------- | ------------------------------------------------------------ |
| ~1 min          | Business problem and MVP scope                               |
| ~1 min          | Dataset, splits, and data preparation                        |
| ~2 min          | Architecture and purpose of each component                   |
| ~3 min          | Live app: normal query, alternative/accessory, failure state |
| ~2 min          | Evaluation metrics, actual outputs, errors/limitations       |
| ~1 min          | AGENTS.md, custom skills, CRISP-DM, final recap              |

Time allocations are illustrative; final video **must be at most 10 minutes**.

### Verification / acceptance gate

- `uv run pytest` and all project-configured format/lint/type checks were actually executed or failures explicitly recorded.
- Data audit, search index, and UI work with **the full English Small catalog**.
- A real LLM API call has been demonstrated, or its unavailability is transparently explained; no fake success.
- Evaluation script produces real metrics with sample sizes and clearly distinguishes judged-pool and catalog-wide contexts.
- README includes correct installation, data setup, run, eval, and troubleshooting instructions.
- Report describes each component's purpose, input/output, and measurement.
- CRISP-DM and agentic development method are visible in deliverables.

______________________________________________________________________

## Suggested minimal dependency groups (Codex verifies compatibility before adding)

- **Environment:** `uv`, `pyproject.toml`, `.env`/`decouple` following AGENTS conventions.
- **Data:** `datasets` or `huggingface_hub`, `pyarrow`, `polars`.
- **Search:** Python `sqlite3` with SQLite built with FTS5 (check availability explicitly).
- **Semantic model:** `sentence-transformers` and its compatible CPU dependencies.
- **LLM API:** a lightweight HTTP client or OpenAI-compatible client targeting `https://openrouter.ai/api/v1`.
- **UI:** `streamlit`.
- **Verification:** existing `pytest`, formatting, linting, and type-check configuration.

Do not install an additional vector DB, ML training framework beyond the embedding model's actual dependencies, workflow engine, or web server just for architectural decoration.

## Codex invocation prompts, one iteration at a time

**Start (Iteration 0):**

> Read `AGENTS.md`, `docs/PROJECT_IDEA.md`, `docs/DATA_SETUP.md` and `docs/IMPLEMENTATION_PLAN.md`. Implement **only Iteration 0** and its acceptance gate. Do not implement later stages, commit, or add unrelated features. Report files changed and exactly which checks executed.

**Continue (Iteration N):**

> Inspect the results of the previous iteration and the relevant specs. Implement **only Iteration N** from `docs/IMPLEMENTATION_PLAN.md`; run the stated tests and verification gate. If a gate fails, fix it or report the blocker. Do not start Iteration N+1 until explicitly requested.

## Research references for implementation details

- Amazon ESCI: https://github.com/amazon-science/esci-data
- Exact dataset: https://huggingface.co/datasets/thebajajra/amazon-esci-english-small
- MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- OpenRouter free models: https://openrouter.ai/collections/free-models
- OpenRouter structured-output constraints: https://openrouter.ai/docs/features/structured-outputs
