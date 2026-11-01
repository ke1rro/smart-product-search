# Smart Product Search — Product Requirements & Technical Design

**Status:** Proposed MVP specification for UCU NLP Assignment 1\
**Language:** English queries and English product text only\
**Dataset:** Amazon ESCI English Small, entire published subset\
**Implementation:** To be carried out later by Codex. This document is a specification, not an implemented system.

## 1. Product vision

Build a locally runnable, AI-assisted product search application. A customer types a natural-language English shopping query. The application searches a historical catalog of Amazon product descriptions, semantically reranks candidates, predicts the relationship of shortlisted products to the request, and presents results in meaningful groups with short, grounded explanations.

This is a **functional shopping discovery application**, not a notebook-only ML benchmark, a general-purpose chatbot, or a vanilla RAG question-answering system.

### User problem

Ordinary keyword matching may miss semantically appropriate products, may mix the requested item with accessories, and may fail to distinguish a suitable alternative from an exact match. We want the interface to distinguish these cases, not merely display ten titles.

### Primary user story

> As a shopper, I want to enter a product request in English and receive matching products, possible alternatives, and useful accessories, so I can understand which results fit my intent and why.

### Example query (illustrative; not a dataset guarantee)

`wireless noise cancelling headphones`

Expected behavior: identify candidate headphones; present predicted exact matches separately from substitutes and complementary accessories, if present; explain matches with reference to actual title, bullet points, and description. If the system cannot find reliable matches, say so.

## 2. Fixed scope

**Must have**

01. Search over **all distinct products represented in the English Small subset**, not a handpicked product sample.
02. A text query field for English shopping requests.
03. Lexical candidate retrieval with SQLite FTS5/BM25.
04. Semantic reranking by the locally executed pretrained `sentence-transformers/all-MiniLM-L6-v2` model.
05. ESCI relationship classification for shortlisted candidates by a remotely hosted LLM through the OpenRouter API.
06. Strictly validated and grounded per-product classifications and concise explanations.
07. A Streamlit UI with result groups: **Exact Matches**, **Alternatives**, **Accessories / Complements**. Irrelevant items are not promoted as matches.
08. Honest failure handling: retrieval-only/semantic results may remain visible if the LLM is unavailable, but **must be marked unclassified**.
09. Evidence-based component-level evaluation, fixed train/test boundaries, and a runnable local demo.
10. Appropriate documentation, CRISP-DM reflection, `AGENTS.md`, and agentic development skills as required by the course.

**Explicitly out of scope (MVP)**

- Real-time Amazon integration, actual shopping carts, purchases, seller stock, delivery promises, or up-to-date prices.
- Real product pictures or product image generation.
- Personalized recommendations based on accounts, browsing histories, or user profiles.
- Multilingual search, automatic query translation, price filtering, rating filtering, or inventory filtering.
- Training/fine-tuning an LLM; multiple competing models; large ablation studies; agent loops or orchestration frameworks.
- Guaranteeing that every relevant product in Amazon is retrievable. The catalog is restricted to products in the published subset.
- Claims that Amazon or a merchant endorsed the predictions.

## 3. Data contract

Each dataset example represents a **query–product judgment**, not an independent catalog item. Important fields:

| Field                            | Meaning / role                                               |
| -------------------------------- | ------------------------------------------------------------ |
| `query_id`, `query`              | Query identifier and search text; group evaluations by query |
| `example_id`                     | Original judgment identifier                                 |
| `product_id`, `product_locale`   | Stable product identity; deduplicate catalog by both         |
| `product_title`                  | Main product listing field                                   |
| `product_bullet_point`           | Additional product attributes                                |
| `product_description`            | Longer, possibly HTML-containing product information         |
| `product_brand`, `product_color` | Optional metadata, only show when present                    |
| `esci_label`                     | Ground-truth E/S/C/I judgment **for evaluation only**        |
| `split`                          | Published `train` or `test`                                  |
| `small_version`                  | Must equal `1` in the selected subset                        |

The catalog is built from **unique product IDs and product metadata**, with one record per `(product_locale, product_id)`. Keep judgments in a separate table keyed by query and product IDs. **Never index the ground-truth label or attach it to a product card at inference time.**

Product text is historical, incomplete, and sometimes noisy or HTML-marked. Render it as text, not as trusted HTML. Product attributes not present in the catalog are unknown, not zero or false.

## 4. E/S/C/I classification semantics

- **E — Exact:** Fulfills the user's specified product type and stated important attributes; may be a different brand when brand is not constrained.
- **S — Substitute:** A plausible alternative serving the underlying purpose but not an exact match for all stated requirements.
- **C — Complement:** An accessory or product used **with** the requested item, rather than replacing it.
- **I — Irrelevant:** Does not meaningfully fulfill or complement the request.

Use Amazon ESCI's definitions as the reference label taxonomy. Negations, numeric specifications, compatibility information, quantities, and explicitly named brands may be essential. Do not classify an accessory as the requested item merely because its text mentions the item.

Prediction is contextual to the user query. The **same product may get different labels for different queries**. Never store a single permanent ESCI class on a product.

## 5. End-to-end architecture

```text
English shopping query
        |
        v
[Input validation + query normalization]
        |
        v
[SQLite FTS5 / BM25 candidate retrieval]  --> top 100 lexical candidates
        |
        v
[Local MiniLM semantic reranking]        --> rerank top 50 candidates
        |
        v
[Select top 10 product candidates]
        |
        v
[OpenRouter LLM, single batched request] --> E/S/C/I + short evidence-based reason
        |
        v
[Schema validation + business rules]
        |
        v
[Streamlit grouped search results]
```

These numbers (100 retrieved, 50 semantically reranked, 10 classified) are **initial MVP configuration defaults**, not measured optimal parameters.

### 5.1 Query preprocessing

**Input:** A user's text. **Output:** a normalized query and validation result.

- Reject blank or unreasonably long queries with an actionable UI message.
- Normalize whitespace and text encoding. Do not discard negations (`without`, `not`), units, sizes, model numbers, or brand identifiers.
- Preserve original text for display and LLM understanding.
- Treat user input as data; it must not override system instructions.

**Verification:** deterministic tests for empty queries, casing, whitespace, brands, model numbers, and negated requirements.

### 5.2 Lexical retrieval: SQLite FTS5 / BM25

**Input:** normalized query. **Output:** ranked candidate product IDs and lexical scores.

- Create a local **full-text search** index over product title, brand, bullets, and cleaned description.
- Favor title and identifying attributes over long descriptions using weighted ranking or field weights; document the weights.
- Return up to 100 candidate records; do not use labeled query–product pairs to fabricate the candidate list.
- Distinguish SQLite FTS5's BM25 sorting convention: **smaller score means better rank** for its built-in `bm25()` ranking function.
- Handle zero hits without pretending semantic reranking can recover products that were never retrieved.

**Known limitation:** BM25 candidate generation can miss semantic matches without shared vocabulary. A reranker cannot recover a missing candidate. Record this limitation in the final report rather than expanding the MVP by default.

**Evaluation:** judged-relevant retrieval Recall@100 on held-out ESCI queries, with the explicit caveat that only annotated products are treated as known relevant; unjudged products are unknown.

### 5.3 Semantic reranking: local pretrained model

**Input:** query + up to 50 BM25 candidate records. **Output:** cosine similarity / semantic rank per candidate.

- Use `sentence-transformers/all-MiniLM-L6-v2` locally (384-dimensional embeddings; model truncates inputs exceeding 256 word pieces).
- Build product text from title, brand, bullet points, and a **bounded** cleaned description. Never let long HTML descriptions crowd out the title and key attributes.
- Encode the query once; batch-encode candidates and cache product embeddings for repeated searches if helpful.
- Compute cosine similarities with normalized vectors.
- Rank with a deterministic fusion of lexical rank and semantic rank (for example, reciprocal rank fusion); specify the formula and constant in configuration. Do **not** directly sum raw BM25 values and cosine similarity: they are not on comparable scales.
- Return top 10 candidates to the LLM.

**Evaluation:** nDCG@10 using graded ESCI judgments on a documented, **judged-candidate** evaluation setting; distinguish this from end-to-end catalog retrieval metrics.

**Known limitation:** MiniLM is a general sentence embedding model, not an ESCI-trained classifier; a high cosine similarity does not prove a product is an Exact match.

### 5.4 OpenRouter LLM classification

**Input:** original query and top 10 products containing stable product IDs and selected **real** metadata. **Output:** one valid ESCI label and short, grounded explanation per product.

- Use **one batched chat completion per search**, not ten individual requests.
- Keep the model ID configurable and pinned for reproducible measurements. A free model is sufficient if its endpoint supports the required output contract. Do not silently substitute a paid model.
- Prefer strict JSON Schema output for a compatible provider; verify parameter support for the selected model + endpoint. If unsupported, request JSON and validate it locally, without claiming schema enforcement by the API.
- Require exactly one output per candidate product ID; reject duplicates, unknown IDs, missing entries, unknown class labels, or invented product fields.
- The model must be told to classify **using only provided product data**, avoid invented specifications, and express uncertainty through an honest reason rather than fabricated product attributes.
- Reasons should briefly refer to evidence in the listing (for example, relevant item type, size, compatibility claim, or missing key attribute). Do not fabricate citations or pretend explanations are gold labels.
- Strip HTML from descriptions before sending; limit total input tokens/characters; treat listing text as untrusted data to reduce prompt-injection risk.
- Limit retries, respect API rate limits, avoid unnecessarily sending repeated requests, and cache outputs keyed by query + ordered product IDs + model ID + prompt version.
- Log request ID, model ID, latency, status, response validation outcome, and token usage if available. Never log the API secret.
- A model failure must not be treated as a genuine `I` prediction. Show **Unclassified** results or an explicit non-AI mode.

**Example response schema (illustrative, never ground truth):**

```json
{
  "results": [
    {
      "product_id": "B012345678",
      "label": "E",
      "reason": "The listing describes wireless headphones with noise cancellation."
    },
    {
      "product_id": "B087654321",
      "label": "C",
      "reason": "This is a protective case for headphones, not the headphones themselves."
    }
  ]
}
```

The real system must produce entries for **all** requested candidates, with IDs copied from the actual retrieval response.

**Evaluation:** ESCI Macro-F1 plus per-class precision/recall/support for a fixed query-grouped subset of **held-out** test judgments. Evaluation uses the supplied, labeled query–product candidates to isolate classification; it is separate from live-search evaluation. Do not include the answer labels in the LLM prompt.

### 5.5 Business logic and final ranking

**Input:** retrieved candidate metadata and validated LLM classifications. **Output:** grouped product cards.

1. Preserve product identity and real catalog fields.
2. Group `E` as **Exact Matches**, `S` as **Alternatives**, and `C` as **Accessories / Complements**.
3. Exclude predicted `I` results from the default visible shopping recommendations; optionally show an informational count.
4. Preserve the deterministic search rank inside each group rather than trusting explanation prose as a score.
5. Display a clear message if no exact matches exist; show alternatives when available.
6. Display a distinct **unclassified** state if LLM predictions are unavailable; never show inferred group labels as validated facts.
7. Deduplicate product IDs across groups and reject missing metadata gracefully.

**Evaluation:** deterministic unit tests of grouping, ordering, deduplication, invalid labels, and fallback status; manual end-to-end checks on fixed example queries.

### 5.6 Streamlit UI

**Primary elements**

- App title, one search bar, Search action, and loading indicators for retrieval/reranking/classification.
- Three visible result sections for predictions, with product title, brand, relevant text excerpts, predicted class, and short explanation.
- Optional compact technical details: candidate count, retrieval and model latency, model name, and whether predictions are currently available.
- Honest empty-state, bad-query, rate-limit, API-error, and dataset-missing messages.
- Clear distinction between **predicted** relevance labels and annotated benchmark labels (the latter must never be displayed as live predictions).

**User-facing behavior:** never show prices, stock, ratings, images, or purchase claims that are absent from the dataset. All text must be legible and practical to demonstrate within a short recorded session.

## 6. End-to-end acceptance scenarios

| Scenario                               | Expected behavior                                                       |
| -------------------------------------- | ----------------------------------------------------------------------- |
| Common product query                   | Product cards returned and grouped when LLM succeeds                    |
| Query with specific brand/size         | Model sees and considers the precise constraint                         |
| Query with negation (`without wheels`) | Negation is preserved and passed to all relevant components             |
| Accessory-heavy query                  | Complements are not mislabeled as exact items merely by keyword overlap |
| No lexical matches                     | Honest empty results, no fabricated alternatives                        |
| OpenRouter 429 / timeout               | Search still usable in **unclassified** mode; error shown               |
| Invalid JSON / missing IDs             | Reject invalid predictions; do not silently assign random classes       |
| Missing product description            | Handle missing data without invented details                            |
| Repeated identical query               | Reuse valid cached predictions when caching is enabled                  |

These cases validate **behavior**, not promised real-world quality. Ground-truth scores must be measured separately.

## 7. Evaluation policy — minimum only, not an ablation study

| Component                      | Minimal meaningful check                                 | Data                                                     |
| ------------------------------ | -------------------------------------------------------- | -------------------------------------------------------- |
| Input preprocessing            | Unit cases for preservation and normalization            | Deterministic fixtures                                   |
| BM25 retrieval                 | Judged-relevant Recall@100                               | Test judgments; incomplete-relevance caveat              |
| MiniLM reranking               | nDCG@10 on fixed judged candidate pools                  | Test judgments with E/S/C/I graded relevance             |
| OpenRouter ESCI classification | Macro-F1; classwise precision/recall/support             | Fixed labeled test subset, gold labels hidden from model |
| Grouping/business rules        | Unit tests of group order, filtering, failures           | Deterministic fixtures                                   |
| Overall application            | Fixed manual demo checklist and full pipeline smoke test | Real catalog, no mock model success claims               |

**Critical evaluation distinction:** E/S/C/I is not inherently a simple total order for all use cases (a Complement is useful but not interchangeable with an Exact item). Document the explicit graded relevance mapping chosen for nDCG and make this metric a scoped ranking proxy, not an absolute definition of utility. For example, an Exact-focused search evaluation might use `E=3, S=2, C=1, I=0`; accessory use cases need a different interpretation. Do not silently treat predicted labels as true labels.

No requirement to train several models or conduct repeated tuning experiments. Report measured values only after running the evaluation, including sample size, selected queries, limitations, and error examples.

## 8. Practical, nonfunctional constraints

- Must run on an ordinary laptop without CUDA or a local LLM. Internet is required only for initial dataset/model download and the OpenRouter classification step.
- Offline retrieval and semantic ranking remain usable once the dataset and embedding model are cached.
- Use Python 3.14 and `uv` if compatible with selected package versions; if a dependency is incompatible, report and resolve the smallest supported alternative rather than hiding a failure.
- Prefer the repository's existing engineering rules in `AGENTS.md`; no unrelated frameworks or services.
- Credentials come from `.env` / environment configuration and must never be committed.
- Keep generated dataset, indexes, embeddings, and LLM caches out of Git.
- Respect the original dataset license and attribution. Store source URL and revision in dataset metadata.

## 9. Architecture decisions and tradeoffs

**Why three stages?** A full LLM scan over hundreds of thousands of products would be too slow and expensive. BM25 narrows the corpus cheaply; MiniLM improves candidate ordering semantically; the LLM uses the expensive contextual decision only on a small shortlist.

**Why no dedicated vector database?** The MVP only reranks tens of lexical candidates per request. Encoding those candidates in batches, with caching as needed, avoids unnecessary infrastructure.

**Why not train a classifier?** The fixed MVP goal is a usable product with a real model/API call and evaluation, not a model-training experiment. ESCI gold labels are primarily used to assess the result, not copied into live predictions.

**What does the dataset not prove?** It does not provide live retail data or exhaustive relevance labels for every possible catalog product. Classification and ranking quality outside the judged pairs cannot be asserted as fully measured.

## 10. Assignment deliverables and hand-off

When Codex implements this specification, it must produce a runnable prototype, setup/run commands, architecture diagram, CRISP-DM summary, evaluation results for each component, and material suitable for a video demo \<=10 minutes. The **development process** must demonstrate use of the repository `AGENTS.md` and create/use agent skills. All implementation steps and verification gates live in `IMPLEMENTATION_PLAN.md`.

## Sources

- Amazon Science dataset and ESCI semantics: https://www.amazon.science/code-and-datasets/shopping-queries-dataset-a-large-scale-esci-benchmark-for-improving-product-search
- Official dataset repository, paper reference and Apache 2.0 license: https://github.com/amazon-science/esci-data
- Exact English Small repack: https://huggingface.co/datasets/thebajajra/amazon-esci-english-small
- MiniLM model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- OpenRouter free models: https://openrouter.ai/collections/free-models
- OpenRouter structured output support: https://openrouter.ai/docs/features/structured-outputs
