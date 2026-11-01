# Smart Product Search — Dataset Setup and Data Contracts

**Status:** Detailed instructions to be implemented later by Codex. No dataset is bundled with these specifications.\
**Source:** Amazon ESCI, English Small, entire dataset\
**Canonical download for this assignment:** `thebajajra/amazon-esci-english-small` on Hugging Face

## 1. Confirmed source and exact size

- Dataset page: https://huggingface.co/datasets/thebajajra/amazon-esci-english-small
- Repository ID: `thebajajra/amazon-esci-english-small`
- Source of original work: Amazon Science Shopping Queries Dataset, https://github.com/amazon-science/esci-data
- Dataset repository reference commit observed during specification preparation: `e7bb589a050171ed4b97891bd517507f702af0c2`. Pin a verified revision in implementation metadata; if inaccessible, record a substitute exact commit SHA.
- Dataset data files: `data/train-00000-of-00002.parquet`, `data/train-00001-of-00002.parquet`, `data/test-00000-of-00001.parquet`.
- Published **download size**: 398,348,357 bytes (~398 MB decimal). Published **logical uncompressed dataset size**: ~790,351,557 bytes. Allow additional space for the cache and the SQLite FTS index.
- Published example counts:

| Split     | Query-product judgments |
| --------- | ----------------------: |
| `train`   |                 419,653 |
| `test`    |                 181,701 |
| **Total** |             **601,354** |

- Published English Small summary: **29,844 distinct queries** (20,888 train queries, 8,956 test queries); average ~20.15 judged products/query.
- Filter identities: `product_locale == "us"`, `small_version == 1`, `esci_label in {"E", "S", "C", "I"}`.
- The Hugging Face packaging declares `apache-2.0`; the original Amazon Science repository also declares Apache 2.0. Preserve source attribution and license notices.

**DO NOT use `shuttie/esci-us-small` for this project without explicit approval.** Despite a similar name, its currently published rows are not the same English Small subset (`small_version=0` in its visible dataset viewer). Do not quietly substitute the large-version data.

## 2. Expected original columns

| Name                   | Type/description             | Required for                           |
| ---------------------- | ---------------------------- | -------------------------------------- |
| `example_id`           | numeric identifier           | Judgment provenance                    |
| `query`                | English search query         | Query data and evaluation              |
| `query_id`             | numeric query identifier     | Query-grouped splitting and evaluation |
| `product_id`           | ASIN-like product identifier | Catalog primary key                    |
| `product_locale`       | locale, expected `us`        | Product key + validation               |
| `esci_label`           | one of E/S/C/I               | **Evaluation only**                    |
| `small_version`        | expected 1                   | Dataset integrity validation           |
| `large_version`        | version flag                 | Metadata                               |
| `split`                | `train` / `test`             | Integrity validation                   |
| `product_title`        | name                         | Search and UI                          |
| `product_description`  | optional HTML-ish text       | Search + LLM evidence                  |
| `product_bullet_point` | optional feature list        | Search + LLM evidence                  |
| `product_brand`        | optional brand name          | Search + UI                            |
| `product_color`        | optional color/attribute     | Search + UI                            |

Ignore `__index_level_0__` when present; it is a serialization artifact, not a semantic key. Check actual schema before relying on any field. Validate that the native split and the row's `split` value agree.

## 3. How to download (instructions for Codex)

**Preferred:** Use the Hugging Face `datasets` library to load the two Parquet-backed splits from the exact repository ID and pin the revision. Download **both complete splits**; never call `.select(range(...))`, `.take(...)`, sampling, or first-N truncation in the final data preparation workflow.

**Alternative:** Use `huggingface_hub.snapshot_download` with `repo_type="dataset"`, the pinned revision, and `allow_patterns=["data/*.parquet", "README.md"]`. This results in the same three data files (subject to Hugging Face mirroring details).

Proposed commands **after Codex has implemented the corresponding scripts**:

```bash
uv sync
uv run python scripts/download_data.py
uv run python scripts/prepare_data.py
uv run python scripts/build_index.py
uv run python scripts/validate_data.py
```

These commands are an **implementation contract**. The scripts do not exist as part of this documentation-only delivery.

Download script requirements:

1. Use a configurable data root (`data/` by default).
2. Check for cached/pinned files; avoid redownloading unchanged content.
3. Fail clearly for network, permission, insufficient disk space, and unexpected dataset revision.
4. Save a machine-readable manifest: source URL, repo ID, revision, file names, file sizes, SHA-256 checksums (where local files exist), UTC download timestamp, and code version when available.
5. Never load all raw text into a UI session on every app startup. Preparation/indexing is an explicit separate step.
6. Never commit the raw data, cache, generated indexes, or provider credentials to the source repository.

## 4. Raw-to-ready pipeline

```text
Hugging Face pinned snapshot
            |
            v
[Verify schema, locale, version, exact row counts]
            |
            v
[Read train + test, preserve raw labels separately]
            |
            +----> [query judgments] --> train/test Parquet (labels restricted to eval)
            |
            +----> [deduplicate products] --> catalog Parquet / SQLite products
                                              |
                                              v
                                      [clean indexable text]
                                              |
                                              v
                                         SQLite FTS5 index
```

### 4.1 Judgment records

Save the benchmark data in an explicit judgment table (suggested schema):

```text
judgments(example_id, query_id, query, product_locale, product_id,
          esci_label, split)
```

Preserve all 601,354 judgments, including those labeled `I` and those with missing optional product fields. Do not drop difficult examples just because a classifier struggles with them. If a row is genuinely malformed, report the exact count and reason; retain original raw data untouched.

### 4.2 Catalog records

Suggested product schema:

```text
products(product_locale, product_id, product_title,
         product_brand, product_color, product_bullet_point,
         product_description, index_text)
UNIQUE(product_locale, product_id)
```

- The dataset may repeat product metadata for different queries. Deduplicate by the composite key `(product_locale, product_id)`.
- Before deduplication, measure how often the same product has different or contradictory metadata. Prefer a deterministic policy such as choosing the record with the most non-empty fields; log conflicts rather than silently mixing contradictory values.
- If multiple records contain useful non-conflicting information, a deterministic merge is acceptable only if it does not introduce labels or query-specific knowledge.
- Product descriptions, bullets, and brand are optional; title must be present for a usable card. Count title-missing rows and document treatment.
- Build the **search catalog from unique products across both splits** if the evaluation is defined as transductive catalog search over known listing text. This is acceptable only when **all gold labels and query–product pair associations remain excluded from serving-time indexes and prompts**. Document the decision as part of evaluation setup.

### 4.3 Normalization

- Normalize Unicode and whitespace; decode/strip HTML tags from product descriptions and bullet points.
- Preserve negations, product numbers, specifications (`13x5.00-6`), brand words, and measurement units.
- Do not remove meaningful technical tokens simply because they look uncommon.
- Use a bounded representation for embeddings/LLM input: title + brand + bullets + selected description; record length limits and truncation order.
- Preserve the original title and optionally original cleaned description for the UI; do not erase product identity or provenance.
- Escape any displayed user/product content in the UI. Product text is untrusted content, not instructions.

### 4.4 Index

Implement **SQLite FTS5** over the unique catalog products. Store real product metadata in ordinary relational columns/table and use FTS as a search index. Ensure a stable join between FTS row and `(product_locale, product_id)`.

- Index title/brand/bullets/description; record field weights.
- Verify insert count equals prepared catalog count.
- Query returns stable ID, title, optional metadata, and **lexical BM25 rank**.
- Preserve a clean, repeatable rebuild process; reject partial or stale index state.
- Document why this is a *historical product catalog*, not a live commerce inventory.

No ANN service or dedicated vector DB is required for the selected MVP; semantic embeddings are computed for shortlisted products at search time with optional product-ID cache.

## 5. Data leakage and evaluation boundaries

**Critical rule: serving and evaluation must not share gold labels as features.**

- **Allowed:** a product's title and description from the full catalog, as known reference data.
- **Allowed:** developing UI/prompt wording on train queries and train judgments.
- **Allowed:** using test query text to run evaluation, with no access to expected labels during inference.
- **Forbidden:** including `esci_label`, `split`, `example_id`, known judged relevance, or query–product gold pairings in retrieval scores or LLM prompts.
- **Forbidden:** ranking products by precomputed gold ESCI labels or selecting candidate IDs from test qrels in the *live-search* evaluation.
- **Distinct evaluation:** when testing the LLM classifier alone, deliberately supply a fixed **judged candidate list** from test and compare LLM predictions to hidden gold. Label this a **conditional classification evaluation**, not end-to-end retrieval.
- Keep the published train/test split unchanged. Evaluate distinct queries grouped by `query_id`; do not create a random row-level split that mixes variants of the same query.

## 6. Minimal data audit (required)

Codex must generate a small, reproducible audit without excessive EDA. The audit should include:

| Audit item         | Acceptance condition                                               |
| ------------------ | ------------------------------------------------------------------ |
| Train/test rows    | Exactly 419,653 / 181,701 for pinned source                        |
| All rows           | Exactly 601,354; if not, fail and explain source/revision mismatch |
| Locale             | All `us`                                                           |
| Small subset       | All `small_version == 1`                                           |
| Gold label values  | Only E, S, C, I                                                    |
| IDs                | No null query/product IDs; record unusual duplicates               |
| Product fields     | Counts of missing/empty title, description, bullets, brand, color  |
| Label distribution | Counts by E/S/C/I and split                                        |
| Query distribution | Distinct query counts and train/test overlap check                 |
| Catalog            | Distinct `(locale, product_id)` count and metadata conflicts       |
| Index              | Indexable record count matches prepared product count              |

If the pinned source does not match published counts, stop and report the discrepancy before modifying the implementation to accommodate it. Count check is against the selected repack, not Amazon's entire multilingual dataset.

## 7. Proposed filesystem layout after implementation

```text
smart-product-search/
  AGENTS.md
  docs/
    PROJECT_IDEA.md
    DATA_SETUP.md
    IMPLEMENTATION_PLAN.md
  data/
    raw/                 # gitignored downloaded Parquet data
    processed/           # gitignored catalog/judgments/manifest
    index/               # gitignored SQLite FTS index
  .artifacts/
    data_audit.json      # gitignored run artifacts
    llm_cache/           # gitignored repeated predictions
    evaluations/         # gitignored detailed run artifacts
  scripts/               # created later by Codex
  src/                   # created later by Codex
  app/                   # created later by Codex
  tests/                 # created later by Codex
```

Keep the code layout simple. Do not create directories without an actual need or auto-commit generated data.

## 8. Reproducibility checklist

Record in each report/evaluation run:

- Pinned dataset revision and counts.
- Model name and exact installed model revision if available.
- OpenRouter model ID, endpoint, prompt version, and whether JSON Schema is supported.
- Reranking and retrieval configurations (`top_k`, field weights, normalization, fusion rule).
- Seed `2026` for any deterministic query sampling.
- Hardware / Python package versions, command line, and UTC date.
- Sample size, evaluation definition and limitations.
- Failed or skipped API calls, rather than treating them as successful predictions.

## 9. References

- Exact English Small repack and fields: https://huggingface.co/datasets/thebajajra/amazon-esci-english-small
- Repack dataset card (counts, sizes, formats): https://huggingface.co/datasets/thebajajra/amazon-esci-english-small/blob/main/README.md
- Original Amazon Science repository, paper citation and Apache 2.0 license: https://github.com/amazon-science/esci-data
- Dataset background: https://www.amazon.science/code-and-datasets/shopping-queries-dataset-a-large-scale-esci-benchmark-for-improving-product-search
