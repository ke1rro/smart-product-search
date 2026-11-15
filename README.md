# Smart Product Search

UCU NLP Assignment 1: an English product discovery prototype using a
historical Amazon ESCI catalog. The pipeline combines SQLite FTS5/BM25
retrieval, local MiniLM reranking, and OpenRouter ESCI classification to distinguish
exact matches, alternatives, and accessories.

**Current status: Iteration 5.** The complete 482,105-product catalog supports BM25
and optional local MiniLM reranking. BM25 remains the default retrieval mode.
Explicit classification uses `nvidia/nemotron-3.5-super-vl:free`, strict validation
and persistent cache replay. The Streamlit interface provides ranked product cards,
explicit AI analysis, and counted groups for predicted matches, alternatives,
complements, and other results. The fixed 199-pair held-out classification evaluation
is measured; labels remain unverified predictions. This is a research prototype,
not an Amazon storefront. Data, caches and indexes stay out of Git.

## Project specifications

- [Technical report: CRISP-DM, components, evaluation, and reproduction](docs/REPORT.md)
- [Product requirements and technical design](docs/PROJECT_IDEA.md)
- [Dataset source, setup, and data contracts](docs/DATA_SETUP.md)
- [Preparation process, measured statistics, and limitations](docs/DATA_PREPARATION.md)
- [Lexical retrieval, ranking, UI, and evaluation protocol](docs/RETRIEVAL.md)
- [Semantic reranking, cache, configuration, and measured validation](docs/SEMANTIC_RERANKING.md)
- [ESCI classification, prompt, API/cache contracts, and evaluation](docs/CLASSIFICATION.md)
- [Product interface, workflow, states, and UI verification](docs/USER_INTERFACE.md)
- [Application architecture diagram](docs/architecture.drawio.png) ([editable draw.io source](docs/architecture.drawio))
- [Implementation iterations and acceptance gates](docs/IMPLEMENTATION_PLAN.md)
- [Engineering and assignment conventions](AGENTS.md)

## Prerequisites and setup

Use managed Python 3.14 and `uv` 0.13.0 or newer. From the repository root:

```bash
make setup
```

This installs Python if needed, syncs `.venv` from `uv.lock`, and enables pre-commit
hooks. Initial dependency installation requires internet access. No API key is
required for data preparation. Run project commands through `uv run`.

Runtime dependencies are `python-decouple` for environment settings,
`huggingface-hub` for pinned downloads, `polars` for native Parquet processing,
`sentence-transformers` and `torch` for local CPU inference, `numpy` for vector
scoring, `httpx2` for reusable HTTP connections, `pydantic` for strict LLM output
validation, and `streamlit` for the product UI. SQLite FTS5 comes from Python's SQLite build.
The project is installed as an editable package so scripts can import its source.
Development tools are Black, isort, Flake8, ty, pytest, pre-commit, and
mdformat with its configured plugins. Add other dependencies only in the iteration
that uses them, updating `pyproject.toml` and `uv.lock` together.

## Configuration

`src/smart_product_search/config.py` exposes `load_settings()`, which reads
`config/search.toml` by default. It accepts an explicit TOML path for checks or
custom configuration. Missing fields use the defaults below; unknown keys and
invalid values are rejected. Loading settings does not create storage directories,
load models, or contact an API.

| Key                  | Default                                    | Meaning                                                        |
| -------------------- | ------------------------------------------ | -------------------------------------------------------------- |
| `data_dir`           | `"data"`                                   | Root for raw and prepared data                                 |
| `index_path`         | `"data/index/products.sqlite3"`            | Complete SQLite product index                                  |
| `retrieval_k`        | `100`                                      | Maximum BM25 candidates                                        |
| `rerank_k`           | `50`                                       | Maximum semantic results returned                              |
| `classify_k`         | `10`                                       | Maximum candidates from the chosen mode to classify, at most10 |
| `embedding_model`    | `"sentence-transformers/all-MiniLM-L6-v2"` | Local pretrained embedding model                               |
| `llm_model`          | `"nvidia/nemotron-3.5-super-vl:free"`      | Explicit configurable OpenRouter model                         |
| `openrouter_api_key` | `""`                                       | Environment-only secret, never a TOML field                    |

Candidate counts must be positive integers with
`classify_k <= rerank_k <= retrieval_k`. Paths and `embedding_model` must not be
blank. Relative storage paths resolve from the repository root regardless of the
current working directory; absolute paths are preserved. `index_path` is independent
of `data_dir`, so adjust both when relocating storage.

The [semantic configuration reference](docs/SEMANTIC_RERANKING.md#central-model-configuration)
documents the pinned model revision, CPU device, batch size, text/sequence limits,
thread count, and bounded product-vector cache. No API key is needed for search.
The [classification configuration reference](docs/CLASSIFICATION.md#run-and-configure)
documents timeout, output tokens, temperature, bounded retries, text budget and
persistent response cache settings.

For classification, copy `.env.example` to `.env` and fill
`OPENROUTER_API_KEY`. Optionally set `OPENROUTER_MODEL` to override `llm_model`.
Environment variables take precedence over `.env`, and either source's model value
takes precedence over TOML. Omit `OPENROUTER_MODEL` to use TOML; an explicitly empty
value leaves the model unselected. Settings are read through `decouple`; the secret
is excluded from the settings representation. Never log or serialize credentials.
The requested free model is the explicit TOML default; no model fallback is configured.

## Development workflow and skills

Implement one authorized iteration at a time, verify its gate, and stop. Read the
relevant specification and skill before acting. Do not stage, commit, push, create
branches, download large resources, or make API calls merely because a skill exists.

- [Dataset audit skill](.agents/skills/dataset-audit/SKILL.md): source integrity,
  split preservation, missingness, deduplication, and leakage checks when data work
  is authorized. Missing data is a prerequisite, not an audit pass.
- [Component evaluation skill](.agents/skills/component-evaluation/SKILL.md): define
  evidence before implementation and separate code tests from NLP quality metrics.

Each skill specifies required inputs, steps, output artifacts, and verification.
Keep detailed run records in ignored `.artifacts/`; report measured results and
limitations. Configuration and data pipeline tests do not establish search or
model quality.

```bash
uv run isort --check-only .
uv run black --check .
uv run flake8 src/ scripts/ tests/ app/
uv run ty check
uv run pytest
uv run pre-commit run --all-files
```

`make check` runs formatting, linting, type checking, and Markdown checks.
`make test` runs pytest. Pre-commit's `--all-files` checks tracked files; use
`uv run pre-commit run --files <paths>` to check new files before tracking them.
CI also runs the focused offline tests with tiny fixtures, without downloading data.

## Prepare and validate the full dataset

Run these implemented commands from the repository root:

```bash
uv sync --locked
uv run python scripts/download_data.py
uv run python scripts/prepare_data.py
uv run python scripts/validate_data.py
```

The download uses the exact pinned `thebajajra/amazon-esci-english-small` snapshot
and verifies all three complete Parquet files against recorded byte sizes and
SHA-256 checksums. Preparation requires exactly 419,653 train and 181,701 test rows;
an unexpected count stops the pipeline. No rows are sampled or relevance-filtered.
The first download requires internet and about 398 MB of transfer; allow several
GB of disk and a few GB of available RAM. A valid cached download is verified and
reused offline. Prepared files can be rebuilt deterministically without API access.

Outputs are `data/processed/catalog.parquet`, `judgments_train.parquet`,
`judgments_test.parquet`, and `manifest.json`. Original Parquet and the source card
remain under `data/raw/<revision>/`; `data/raw/manifest.json` records provenance.
Gold labels and query associations belong only to the judgment files. The catalog
contains listing metadata, including products represented in either official split.

Each script accepts `--config PATH` and `--data-dir PATH`. Preparation and validation
also accept `--audit-dir PATH`, defaulting to `.artifacts/dataset-audit/`. The audit
JSON and Markdown contain measured distributions, missingness, conflicts, overlap,
and validation checks. See the preparation document for contracts and limitations.

If validation fails, inspect the named file or contract. Preserve original raw
files; do not weaken counts or replace the source. After a preparation code change,
rerun preparation because the validator checks its recorded source fingerprint.

On macOS, Python ignores `.pth` files with the filesystem `hidden` flag. If a local
environment sets this flag and scripts cannot import `smart_product_search` after
`uv sync --locked`, clear it on the editable package's file immediately before the
command: `chflags nohidden .venv/lib/python3.14/site-packages/__editable__.nlp_hm-0.1.0.pth`.
This is a local environment repair, not a dataset download failure.

## Build and search the full catalog

After preparing the complete dataset:

```bash
uv run python scripts/build_index.py
uv run streamlit run app/main.py
```

The builder verifies catalog provenance and indexes all 482,105 unique products.
It rebuilds atomically and records its counts/checksum/settings in
`.artifacts/iteration-2/index-build.json`. The page displays real
products from `data/index/products.sqlite3`; it does not parse dataset files on
page load. An existing SQLite build with FTS5 support is required.

Semantic mode retrieves the original top 100 BM25 candidates, ranks them by
normalized MiniLM cosine similarity, and returns up to 50. The page displays the
first 20 results, with **Show more products** revealing the remaining saved results.
Scores are optional under **Search details**. Select **BM25** to preserve lexical ordering without
initializing the model. The first semantic search downloads the pinned weights
through Sentence Transformers; subsequent loads reuse the Hugging Face cache.

The unified API is `search_products(query, mode="bm25")` from
`smart_product_search.search`, with `mode="semantic"` for optional reranking. Original
metadata and BM25 scores are preserved; semantic scores are separate fields.

For code, call `retrieve_products(query, k=100)` from
`smart_product_search.retrieval`, optionally supplying `index_path=Path(...)`.
Queries are treated as text, not FTS/SQL commands. Smaller BM25 scores rank better;
product identity breaks score ties. Titles receive greater weight than descriptions.
See [retrieval contracts and limitations](docs/RETRIEVAL.md).

```bash
uv run python scripts/evaluate_retrieval.py
```

This measures one fixed BM25 system on 200 systematically selected official test
queries against the full catalog, with raw IDs/ranks/timings saved in
`.artifacts/evaluations/retrieval/`. It reports judged-positive coverage at K=10/100
for E, S, and E-or-S, plus actual API latency. These are sampled known-judgment
coverage measurements, not guaranteed corpus-wide recall. No parameter tuning,
model comparison, or ablation is performed.

```bash
uv run python scripts/evaluate_semantic.py
uv run python scripts/smoke_semantic.py
```

The paired semantic evaluation requires the unchanged Iteration 2 selection and
baseline artifacts. It reuses those 200 queries and records coverage at K=10/50,
conditional judged-pool nDCG@10, CPU latency, and cache counts under
`.artifacts/evaluations/semantic/`. The real-model smoke command checks three
queries, original product identities, normalized vectors, repeated ordering, and
cache reuse. Ordinary unit tests use controlled embeddings and need no downloads.
See [the protocol, measured results, and limitations](docs/SEMANTIC_RERANKING.md).

## Explicit ESCI classification

Search first, then click **Analyze with AI**. One JSON-mode OpenRouter
request predicts E/S/C/I with a short metadata-grounded reason for every supplied
ID. Predictions are unverified. The UI groups them into **Exact Matches (E)**,
**Alternatives (S)**, **Accessories & Complements (C)**, and a collapsed
**Other Results (I)** section. Each group shows its count and preserves retrieval
order. Remaining products stay separately accessible as not analyzed, without an
invented category. Ordinary reruns do not request inference. Identical successful
requests replay the persistent cache, including offline without a key. Failure
shows an explicit unavailable warning while retaining search results. Editing the
query or mode clears stale analysis and requires **Search** before analyzing again.
See [the UI workflow and limitations](docs/USER_INTERFACE.md).

```bash
uv run python scripts/evaluate_classification.py --allow-api --output-dir .artifacts/evaluations/classification/nemotron
uv run python scripts/evaluate_classification.py --output-dir .artifacts/evaluations/classification/nemotron/replay
```

The first command explicitly permits at most 20 classification POSTs, with no
evaluation retries. The second only replays valid cache entries. The fixed subset
has 199 judged pairs across 20 official test queries; it evaluates classification
separately from live retrieval. All199 pairs were classified: accuracy 53.27%,
macro-F1 0.4653, with 100% valid responses and coverage. The run reused one
cached batch and made 19 successful API calls. Median/p95 successful new request
latency was 2.469/4.255 seconds. Complement has only five examples, making its
per-class result unstable. These measurements do not establish full-catalog or
end-to-end search relevance. Details, class metrics, baseline and exact commands
are in [the classification document](docs/CLASSIFICATION.md) and
`.artifacts/iteration-4/acceptance.md`. No automatic model substitution is configured.

## Repository layout

- `app/main.py`: product search, explicit analysis, and grouped result cards.
- `src/smart_product_search/`: settings, data, retrieval, reranking, classification and evaluation.
- `scripts/`: data pipeline, index construction, evaluation, and real-model smoke commands.
- `tests/`: offline configuration/data tests plus FTS ranking, integrity, safety, metrics, and UI tests.
- `notebooks/`: optional exploration and analysis.
- `config/`: human-edited runtime TOML.
- `docs/`: supplied requirements, data contract, and iteration plan.
- `.agents/skills/`: reusable development procedures.
- `.artifacts/`: ignored local run records, evaluation outputs, and caches.

Raw/prepared data, indexes, embeddings, caches, evaluation outputs, `.venv`, and
`.env` are excluded from Git. Store generated outputs in these ignored locations.

## Optional notebooks

```bash
make notebooks
```

This installs the optional `notebooks` dependency group and starts JupyterLab in
`notebooks/`. Select the project's Python 3.14 kernel and follow
[notebook conventions](notebooks/README.md). Use `uv sync --locked` to return to the
lean default environment. Iteration 0 requires no notebook execution.
