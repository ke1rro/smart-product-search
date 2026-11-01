# Smart Product Search

UCU NLP Assignment 1: an English product discovery prototype planned around a
historical Amazon ESCI catalog. The intended pipeline combines SQLite FTS5/BM25
retrieval, local MiniLM reranking, and OpenRouter ESCI classification to distinguish
exact matches, alternatives, and accessories.

**Current status: Iteration 2.** The full catalog is searchable using SQLite FTS5
and BM25, with a minimal Streamlit result list. Data preparation and validation
remain available. Semantic reranking, LLM calls, ESCI predictions, and the final
product UI are not implemented. Data and indexes remain local and excluded from Git.

## Project specifications

- [Product requirements and technical design](docs/PROJECT_IDEA.md)
- [Dataset source, setup, and data contracts](docs/DATA_SETUP.md)
- [Preparation process, measured statistics, and limitations](docs/DATA_PREPARATION.md)
- [Lexical retrieval, ranking, UI, and evaluation protocol](docs/RETRIEVAL.md)
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
and `streamlit` for the minimal UI. SQLite FTS5 comes from Python's SQLite build.
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

| Key                  | Default                                    | Meaning                                                                |
| -------------------- | ------------------------------------------ | ---------------------------------------------------------------------- |
| `data_dir`           | `"data"`                                   | Root for raw and prepared data                                         |
| `index_path`         | `"data/index/products.sqlite3"`            | Planned SQLite index file                                              |
| `retrieval_k`        | `100`                                      | Maximum BM25 candidates                                                |
| `rerank_k`           | `50`                                       | Maximum lexical candidates to rerank                                   |
| `classify_k`         | `10`                                       | Maximum reranked candidates to classify                                |
| `embedding_model`    | `"sentence-transformers/all-MiniLM-L6-v2"` | Planned local embedding model                                          |
| `llm_model`          | `""`                                       | Unselected until a compatible free endpoint is verified in Iteration 4 |
| `openrouter_api_key` | `""`                                       | Environment-only secret, never a TOML field                            |

Candidate counts must be positive integers with
`classify_k <= rerank_k <= retrieval_k`. Paths and `embedding_model` must not be
blank. Relative storage paths resolve from the repository root regardless of the
current working directory; absolute paths are preserved. `index_path` is independent
of `data_dir`, so adjust both when relocating storage.

When API configuration becomes needed, copy `.env.example` to `.env` and fill
`OPENROUTER_API_KEY`. Optionally set `OPENROUTER_MODEL` to override `llm_model`.
Environment variables take precedence over `.env`, and either source's model value
takes precedence over TOML. Omit `OPENROUTER_MODEL` to use TOML; an explicitly empty
value leaves the model unselected. Settings are read through `decouple`; the secret
is excluded from the settings representation. Never log or serialize credentials.
No model is selected automatically, and no paid fallback is configured.

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
`.artifacts/iteration-2/index-build.json`. The page displays real, unclassified
lexical candidates from `data/index/products.sqlite3`; it does not parse dataset
files on page load. An existing SQLite build with FTS5 support is required.

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

Semantic reranking, remote model calls, ESCI grouping, and the final shopping UX
remain unavailable until their authorized iterations. No API key is required for
lexical retrieval or its evaluation.

## Repository layout

- `app/main.py`: minimal local lexical search UI.
- `src/smart_product_search/`: settings, preparation/audit, lexical retrieval and evaluation.
- `scripts/`: data pipeline, complete index construction, and retrieval evaluation commands.
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
