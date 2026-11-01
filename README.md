# Smart Product Search

UCU NLP Assignment 1: an English product discovery prototype planned around a
historical Amazon ESCI catalog. The intended pipeline combines SQLite FTS5/BM25
retrieval, local MiniLM reranking, and OpenRouter ESCI classification to distinguish
exact matches, alternatives, and accessories.

**Current status: Iteration 0 only.** Configuration, agent skills, and development
tooling are available. Data preparation, search, model inference, and the Streamlit
application are not implemented. No dataset or model is bundled or downloaded.

## Project specifications

- [Product requirements and technical design](docs/PROJECT_IDEA.md)
- [Dataset source, setup, and data contracts](docs/DATA_SETUP.md)
- [Implementation iterations and acceptance gates](docs/IMPLEMENTATION_PLAN.md)
- [Engineering and assignment conventions](AGENTS.md)

## Prerequisites and setup

Use managed Python 3.14 and `uv` 0.13.0 or newer. From the repository root:

```bash
make setup
```

This installs Python if needed, syncs `.venv` from `uv.lock`, and enables pre-commit
hooks. Initial dependency installation requires internet access. No API key is
required for Iteration 0. Run project commands through `uv run`.

The only runtime dependency at this stage is `python-decouple` for environment
settings. Development tools are Black, isort, Flake8, ty, pytest, pre-commit, and
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
limitations. Iteration 0 tests configuration only and makes no NLP quality claims.

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
CI also runs the configuration tests.

## Planned run commands, unavailable until implemented

Do not run these commands yet. Their scripts and application do not exist.

| Iteration | Planned command                          | Purpose                                        |
| --------- | ---------------------------------------- | ---------------------------------------------- |
| 1         | `uv run python scripts/download_data.py` | Download the full pinned English Small dataset |
| 1         | `uv run python scripts/prepare_data.py`  | Separate catalog and judgments                 |
| 1         | `uv run python scripts/validate_data.py` | Validate schema, counts, and boundaries        |
| 2         | `uv run python scripts/build_index.py`   | Build the catalog FTS5 index                   |
| 2         | `uv run streamlit run app/main.py`       | Start the planned local search UI              |

Data and model downloads require internet and disk space. The later OpenRouter
classification stage requires credentials and a verified free model endpoint.
The full application is not runnable at this gate.

## Repository layout

- `app/`: future application entry points.
- `src/smart_product_search/`: settings now; NLP components in later iterations.
- `scripts/`: future data preparation and evaluation commands.
- `tests/`: configuration parsing, defaults, validation, and secret handling tests.
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
