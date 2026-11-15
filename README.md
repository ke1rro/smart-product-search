# Smart Product Search

Search an English product catalog, compare alternatives, and find accessories.
The application combines BM25 retrieval, optional local MiniLM reranking, and
explicit AI classification through OpenRouter. It uses historical Amazon ESCI
listings and is a research prototype, not an Amazon storefront.

## 1. Install prerequisites

You need Git and [uv](https://docs.astral.sh/uv/getting-started/installation/)
0.13.0 or newer. The commands below use a macOS or Linux terminal.
No GPU is required. Allow several GB of free disk space and available RAM.
Initial dependency, dataset, and model downloads require internet access.

Install uv if needed, then reopen your terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install its locked dependencies:

```bash
git clone https://github.com/ke1rro/NLP-HM.git
cd NLP-HM
uv python install 3.14
uv sync --locked
export PYTHONPATH="$PWD/src"
```

Run all following commands from the repository root. `uv` manages `.venv`, so
you do not need to activate it or install packages separately with pip.
The `PYTHONPATH` setting also avoids editable package import issues on macOS.

## 2. Configure AI analysis

BM25 and Semantic search work without an API key. To enable new AI analyses,
create a local configuration file:

```bash
cp -n .env.example .env
```

Open `.env` and set `OPENROUTER_API_KEY` to your key from
[OpenRouter](https://openrouter.ai/settings/keys). Keep an existing `.env` if you
already have one. Never commit this file.

The default model is `nvidia/nemotron-3.5-super-vl:free` in
[`config/search.toml`](config/search.toml). `OPENROUTER_MODEL` is an optional
override. Leave it commented out to use the TOML setting, rather than assigning
an empty value. Exported environment variables take precedence over `.env`.
Restart the application after changing credentials.

## 3. Prepare the catalog

Data and the search index are not included in Git. On the first installation,
run these commands in order:

```bash
uv run --locked python scripts/download_data.py
uv run --locked python scripts/prepare_data.py
uv run --locked python scripts/validate_data.py
uv run --locked python scripts/build_index.py
```

The download is about 398 MB. The scripts verify a pinned dataset revision and
checksums, preserve the official 419,653 train and 181,701 test judgments, and
produce a catalog of **482,105 unique products**. Relevance labels stay in separate
judgment files and are excluded from the search index and inference prompts.

Prepared files are saved in `data/processed/`. The complete search index is
`data/index/products.sqlite3`. Valid downloaded files are reused. Preparation and
index construction are deterministic and can be repeated, but do not need to run
before every application launch.

## 4. Start the application

```bash
uv run --locked streamlit run app/main.py
```

Open **http://localhost:8501**. Stop the server with **Ctrl+C**.
For subsequent launches from a new terminal:

```bash
cd /path/to/NLP-HM
export PYTHONPATH="$PWD/src"
uv run --locked streamlit run app/main.py
```

### Search workflow

1. Enter an English shopping query, such as `wireless noise cancelling headphones`.
2. Choose **BM25** or **Semantic**, then click **Search**.
   BM25 is the default and returns up to 100 products using word matches.
   Semantic reranks those candidates with `sentence-transformers/all-MiniLM-L6-v2`
   on CPU and returns up to 50. Its first use downloads the pinned model weights
   through Sentence Transformers. Later loads reuse the Hugging Face cache.
3. Browse ranked product cards. Use **Show more products** to expand the list.
4. Optionally click **Analyze with AI** to classify the current top 10 products.
   Validated predictions appear as **Exact Matches**, **Alternatives**,
   **Accessories & Complements**, and **Other Results**, with short explanations.

AI analysis runs only on an explicit click. Successful identical requests reuse
the persistent cache, including without API access. The analysis source is shown
as `api` or `cache`. Changing the query or mode clears stale analysis and requires
a new search. If classification fails, the retrieved products remain available.
Retrieval scores are optional under **Search details**.

## Troubleshooting

- **`No module named 'smart_product_search'`:** run `uv sync --locked`, then
  `export PYTHONPATH="$PWD/src"` from the repository root in the current terminal.
- **Product catalog unavailable:** complete all four data preparation commands
  above. Starting Streamlit does not download data or build an index.
- **AI authentication failed:** check the key in `.env`, any exported override,
  and restart Streamlit. Search itself does not require OpenRouter.
- **AI rate limit or provider unavailable:** wait before explicitly retrying.
  Free model availability depends on OpenRouter and its provider. Existing cached
  analyses remain usable. The application does not switch models automatically.

## Tests and development

The standard tests use small fixtures, controlled embeddings, and mocked HTTP.
They require neither the full dataset nor model downloads or live API access.

```bash
make check
uv run --locked pytest
uv run --locked pre-commit run --all-files
```

`make check` runs isort, Black, Flake8, ty, and Markdown formatting checks.
To install Git hooks for development, run `uv run --locked pre-commit install`.
Engineering conventions and reusable workflows are in [AGENTS.md](AGENTS.md)
and [.agents/skills/](.agents/skills/).

For the optional EDA notebook:

```bash
uv sync --locked --group notebooks
uv run --group notebooks jupyter lab notebooks/
```

Open `notebooks/esci_eda.ipynb` with the project's Python 3.14 kernel. Rerunning it
requires the prepared dataset. Exporting its plots also requires Chrome or
Chromium for Kaleido. Saved outputs can be viewed without rerunning the notebook.

## Documentation and limitations

- [Technical report and evaluation results](docs/REPORT.md)
- [Architecture diagram](docs/architecture.drawio.png)
  and [editable source](docs/architecture.drawio)
- [Dataset preparation and quality checks](docs/DATA_PREPARATION.md)
- [BM25 retrieval](docs/RETRIEVAL.md) and [semantic reranking](docs/SEMANTIC_RERANKING.md)
- [Classification configuration, evaluation, and reproduction commands](docs/CLASSIFICATION.md)
- [UI behavior and verification](docs/USER_INTERFACE.md)

The catalog contains historical, sometimes incomplete or noisy metadata.
No live prices, stock, ratings, or purchasing are provided. AI categories and
explanations are predictions that can be wrong. ESCI judgments are incomplete
over the full catalog, and the held-out classification evaluation does not
establish end-to-end search quality. Detailed measurements and limitations are
in the linked documents.

Data, indexes, model and response caches, `.env`, and local `.artifacts/` outputs
stay outside Git. Runtime settings, candidate counts, model identifiers, and
cache paths are centralized in `config/search.toml`.
