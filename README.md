# NLP Assignment 1

Repository scaffold for the UCU NLP course assignment. The application idea,
datasets, and NLP stack have not been selected. No application is runnable yet.

## Environment setup

Install `uv` 0.13.0 or newer, then run from the repository root:

```bash
make setup
```

This installs managed Python 3.14, creates `.venv` from `uv.lock`, installs the
development tools, and enables pre-commit hooks. Run project commands with `uv run`.

The default environment includes Black, isort, Flake8, ty, pytest, pre-commit, and
the configured Markdown formatter plugins. There are no runtime dependencies yet.
Add application dependencies with `uv add` when the application requires them;
use `python-decouple` for environment settings as required by `AGENTS.md`.
Keep `pyproject.toml` and `uv.lock` together when sharing dependency changes.

## Repository layout

- `app/`: application entry points.
- `src/`: reusable NLP components.
- `scripts/`: data preparation and evaluation commands.
- `tests/`: automated code tests.
- `notebooks/`: exploration and analysis.
- `config/`: human-edited runtime TOML files.
- `.artifacts/`: ignored temporary experiment outputs.

`config/example.toml` and `.env.example` are placeholders, with no active application
settings or credentials required for setup.

## Verification

```bash
make check
make test
uv run pre-commit run --all-files
```

The scaffold has no Python source files or tests. Formatting, linting, and type
checks therefore have no Python code to inspect. pytest exits with code 5 until
tests are added. This is not evidence of application or NLP quality.

Pre-commit checks tracked files with `--all-files`. Before files are tracked, check
specific files using `uv run pre-commit run --files <paths>`.

## Optional notebooks

```bash
make notebooks
```

This installs the optional `notebooks` dependency group and starts JupyterLab in
`notebooks/`. Select the project's Python 3.14 kernel. Use
`uv sync --locked` to return to the default development environment.
