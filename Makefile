.PHONY: setup check test notebooks

setup:
	uv python install 3.14
	uv sync --locked
	uv run pre-commit install

check:
	uv run isort --check-only .
	uv run black --check .
	uv run flake8 src/ scripts/ tests/ app/
	uv run ty check
	uv run mdformat --check README.md notebooks/README.md docs/ .agents/skills/

test:
	uv run pytest

notebooks:
	uv sync --locked --group notebooks
	uv run --group notebooks jupyter lab notebooks/
