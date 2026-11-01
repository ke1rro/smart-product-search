import argparse
import logging
import sys
from pathlib import Path

from smart_product_search.config import PROJECT_ROOT, load_settings
from smart_product_search.semantic_evaluation import evaluate_semantic


def main() -> None:
    """Validate MiniLM reranking against BM25 on the unchanged Iteration 2 test queries."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/search.toml")
    parser.add_argument(
        "--fixed-queries",
        type=Path,
        default=PROJECT_ROOT / ".artifacts/evaluations/retrieval/selected_queries.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / ".artifacts/evaluations/semantic"
    )
    arguments = parser.parse_args()
    evaluate_semantic(
        load_settings(arguments.config), arguments.fixed_queries, arguments.output_dir
    )


if __name__ == "__main__":
    main()
