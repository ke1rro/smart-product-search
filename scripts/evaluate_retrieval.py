import argparse
import logging
import sys
from pathlib import Path

from smart_product_search.config import PROJECT_ROOT, load_settings
from smart_product_search.retrieval_evaluation import evaluate_retrieval


def main() -> None:
    """Measure fixed lexical retrieval coverage and latency on official ESCI test queries."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/search.toml")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / ".artifacts/evaluations/retrieval"
    )
    arguments = parser.parse_args()
    settings = load_settings(arguments.config)
    evaluate_retrieval(
        arguments.data_dir or settings.data_dir,
        arguments.index_path or settings.index_path,
        arguments.output_dir,
        arguments.sample_size,
    )


if __name__ == "__main__":
    main()
