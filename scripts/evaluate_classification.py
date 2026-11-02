import argparse
import logging
import sys
from pathlib import Path

from smart_product_search.classification_evaluation import evaluate_classification
from smart_product_search.config import PROJECT_ROOT, load_settings


def main() -> None:
    """Evaluate fixed judged test pairs; allow remote inference only with --allow-api."""
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
        "--output-dir", type=Path, default=PROJECT_ROOT / ".artifacts/evaluations/classification"
    )
    parser.add_argument(
        "--allow-api",
        action="store_true",
        help="Allow at most20 classification POSTs; otherwise replay cache only",
    )
    arguments = parser.parse_args()
    summary = evaluate_classification(
        load_settings(arguments.config), arguments.output_dir, arguments.allow_api
    )
    logging.getLogger(__name__).info(
        "Validated %s/%s batches; POSTs=%s; results at %s",
        summary["classified_queries"],
        summary["query_count"],
        summary["api_requests"],
        arguments.output_dir,
    )


if __name__ == "__main__":
    main()
