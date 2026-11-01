import argparse
import logging
import sys
from pathlib import Path

from smart_product_search.config import PROJECT_ROOT, load_settings
from smart_product_search.retrieval import build_prepared_index
from smart_product_search.source import write_json


def main() -> None:
    """Build a complete label-free SQLite FTS5 index from the prepared catalog."""
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
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / ".artifacts/iteration-2/index-build.json"
    )
    arguments = parser.parse_args()
    settings = load_settings(arguments.config)
    metadata = build_prepared_index(
        arguments.data_dir or settings.data_dir, arguments.index_path or settings.index_path
    )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(arguments.report, metadata)


if __name__ == "__main__":
    main()
