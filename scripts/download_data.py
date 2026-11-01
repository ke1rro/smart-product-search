import argparse
import logging
import sys
from pathlib import Path

from smart_product_search.config import PROJECT_ROOT, load_settings
from smart_product_search.source import download_dataset


def main() -> None:
    """Download and verify the complete pinned ESCI English Small snapshot."""
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
    arguments = parser.parse_args()
    download_dataset(arguments.data_dir or load_settings(arguments.config).data_dir)


if __name__ == "__main__":
    main()
