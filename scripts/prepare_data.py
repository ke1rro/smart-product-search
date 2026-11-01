import argparse
import logging
import sys
from pathlib import Path

from smart_product_search.config import PROJECT_ROOT, load_settings
from smart_product_search.workflow import prepare_dataset


def main() -> None:
    """Prepare the complete normalized catalog and separate gold judgment splits."""
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
    parser.add_argument("--audit-dir", type=Path, default=PROJECT_ROOT / ".artifacts/dataset-audit")
    arguments = parser.parse_args()
    prepare_dataset(
        arguments.data_dir or load_settings(arguments.config).data_dir, arguments.audit_dir
    )


if __name__ == "__main__":
    main()
