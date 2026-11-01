import hashlib
import json
import logging
from pathlib import Path

import polars as pl

from smart_product_search.audit import build_audit, write_audit
from smart_product_search.preparation import (
    build_catalog,
    extract_judgments,
    read_raw_splits,
    validate_prepared,
)
from smart_product_search.source import (
    DATASET_ID,
    DATASET_REVISION,
    calculate_sha256,
    verify_source,
    write_json,
)

OUTPUT_FILES = ["catalog.parquet", "judgments_train.parquet", "judgments_test.parquet"]
logger = logging.getLogger(__name__)


def preparation_fingerprint() -> str:
    """Fingerprint the exact preparation source independently of uncommitted Git state.

    Returns:
        SHA-256 over the source, normalization, audit, and workflow implementation files.
    """
    digest = hashlib.sha256()
    for name in ("source.py", "preparation.py", "audit.py", "workflow.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def write_parquet(path: Path, records: pl.DataFrame) -> None:
    """Atomically replace one deterministic Parquet artifact.

    Args:
        path: Destination whose parent already exists.
        records: Deterministically ordered records to persist without filtering.
    """
    temporary = path.with_suffix(".tmp")
    records.write_parquet(temporary, compression="zstd", row_group_size=128000)
    temporary.replace(path)


def prepare_dataset(data_dir: Path, audit_dir: Path) -> dict:
    """Prepare the complete label-free catalog and separate official judgment splits.

    Args:
        data_dir: Root containing the verified raw snapshot and generated processed/ files.
        audit_dir: Ignored directory for measured EDA and quality records.

    Returns:
        Deterministic prepared manifest with source and output checksums and code fingerprint.
    """
    raw_splits = read_raw_splits(data_dir)
    logger.info("Normalizing listings and selecting unique catalog products")
    catalog = build_catalog(pl.concat(list(raw_splits.values())))
    judgments = {split: extract_judgments(table) for split, table in raw_splits.items()}
    checks = validate_prepared(raw_splits, catalog, judgments)
    source_manifest = verify_source(data_dir)
    audit = build_audit(raw_splits, catalog, source_manifest)
    audit["checks"] = checks
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(processed_dir / "catalog.parquet", catalog)
    for split, table in judgments.items():
        write_parquet(processed_dir / f"judgments_{split}.parquet", table)
    manifest = {
        "repo_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "source_manifest_sha256": calculate_sha256(data_dir / "raw/manifest.json"),
        "preparation_fingerprint": preparation_fingerprint(),
        "polars_version": pl.__version__,
        "catalog_products": catalog.height,
        "split_rows": {split: table.height for split, table in judgments.items()},
        "files": [
            {
                "name": name,
                "size_bytes": (processed_dir / name).stat().st_size,
                "sha256": calculate_sha256(processed_dir / name),
            }
            for name in OUTPUT_FILES
        ],
    }
    write_json(processed_dir / "manifest.json", manifest)
    write_audit(audit, audit_dir)
    logger.info(
        "Prepared %s products and %s judgments",
        catalog.height,
        sum(table.height for table in judgments.values()),
    )
    return manifest


def validate_dataset(data_dir: Path, audit_dir: Path) -> dict:
    """Independently reload and validate source provenance and every prepared artifact.

    Args:
        data_dir: Root containing raw and processed manifests and Parquet outputs.
        audit_dir: Ignored directory in which to refresh the measured audit.

    Returns:
        Successful provenance, preservation, catalog integrity, and leakage checks.
    """
    source_manifest = verify_source(data_dir)
    processed_dir = data_dir / "processed"
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["repo_id"] != DATASET_ID or manifest["revision"] != DATASET_REVISION:
        raise ValueError("Prepared source identity differs from the approved pinned revision")
    if manifest["source_manifest_sha256"] != calculate_sha256(data_dir / "raw/manifest.json"):
        raise ValueError("Prepared source manifest checksum is stale")
    if manifest["preparation_fingerprint"] != preparation_fingerprint():
        raise ValueError("Preparation implementation changed; rerun prepare_data.py")
    if {record["name"] for record in manifest["files"]} != set(OUTPUT_FILES):
        raise ValueError("Prepared manifest must name the complete catalog and judgment outputs")
    for record in manifest["files"]:
        path = processed_dir / record["name"]
        if (
            calculate_sha256(path) != record["sha256"]
            or path.stat().st_size != record["size_bytes"]
        ):
            raise ValueError(f"Prepared file checksum mismatch: {record['name']}")
    raw_splits = read_raw_splits(data_dir)
    catalog = pl.read_parquet(processed_dir / "catalog.parquet")
    judgments = {
        split: pl.read_parquet(processed_dir / f"judgments_{split}.parquet") for split in raw_splits
    }
    checks = validate_prepared(raw_splits, catalog, judgments)
    if manifest["catalog_products"] != catalog.height or manifest["split_rows"] != {
        split: table.height for split, table in judgments.items()
    }:
        raise ValueError("Prepared manifest counts disagree with the actual artifact records")
    audit = build_audit(raw_splits, catalog, source_manifest)
    audit["checks"] = checks
    write_audit(audit, audit_dir)
    logger.info("All source, schema, split, preservation, catalog, and leakage checks passed")
    return checks
