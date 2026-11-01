import hashlib
import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import snapshot_download

DATASET_ID = "thebajajra/amazon-esci-english-small"
DATASET_REVISION = "e7bb589a050171ed4b97891bd517507f702af0c2"
SOURCE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/tree/{DATASET_REVISION}"
PARQUET_FILES = {
    "data/train-00000-of-00002.parquet": (
        140273290,
        "495392d58dc9ea4212614149e007706fe8624140f5c40087645b9c27621042ce",
    ),
    "data/train-00001-of-00002.parquet": (
        137095927,
        "f3f09b40b25d7f7a84e60b0529b4c090f0f4c1dc9d954b53ed36df796d2ed8d9",
    ),
    "data/test-00000-of-00001.parquet": (
        120979140,
        "64349033225f9769f9955d5c47eec96c700350a9ad00f00ebdef20318fdc6e73",
    ),
}
logger = logging.getLogger(__name__)


def calculate_sha256(path: Path) -> str:
    """Calculate a streaming SHA-256 checksum.

    Args:
        path: File to hash without loading it entirely into memory.

    Returns:
        Hexadecimal checksum of the file bytes.
    """
    with path.open("rb") as source_file:
        return hashlib.file_digest(source_file, "sha256").hexdigest()


def write_json(path: Path, record: dict) -> None:
    """Atomically write a deterministic JSON record.

    Args:
        path: Destination file whose parent already exists.
        record: JSON-serializable mapping of metadata and measured results.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_source(data_dir: Path) -> dict:
    """Verify pinned identity, completeness, sizes, and checksums before reading data.

    Args:
        data_dir: Root containing raw/manifest.json and the pinned snapshot.

    Returns:
        Source manifest with repository, revision, provenance, and file records.
    """
    manifest = json.loads((data_dir / "raw/manifest.json").read_text(encoding="utf-8"))
    if manifest["repo_id"] != DATASET_ID or manifest["revision"] != DATASET_REVISION:
        raise ValueError("Unexpected dataset source or revision in raw manifest")
    expected_names = {*PARQUET_FILES, "README.md"}
    if {record["name"] for record in manifest["files"]} != expected_names or len(
        manifest["files"]
    ) != len(expected_names):
        raise ValueError("Raw manifest must contain exactly the pinned source files")
    for record in manifest["files"]:
        path = data_dir / "raw" / DATASET_REVISION / record["name"]
        size, checksum = PARQUET_FILES.get(record["name"], (record["size_bytes"], record["sha256"]))
        if path.stat().st_size != size or size != record["size_bytes"]:
            raise ValueError(f"Source file size mismatch: {record['name']}")
        if calculate_sha256(path) != checksum or checksum != record["sha256"]:
            raise ValueError(f"Source file checksum mismatch: {record['name']}")
    return manifest


def download_dataset(data_dir: Path) -> dict:
    """Download the complete pinned snapshot or verify an existing cached download.

    Args:
        data_dir: Configurable local data root; source files stay under raw/.

    Returns:
        Verified manifest with UTC download time, source revision, checksums, and code version.
    """
    if (data_dir / "raw/manifest.json").exists():
        manifest = verify_source(data_dir)
        logger.info("Reused verified pinned source without network access")
        return manifest
    raw_dir = data_dir / "raw" / DATASET_REVISION
    raw_dir.mkdir(parents=True, exist_ok=True)
    required_bytes = sum(size for size, _ in PARQUET_FILES.values()) * 3 + 1_000_000_000
    if shutil.disk_usage(raw_dir).free < required_bytes:
        raise OSError(f"Insufficient disk space: require {required_bytes} free bytes at {data_dir}")
    logger.info("Downloading complete %s at %s", DATASET_ID, DATASET_REVISION)
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=[*PARQUET_FILES, "README.md"],
        local_dir=raw_dir,
        token=False,
        max_workers=3,
    )
    records = []
    for name in sorted([*PARQUET_FILES, "README.md"]):
        path = raw_dir / name
        size, checksum = path.stat().st_size, calculate_sha256(path)
        if name in PARQUET_FILES and (size, checksum) != PARQUET_FILES[name]:
            raise ValueError(f"Downloaded source size or checksum mismatch: {name}")
        records.append({"name": name, "size_bytes": size, "sha256": checksum})
    code_version = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    manifest = {
        "repo_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "source_url": SOURCE_URL,
        "original_source_url": "https://github.com/amazon-science/esci-data",
        "license": "Apache-2.0",
        "license_url": "https://github.com/amazon-science/esci-data/blob/main/LICENSE",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "code_version": code_version.stdout.strip() if code_version.returncode == 0 else None,
        "files": records,
    }
    write_json(data_dir / "raw/manifest.json", manifest)
    logger.info("Verified %s source files and recorded their checksums", len(records))
    return manifest
