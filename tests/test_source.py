import hashlib
import json
from pathlib import Path

import pytest

from smart_product_search import source


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a tiny local snapshot at the network boundary."""
    expected_files = {}
    for name in source.PARQUET_FILES:
        content = f"fixture: {name}".encode()
        expected_files[name] = (len(content), hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(source, "PARQUET_FILES", expected_files)

    def download(**options: object) -> str:
        destination = options["local_dir"]
        assert isinstance(destination, Path)
        assert options["revision"] == source.DATASET_REVISION
        assert options["repo_id"] == source.DATASET_ID
        for name in expected_files:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture: {name}", encoding="utf-8")
        (destination / "README.md").write_text("license: apache-2.0\n", encoding="utf-8")
        return str(destination)

    monkeypatch.setattr(source, "snapshot_download", download)
    return tmp_path


def test_download_records_and_reuses_verified_source(snapshot: Path) -> None:
    """Preserve a complete verified manifest on repeated downloads."""
    manifest = source.download_dataset(snapshot)
    manifest_path = snapshot / "raw/manifest.json"
    original_bytes = manifest_path.read_bytes()

    assert manifest["revision"] == source.DATASET_REVISION
    assert len(manifest["files"]) == 4
    assert source.download_dataset(snapshot) == manifest
    assert manifest_path.read_bytes() == original_bytes


def test_reject_corrupted_cached_file(snapshot: Path) -> None:
    """Reject modified cached bytes before they can enter preparation."""
    source.download_dataset(snapshot)
    path = snapshot / "raw" / source.DATASET_REVISION / next(iter(source.PARQUET_FILES))
    path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="checksum|size"):
        source.verify_source(snapshot)


@pytest.mark.parametrize("field,value", [("repo_id", "unapproved/source"), ("revision", "main")])
def test_reject_source_substitution(snapshot: Path, field: str, value: str) -> None:
    """Reject manifests pointing to a different dataset or a floating revision."""
    source.download_dataset(snapshot)
    path = snapshot / "raw/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source|revision"):
        source.verify_source(snapshot)
