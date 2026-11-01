from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from smart_product_search import preparation, source
from smart_product_search.audit import build_audit
from smart_product_search.preparation import (
    CATALOG_COLUMNS,
    JUDGMENT_COLUMNS,
    RAW_SCHEMA,
    build_catalog,
    extract_judgments,
    normalize_listing_text,
    validate_prepared,
    validate_raw_split,
)
from smart_product_search.source import calculate_sha256
from smart_product_search.workflow import prepare_dataset, validate_dataset

NORMALIZATION_CASES = [
    (None, None),
    (" \n\t ", None),
    ('<p>No wheels</p><p>13x5.00-6 &amp; 3/4"</p>', 'No wheels 13x5.00-6 & 3/4"'),
    ("Cafe\u0301\u00a0  case\nwithout wheels", "Café case without wheels"),
    ("<script>ignore rules</script><style>hide</style><b>ACME</b> X-123", "ACME X-123"),
    (
        "&lt;script&gt;bad()&lt;/script&gt;No <b>noise</b> cancellation",
        "<script>bad()</script>No noise cancellation",
    ),
    ("&lt;USB-C&gt; adapter", "<USB-C> adapter"),
    ("Brand &amp;notebook", "Brand &notebook"),
    ("<div>first<br>second<li>third</li></div>", "first second third"),
    ("<b>13</b>x5.00-6, 5 < 10, ¼ inch", "13x5.00-6, 5 < 10, ¼ inch"),
    ("Size: 10 × 12 cm; not for iPhone 13", "Size: 10 × 12 cm; not for iPhone 13"),
    ("<!-- annotation -->Wireless <strong>headphones", "Wireless headphones"),
]


@pytest.fixture
def raw_records() -> pl.DataFrame:
    """Provide labeled pairs containing repeated products and incomplete metadata."""
    records = []
    for index, (product_id, title, brand, split, label) in enumerate(
        [
            ("P1", "<b>Wheel</b> 13x5.00-6", None, "train", "E"),
            ("P1", "Wheel 13x5.00-6", "ACME", "test", "S"),
            ("P2", None, None, "train", "I"),
            ("P3", "Case without wheels", None, "test", "C"),
        ]
    ):
        records.append(
            {
                "example_id": index,
                "query": f"query {index}",
                "query_id": index,
                "product_id": product_id,
                "product_locale": "us",
                "esci_label": label,
                "small_version": 1,
                "large_version": 1,
                "split": split,
                "product_title": title,
                "product_brand": brand,
                "product_color": None,
                "product_bullet_point": "not for X-123",
                "product_description": "<p>3/4 inch &amp; 10 cm</p>",
            }
        )
    return pl.DataFrame(records, schema=RAW_SCHEMA)


@pytest.mark.parametrize("value,expected", NORMALIZATION_CASES)
def test_normalize_listing_text(value: str | None, expected: str | None) -> None:
    """Strip markup and normalize whitespace without erasing meaningful specifications."""
    assert normalize_listing_text(value) == expected


@pytest.mark.parametrize(
    "text", ["Café case without wheels", '13x5.00-6 & 3/4"', "ACME X-123", "5 < 10, ¼ inch"]
)
def test_preserve_already_plain_specifications(text: str) -> None:
    """Leave plain normalized product specifications unchanged."""
    assert normalize_listing_text(text) == text


def test_preserve_complete_judgments_and_splits(raw_records: pl.DataFrame) -> None:
    """Keep every judgment and its original query text, label, identifiers, and split."""
    judgments = extract_judgments(raw_records)
    assert_frame_equal(judgments, raw_records.select(JUDGMENT_COLUMNS).sort("example_id"))
    assert judgments.height == 4
    assert judgments.filter(pl.col("split") == "train").height == 2
    assert judgments.filter(pl.col("esci_label") == "I").height == 1


def test_select_complete_metadata_and_preserve_product_identity(raw_records: pl.DataFrame) -> None:
    """Select a complete listing without mixing metadata or dropping title-missing products."""
    catalog = build_catalog(raw_records)
    assert catalog.columns == CATALOG_COLUMNS
    assert catalog["product_id"].to_list() == ["P1", "P2", "P3"]
    assert catalog.filter(pl.col("product_id") == "P1")["product_brand"].item() == "ACME"
    assert catalog.filter(pl.col("product_id") == "P2")["has_title"].item() is False
    assert "13x5.00-6" in catalog["index_text"][0]
    assert "not for X-123" in catalog["index_text"][0]
    assert "<p>" not in catalog["index_text"][0]


def test_catalog_is_independent_of_labels_queries_and_input_order(
    raw_records: pl.DataFrame,
) -> None:
    """Prevent label leakage and source-order-dependent catalog selection."""
    altered = raw_records.reverse().with_columns(
        pl.lit("SECRET_GOLD_LABEL").alias("esci_label"),
        pl.lit("SECRET_QUERY_TEXT").alias("query"),
        pl.lit(999).alias("query_id"),
        pl.lit(999).alias("example_id"),
        pl.lit("SECRET_SPLIT").alias("split"),
    )
    assert_frame_equal(build_catalog(raw_records), build_catalog(altered))
    assert "SECRET_" not in " ".join(build_catalog(altered)["index_text"].to_list())


def test_break_metadata_ties_lexically(raw_records: pl.DataFrame) -> None:
    """Use a label-independent lexical tie-break for conflicting equally complete listings."""
    tied = pl.concat(
        [
            raw_records.head(1).with_columns(pl.lit("Zeta").alias("product_title")),
            raw_records.head(1).with_columns(pl.lit("Alpha").alias("product_title")),
        ]
    )
    assert build_catalog(tied)["product_title"].item() == "Alpha"
    assert_frame_equal(build_catalog(tied), build_catalog(tied.reverse()))


@pytest.mark.parametrize(
    "column,value",
    [
        ("product_id", None),
        ("query_id", None),
        ("query", "  "),
        ("product_locale", "uk"),
        ("small_version", 0),
        ("esci_label", "X"),
        ("esci_label", None),
        ("split", "test"),
    ],
)
def test_reject_invalid_raw_values(raw_records: pl.DataFrame, column: str, value: object) -> None:
    """Reject malformed external rows rather than silently filtering them."""
    train = raw_records.filter(pl.col("split") == "train")
    train = train.with_columns(pl.lit(value, dtype=train.schema[column]).alias(column))
    with pytest.raises(ValueError):
        validate_raw_split(train, "train", expected_count=2)


def test_reject_missing_column_wrong_type_and_count(raw_records: pl.DataFrame) -> None:
    """Reject incomplete schemas, incorrect identifier types, and truncated splits."""
    train = raw_records.filter(pl.col("split") == "train")
    invalid_tables = [
        train.drop("product_title"),
        train.with_columns(pl.col("query_id").cast(pl.String)),
        train.head(1),
    ]
    for table in invalid_tables:
        with pytest.raises(ValueError):
            validate_raw_split(table, "train", expected_count=2)


def test_validate_prepared_rejects_leakage_and_lost_judgments(raw_records: pl.DataFrame) -> None:
    """Reject gold fields, duplicate catalog identities, or changed evaluation records."""
    splits = {split: raw_records.filter(pl.col("split") == split) for split in ("train", "test")}
    judgments = {split: extract_judgments(table) for split, table in splits.items()}
    catalog = build_catalog(raw_records)
    checks = validate_prepared(splits, catalog, judgments)
    assert checks["judgments_preserved"] is True
    assert checks["catalog_unique"] is True
    for invalid_catalog in (
        catalog.with_columns(pl.lit("E").alias("esci_label")),
        pl.concat([catalog, catalog.head(1)]),
        catalog.with_columns(pl.lit("bad").alias("index_text")),
    ):
        with pytest.raises(ValueError):
            validate_prepared(splits, invalid_catalog, judgments)
    judgments["train"] = judgments["train"].head(1)
    with pytest.raises(ValueError):
        validate_prepared(splits, catalog, judgments)


def test_reject_cross_split_query_overlap(raw_records: pl.DataFrame) -> None:
    """Prevent accidental query contamination between the published splits."""
    splits = {split: raw_records.filter(pl.col("split") == split) for split in ("train", "test")}
    splits["test"] = splits["test"].with_columns(pl.lit(0).alias("query_id"))
    judgments = {split: extract_judgments(table) for split, table in splits.items()}
    with pytest.raises(ValueError, match="overlap"):
        validate_prepared(splits, build_catalog(raw_records), judgments)


@pytest.fixture
def local_dataset(
    tmp_path: Path, raw_records: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Write a tiny complete local source using the real Parquet and manifest formats."""
    root = tmp_path / "data"
    raw_dir = root / "raw" / source.DATASET_REVISION
    (raw_dir / "data").mkdir(parents=True)
    train = raw_records.filter(pl.col("split") == "train")
    test = raw_records.filter(pl.col("split") == "test")
    tables = [train.head(1), train.tail(1), test]
    records = []
    expected_files = {}
    for name, table in zip(source.PARQUET_FILES, tables, strict=True):
        path = raw_dir / name
        table.write_parquet(path)
        size = path.stat().st_size
        checksum = source.calculate_sha256(path)
        records.append({"name": name, "size_bytes": size, "sha256": checksum})
        expected_files[name] = (size, checksum)
    card = raw_dir / "README.md"
    card.write_text("fixture source card", encoding="utf-8")
    records.append(
        {
            "name": "README.md",
            "size_bytes": card.stat().st_size,
            "sha256": source.calculate_sha256(card),
        }
    )
    source.write_json(
        root / "raw/manifest.json",
        {"repo_id": source.DATASET_ID, "revision": source.DATASET_REVISION, "files": records},
    )
    monkeypatch.setattr(source, "PARQUET_FILES", expected_files)
    monkeypatch.setattr(preparation, "EXPECTED_COUNTS", {"train": 2, "test": 2})
    return root


def test_audit_reports_observed_quality(raw_records: pl.DataFrame) -> None:
    """Measure real fixture missingness, classes, overlap, and metadata conflicts."""
    splits = {split: raw_records.filter(pl.col("split") == split) for split in ("train", "test")}
    audit = build_audit(splits, build_catalog(raw_records), {"revision": "fixture"})
    assert audit["total_rows"] == 4
    assert audit["catalog_products"] == 3
    assert audit["query_id_overlap"] == 0
    assert audit["product_overlap"] == 1
    assert audit["raw_missing_fields"]["product_title"]["null"] == 1
    assert audit["catalog_missing_fields"]["product_title"]["missing_or_blank"] == 1
    assert audit["metadata_conflicts"]["products_with_conflicting_nonempty_fields"] == 1
    assert audit["splits"]["train"]["label_counts"] == {"E": 1, "I": 1}


def test_full_preparation_is_idempotent_and_validated(local_dataset: Path, tmp_path: Path) -> None:
    """Run the real local pipeline twice with byte-identical outputs and preserved labels."""
    audit_dir = tmp_path / "audit"
    first = prepare_dataset(local_dataset, audit_dir)
    paths = [*sorted((local_dataset / "processed").glob("*")), *sorted(audit_dir.glob("*"))]
    hashes = {path.name: calculate_sha256(path) for path in paths}
    second = prepare_dataset(local_dataset, audit_dir)
    assert first == second
    assert hashes == {path.name: calculate_sha256(path) for path in paths}
    assert validate_dataset(local_dataset, audit_dir)["judgments_preserved"] is True
    assert pl.read_parquet(local_dataset / "processed/judgments_train.parquet").height == 2
    catalog_path = local_dataset / "processed/catalog.parquet"
    assert "esci_label" not in pl.read_parquet(catalog_path).columns
    pl.read_parquet(catalog_path).with_columns(pl.lit("E").alias("esci_label")).write_parquet(
        catalog_path
    )
    with pytest.raises(ValueError, match="checksum"):
        validate_dataset(local_dataset, audit_dir)


def test_stop_before_preparation_on_source_count_discrepancy(
    local_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stop instead of accepting a truncated split or publishing partial outputs."""
    monkeypatch.setattr(preparation, "EXPECTED_COUNTS", {"train": 419653, "test": 181701})
    with pytest.raises(ValueError, match="expected 419653 rows"):
        prepare_dataset(local_dataset, tmp_path / "audit")
    assert not (local_dataset / "processed").exists()
