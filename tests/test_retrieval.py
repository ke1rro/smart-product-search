from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from smart_product_search.config import PROJECT_ROOT, load_settings
from smart_product_search.preparation import CATALOG_COLUMNS, PRODUCT_FIELDS
from smart_product_search.retrieval import build_index, retrieve_products
from smart_product_search.source import calculate_sha256


def write_catalog(path: Path, products: list[tuple[str, str, str | None]]) -> None:
    """Write a tiny catalog with the same serving schema as prepared complete data.

    Args:
        path: Destination Parquet file.
        products: Product ID, title, and optional description records.
    """
    rows = [
        {
            "product_locale": "us",
            "product_id": product_id,
            "product_title": title,
            "product_brand": None,
            "product_color": None,
            "product_bullet_point": None,
            "product_description": description,
            "product_title_original": title,
            "has_title": True,
            "index_text": title + (" " + description if description else ""),
        }
        for product_id, title, description in products
    ]
    schema = {name: pl.Boolean if name == "has_title" else pl.String for name in CATALOG_COLUMNS}
    pl.DataFrame(rows, schema=schema).write_parquet(path)


@pytest.fixture
def catalog_path(tmp_path: Path) -> Path:
    """Provide distinct title/description matches, ties, and absent descriptions."""
    path = tmp_path / "catalog.parquet"
    write_catalog(
        path,
        [
            ("P_TITLE", "Quantum headphones", "Comfortable ear pads"),
            ("P_DESC", "Audio accessory", "Quantum headphones accessory"),
            ("P_NULL", "UniqueNullDescription headset", None),
            ("P_A", "Twin headset", None),
            ("P_B", "Twin headset", None),
        ],
    )
    return path


@pytest.fixture
def index_path(catalog_path: Path, tmp_path: Path) -> Path:
    """Build a real tiny SQLite FTS5 database for retrieval tests."""
    path = tmp_path / "search.sqlite3"
    build_index(catalog_path, path)
    return path


def test_index_covers_complete_catalog_and_rebuilds(catalog_path: Path, index_path: Path) -> None:
    """Keep exactly one searchable row per input product across repeated rebuilds."""
    original_hash = calculate_sha256(index_path)
    manifest = build_index(catalog_path, index_path)
    assert manifest["product_count"] == 5
    assert manifest["indexed_count"] == 5
    assert calculate_sha256(index_path) == original_hash
    records = retrieve_products("headset headphones accessory", index_path=index_path)
    assert {record.product_id for record in records} == {
        "P_TITLE",
        "P_DESC",
        "P_NULL",
        "P_A",
        "P_B",
    }


def test_title_matches_outrank_description_matches(index_path: Path) -> None:
    """Give an otherwise comparable title match more BM25 weight than a description match."""
    results = retrieve_products("quantum", index_path=index_path)
    assert [record.product_id for record in results] == ["P_TITLE", "P_DESC"]
    assert results[0].bm25_score < results[1].bm25_score < 0
    assert results[0].product_title == "Quantum headphones"
    assert results[0].product_description == "Comfortable ear pads"


def test_preserve_missing_description(index_path: Path) -> None:
    """Retrieve a title-only product without inventing a description."""
    results = retrieve_products("UniqueNullDescription", index_path=index_path)
    assert len(results) == 1
    assert results[0].product_id == "P_NULL"
    assert results[0].product_description is None
    assert results[0].product_brand is None


@pytest.mark.parametrize("query", [None, "", " \n\t ", "!@#$%^&*():{}[]\"'", "🚀"])
def test_empty_query_returns_no_candidates(index_path: Path, query: str | None) -> None:
    """Treat absent or token-free input as an empty search."""
    assert retrieve_products(query, index_path=index_path) == []


@pytest.mark.parametrize(
    "query",
    [
        'headset" OR *',
        "product_title:headset",
        "NEAR(headset, 2)",
        "headset NOT quantum",
        "'; DROP TABLE products; --",
    ],
)
def test_fts_and_sql_syntax_is_data(index_path: Path, query: str) -> None:
    """Quote query tokens instead of executing user-supplied FTS or SQL syntax."""
    results = retrieve_products(query, index_path=index_path)
    assert len({record.product_id for record in results}) == len(results)
    assert len(retrieve_products("Twin", index_path=index_path)) == 2


def test_stable_ties_and_normalization(index_path: Path) -> None:
    """Return stable IDs for tied scores and equivalent casing/whitespace/punctuation."""
    first = retrieve_products("Twin", index_path=index_path)
    assert [record.product_id for record in first] == ["P_A", "P_B"]
    assert first[0].bm25_score == first[1].bm25_score
    assert first == retrieve_products("  TWIN!!! \n", index_path=index_path)
    assert retrieve_products("notarealproductzzzz", index_path=index_path) == []


def test_default_limit_is_100(tmp_path: Path) -> None:
    """Bound default candidate output even when more products match."""
    catalog = tmp_path / "many.parquet"
    index = tmp_path / "many.sqlite3"
    write_catalog(catalog, [(f"P{number:03}", "Common product", None) for number in range(110)])
    metadata = build_index(catalog, index)
    assert metadata["product_count"] == 110
    assert metadata["indexed_count"] == 110
    results = retrieve_products("common", index_path=index)
    assert len(results) == 100
    assert results[0].product_id == "P000"
    assert len(retrieve_products("common", k=3, index_path=index)) == 3
    assert len(retrieve_products("common", k=110, index_path=index)) == 110
    assert retrieve_products("common", k=0, index_path=index) == []


def test_reject_duplicate_or_label_contaminated_catalog(
    catalog_path: Path, index_path: Path
) -> None:
    """Reject unsafe input before replacing an existing valid index."""
    original_hash = calculate_sha256(index_path)
    catalog = pl.read_parquet(catalog_path)
    for invalid in (
        pl.concat([catalog, catalog.head(1)]),
        catalog.with_columns(pl.lit("E").alias("esci_label")),
        catalog.drop(PRODUCT_FIELDS[0]),
    ):
        invalid.write_parquet(catalog_path)
        with pytest.raises(ValueError):
            build_index(catalog_path, index_path)
        assert calculate_sha256(index_path) == original_hash


def test_missing_index_is_not_silently_created(tmp_path: Path) -> None:
    """Raise an actionable missing-index error while leaving the filesystem untouched."""
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(FileNotFoundError):
        retrieve_products("headphones", index_path=path)
    assert not path.exists()


def test_reject_overlong_query(index_path: Path) -> None:
    """Reject excessive input before constructing a large FTS expression."""
    with pytest.raises(ValueError, match="query|Query"):
        retrieve_products("a" * 1001, index_path=index_path)


def test_ui_searches_real_index_and_handles_empty_input(
    index_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Show actual title-only results and an actionable empty-input state in Streamlit."""
    settings = replace(load_settings(), index_path=index_path)
    monkeypatch.setattr("smart_product_search.config.load_settings", lambda: settings)
    app = AppTest.from_file(PROJECT_ROOT / "app/main.py").run()
    assert len(app.exception) == 0
    app.selectbox[0].set_value("BM25 only")
    app.text_input[0].set_value("UniqueNullDescription")
    app.button[0].click().run()
    assert len(app.exception) == 0
    assert any("UniqueNullDescription headset" in element.value for element in app.text)
    assert any("No description available" in element.value for element in app.caption)
    app.text_input[0].set_value("")
    app.button[0].click().run()
    assert len(app.info) > 0
    assert len(app.exception) == 0


def test_ui_reports_missing_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Display missing-index setup instructions without creating an empty database."""
    settings = replace(load_settings(), index_path=tmp_path / "missing.sqlite3")
    monkeypatch.setattr("smart_product_search.config.load_settings", lambda: settings)
    app = AppTest.from_file(PROJECT_ROOT / "app/main.py").run()
    assert len(app.exception) == 0
    assert any("build_index.py" in element.value for element in app.error)
    assert not settings.index_path.exists()
