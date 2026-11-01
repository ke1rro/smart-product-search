import json
import logging
import re
import sqlite3
import unicodedata
from collections.abc import Generator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from smart_product_search.config import load_settings
from smart_product_search.preparation import CATALOG_COLUMNS, PRODUCT_FIELDS, PRODUCT_KEYS
from smart_product_search.source import DATASET_ID, DATASET_REVISION, calculate_sha256
from smart_product_search.workflow import preparation_fingerprint

INDEX_SCHEMA_VERSION = 1
FIELD_WEIGHTS = (5.0, 3.0, 1.0, 2.0, 1.0)
TOKENIZER = "unicode61 remove_diacritics 2"
MAX_QUERY_CHARACTERS = 1000
MAX_QUERY_TERMS = 64
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductResult:
    """A real catalog candidate with a lexical score, not a predicted relevance class.

    Attributes:
        product_locale: Locale forming part of stable product identity.
        product_id: Original product identifier.
        product_title: Normalized listing title, if present.
        product_brand: Optional real brand metadata.
        product_color: Optional real color metadata.
        product_bullet_point: Optional normalized product attributes.
        product_description: Optional normalized full listing description.
        bm25_score: SQLite BM25 score; smaller values rank better.
    """

    product_locale: str
    product_id: str
    product_title: str | None
    product_brand: str | None
    product_color: str | None
    product_bullet_point: str | None
    product_description: str | None
    bm25_score: float


def build_fts_query(query: str | None) -> str:
    """Construct an OR expression from quoted literal tokens, never user FTS operators.

    Args:
        query: Natural-language text or an absent query; excessive/non-text input is rejected.

    Returns:
        Parameter-safe FTS expression, or an empty string if there are no searchable tokens.
    """
    if query is None:
        return ""
    if not isinstance(query, str) or len(query) > MAX_QUERY_CHARACTERS:
        raise ValueError(f"Query must be text with at most {MAX_QUERY_CHARACTERS} characters")
    normalized = unicodedata.normalize("NFC", query).casefold()
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+", normalized)))
    if len(terms) > MAX_QUERY_TERMS:
        raise ValueError(f"Query must contain at most {MAX_QUERY_TERMS} distinct terms")
    return " OR ".join(f'"{term}"' for term in terms)


@contextmanager
def open_index(index_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Open a compatible existing search database read-only without creating missing files.

    Args:
        index_path: Completed SQLite search index path.

    Yields:
        Row-based connection automatically closed when the context exits.
    """
    if not index_path.is_file():
        raise FileNotFoundError(f"Search index missing at {index_path}; run scripts/build_index.py")
    with closing(
        sqlite3.connect(index_path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as connection:
        connection.row_factory = sqlite3.Row
        if connection.execute("PRAGMA user_version").fetchone()[0] != INDEX_SCHEMA_VERSION:
            raise ValueError("Unsupported or incomplete search index; run scripts/build_index.py")
        yield connection


def build_index(catalog_path: Path, index_path: Path) -> dict:
    """Atomically rebuild every catalog product into a deterministic metadata/FTS5 database.

    Validate the serving-only schema and unique identities before touching the target.
    Insert products in composite-ID order, build the external-content index, check actual
    indexed document counts and integrity, and publish the database only on success.

    Args:
        catalog_path: Complete prepared serving catalog Parquet, never a judgment table.
        index_path: Destination SQLite file; a previous valid file survives a failed rebuild.

    Returns:
        Deterministic build metadata with counts, catalog checksum, tokenizer, and field weights.
    """
    if catalog_path.resolve() == index_path.resolve():
        raise ValueError("Index destination must differ from the catalog source")
    schema = pl.read_parquet_schema(catalog_path)
    if list(schema) != CATALOG_COLUMNS:
        raise ValueError("Expected the label-free prepared catalog schema")
    fields = [*PRODUCT_KEYS, *PRODUCT_FIELDS]
    if any(schema[name] != pl.String for name in fields):
        raise ValueError("Catalog identity and listing fields must be strings")
    catalog = pl.read_parquet(catalog_path, columns=fields).sort(PRODUCT_KEYS)
    if catalog.select(PRODUCT_KEYS).is_duplicated().any():
        raise ValueError("Duplicate product identities in catalog")
    if catalog.select(
        pl.any_horizontal(
            pl.col(PRODUCT_KEYS).is_null(), pl.col(PRODUCT_KEYS).str.strip_chars().eq("")
        ).any()
    ).item():
        raise ValueError("Missing catalog product identity")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_name(index_path.name + ".building")
    temporary.unlink(missing_ok=True)
    logger.info("Building SQLite FTS5 for all %s catalog products", catalog.height)
    try:
        with closing(sqlite3.connect(temporary)) as connection:
            connection.executescript(
                "PRAGMA page_size=4096;"
                "CREATE TABLE products ("
                "rowid INTEGER PRIMARY KEY, product_locale TEXT NOT NULL, product_id TEXT NOT NULL,"
                "product_title TEXT, product_brand TEXT, product_color TEXT,"
                "product_bullet_point TEXT, product_description TEXT,"
                "UNIQUE(product_locale, product_id));"
                "CREATE VIRTUAL TABLE product_fts USING fts5("
                "product_title, product_brand, product_color, product_bullet_point,"
                "product_description, content='products', content_rowid='rowid',"
                "tokenize='unicode61 remove_diacritics 2');"
                "CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            )
            for batch in catalog.iter_slices(10000):
                connection.executemany(
                    "INSERT INTO products (product_locale,product_id,product_title,product_brand,"
                    "product_color,product_bullet_point,product_description) "
                    "VALUES (?,?,?,?,?,?,?)",
                    batch.iter_rows(),
                )
            connection.execute("INSERT INTO product_fts(product_fts) VALUES('rebuild')")
            connection.execute("INSERT INTO product_fts(product_fts) VALUES('optimize')")
            connection.execute(
                "INSERT INTO product_fts(product_fts,rank) VALUES('integrity-check',1)"
            )
            count = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
            indexed_count = connection.execute(
                "SELECT COUNT(*) FROM product_fts_docsize"
            ).fetchone()[0]
            if count != catalog.height or indexed_count != count:
                raise ValueError("Catalog, metadata table, and actual FTS document counts differ")
            metadata = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "product_count": count,
                "indexed_count": indexed_count,
                "catalog_sha256": calculate_sha256(catalog_path),
                "index_code_sha256": calculate_sha256(Path(__file__)),
                "sqlite_version": sqlite3.sqlite_version,
                "fields": PRODUCT_FIELDS,
                "field_weights": list(FIELD_WEIGHTS),
                "tokenizer": TOKENIZER,
                "query_policy": "quoted literal Unicode alphanumeric tokens combined with OR",
            }
            connection.execute(
                "INSERT INTO index_metadata VALUES (?,?)",
                ("build", json.dumps(metadata, sort_keys=True)),
            )
            connection.execute(f"PRAGMA user_version={INDEX_SCHEMA_VERSION}")
            connection.commit()
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity check failed")
        temporary.replace(index_path)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("Published %s with %s searchable products", index_path, count)
    return metadata


def build_prepared_index(data_dir: Path, index_path: Path) -> dict:
    """Verify the complete approved prepared catalog before building the serving index.

    Args:
        data_dir: Root containing the completed Iteration 1 processed manifest and catalog.
        index_path: Destination for the complete SQLite search index.

    Returns:
        Complete verified index build metadata.
    """
    processed = data_dir / "processed"
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["repo_id"], manifest["revision"]) != (DATASET_ID, DATASET_REVISION):
        raise ValueError("Prepared catalog is not from the approved pinned dataset")
    if manifest["catalog_products"] != 482105:
        raise ValueError("Require the complete prepared catalog of 482105 products")
    if manifest["preparation_fingerprint"] != preparation_fingerprint():
        raise ValueError("Prepared catalog provenance is stale; run prepare_data.py")
    catalog_path = processed / "catalog.parquet"
    expected = next(record for record in manifest["files"] if record["name"] == "catalog.parquet")
    if calculate_sha256(catalog_path) != expected["sha256"]:
        raise ValueError("Prepared catalog checksum mismatch")
    if (
        pl.scan_parquet(catalog_path).select(pl.len()).collect().item()
        != manifest["catalog_products"]
    ):
        raise ValueError("Prepared catalog record count differs from manifest")
    return build_index(catalog_path, index_path)


def retrieve_products(
    query: str | None, k: int = 100, *, index_path: Path | None = None
) -> list[ProductResult]:
    """Retrieve catalog candidates using weighted SQLite BM25 and deterministic identity ties.

    Args:
        query: Arbitrary English shopping text, not FTS/SQL syntax; empty input returns no results.
        k: Maximum candidate count; zero returns no results and negative values are rejected.
        index_path: Optional explicit index path; otherwise use config/search.toml.

    Returns:
        Ranked real products with metadata and BM25 scores; smaller scores rank better.
    """
    if type(k) is not int or k < 0:
        raise ValueError("Candidate limit k must be a nonnegative integer")
    expression = build_fts_query(query)
    if not expression or k == 0:
        return []
    path = index_path if index_path is not None else load_settings().index_path
    with open_index(path) as connection:
        rows = connection.execute(
            "SELECT p.product_locale,p.product_id,p.product_title,p.product_brand,p.product_color,"
            "p.product_bullet_point,p.product_description,"
            "bm25(product_fts,?,?,?,?,?) AS bm25_score "
            "FROM product_fts JOIN products p ON p.rowid=product_fts.rowid "
            "WHERE product_fts MATCH ? "
            "ORDER BY bm25_score ASC,p.product_locale ASC,p.product_id ASC LIMIT ?",
            (*FIELD_WEIGHTS, expression, k),
        ).fetchall()
    logger.debug("Retrieved %s lexical candidates", len(rows))
    return [ProductResult(**dict(row)) for row in rows]
