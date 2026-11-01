import logging
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

import polars as pl

from smart_product_search.source import DATASET_REVISION, verify_source

EXPECTED_COUNTS = {"train": 419653, "test": 181701}
PRODUCT_KEYS = ["product_locale", "product_id"]
PRODUCT_FIELDS = [
    "product_title",
    "product_brand",
    "product_color",
    "product_bullet_point",
    "product_description",
]
RAW_SCHEMA = {
    "example_id": pl.Int64,
    "query": pl.String,
    "query_id": pl.Int64,
    "product_id": pl.String,
    "product_locale": pl.String,
    "esci_label": pl.String,
    "small_version": pl.Int64,
    "large_version": pl.Int64,
    "split": pl.String,
    "product_title": pl.String,
    "product_description": pl.String,
    "product_bullet_point": pl.String,
    "product_brand": pl.String,
    "product_color": pl.String,
}
JUDGMENT_COLUMNS = [
    "example_id",
    "query_id",
    "query",
    "product_locale",
    "product_id",
    "esci_label",
    "split",
]
CATALOG_COLUMNS = [
    *PRODUCT_KEYS,
    *PRODUCT_FIELDS,
    "product_title_original",
    "has_title",
    "index_text",
]
BLOCK_TAGS = {"br", "p", "div", "li", "ul", "ol", "tr", "td", "table", "hr", "h1", "h2", "h3"}
logger = logging.getLogger(__name__)


class ListingText(HTMLParser):
    """Plain listing text with block boundaries preserved and script/style content removed.

    Attributes:
        parts: Visible text fragments in their original order.
        ignored_depth: Nesting depth of ignored script and style elements.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Skip executable/style elements and separate structural listing blocks.

        Args:
            tag: Lowercase element name supplied by HTMLParser.
            attrs: Parsed attributes, deliberately excluded from product text.
        """
        if tag in {"script", "style"}:
            self.ignored_depth += 1
        if tag in BLOCK_TAGS and self.ignored_depth == 0:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        """Close ignored elements and separate structural listing blocks.

        Args:
            tag: Lowercase closing element name.
        """
        if tag in {"script", "style"} and self.ignored_depth:
            self.ignored_depth -= 1
        if tag in BLOCK_TAGS and self.ignored_depth == 0:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        """Keep visible text while discarding script/style bodies.

        Args:
            data: Text fragment emitted by HTMLParser.
        """
        if self.ignored_depth == 0:
            self.parts.append(data)


def normalize_listing_text(value: str | None) -> str | None:
    """Normalize NFC Unicode and whitespace while stripping untrusted HTML markup.

    Args:
        value: Optional raw product text; missing or empty text remains missing.

    Returns:
        Plain listing text preserving case, negations, identifiers, units, and punctuation.
    """
    if value is None:
        return None
    parser = ListingText()
    parser.feed(value)
    parser.close()
    text = " ".join(unicodedata.normalize("NFC", "".join(parser.parts)).split())
    return text or None


def validate_raw_split(records: pl.DataFrame, split: str, expected_count: int) -> None:
    """Reject incomplete, malformed, or incorrectly identified source split records.

    Args:
        records: Complete native split with the published source columns.
        split: Expected official split name.
        expected_count: Published count, supplied explicitly for the split being checked.
    """
    if records.height != expected_count:
        raise ValueError(
            f"{split}: expected {expected_count} rows, observed {records.height}; stop source audit"
        )
    if any(records.schema.get(name) != dtype for name, dtype in RAW_SCHEMA.items()):
        raise ValueError(f"{split}: missing or incorrectly typed source columns: {records.schema}")
    if set(records.columns) - {*RAW_SCHEMA, "__index_level_0__"}:
        raise ValueError(f"{split}: unexpected source columns")
    required = ["example_id", "query_id", "product_id", "query", "product_locale", "split"]
    invalid = records.select(
        pl.any_horizontal(
            pl.col(required).is_null(),
            pl.col("product_id", "query").str.strip_chars().eq(""),
            pl.col("product_locale") != "us",
            pl.col("small_version") != 1,
            ~pl.col("esci_label").is_in(["E", "S", "C", "I"]),
            pl.col("split") != split,
        )
        .fill_null(True)
        .sum()
    ).item()
    if invalid:
        raise ValueError(
            f"{split}: {invalid} rows violate identifier, locale, version, "
            "label, or split contracts"
        )


def read_raw_splits(data_dir: Path) -> dict[str, pl.DataFrame]:
    """Read and validate both complete source splits after checksum verification.

    Args:
        data_dir: Data root containing the pinned downloaded snapshot.

    Returns:
        Mapping from official split name to every original row in that split.
    """
    verify_source(data_dir)
    raw_splits = {}
    for split, expected_count in EXPECTED_COUNTS.items():
        logger.info("Reading and validating complete %s split", split)
        paths = sorted((data_dir / "raw" / DATASET_REVISION / "data").glob(f"{split}-*.parquet"))
        table = pl.read_parquet(paths)
        validate_raw_split(table, split, expected_count)
        raw_splits[split] = table
    return raw_splits


def build_catalog(records: pl.DataFrame) -> pl.DataFrame:
    """Build a deterministic catalog using listing metadata only.

    Prefer a nonempty title, then the greatest number of nonempty normalized fields,
    then lexical order of normalized metadata and original title. Never merge conflicting rows.

    Args:
        records: Source pairs from both splits; query and judgment fields are ignored.

    Returns:
        Sorted unique products with original titles, normalized metadata, and plain index text.
    """
    listings = records.select([*PRODUCT_KEYS, *PRODUCT_FIELDS]).unique()
    listings = listings.with_columns(pl.col("product_title").alias("product_title_original"))
    listings = listings.with_columns(
        pl.col(PRODUCT_FIELDS).map_elements(normalize_listing_text, return_dtype=pl.String)
    ).with_columns(
        pl.col("product_title").is_not_null().alias("has_title"),
        pl.sum_horizontal(pl.col(PRODUCT_FIELDS).is_not_null()).alias("metadata_completeness"),
    )
    catalog = (
        listings.sort(
            [
                *PRODUCT_KEYS,
                "has_title",
                "metadata_completeness",
                *PRODUCT_FIELDS,
                "product_title_original",
            ],
            descending=[False, False, True, True, *([False] * (len(PRODUCT_FIELDS) + 1))],
            nulls_last=True,
        )
        .unique(subset=PRODUCT_KEYS, keep="first", maintain_order=True)
        .with_columns(
            pl.concat_str(PRODUCT_FIELDS, separator=" ", ignore_nulls=True).alias("index_text")
        )
    )
    return catalog.select(CATALOG_COLUMNS)


def extract_judgments(records: pl.DataFrame) -> pl.DataFrame:
    """Preserve all original query-product judgments separately from serving metadata.

    Args:
        records: Native source records without filtering or label-based selection.

    Returns:
        Judgment identifiers, original query text, gold label, and split sorted by example ID.
    """
    return records.select(JUDGMENT_COLUMNS).sort("example_id")


def validate_prepared(
    raw_splits: dict[str, pl.DataFrame], catalog: pl.DataFrame, judgments: dict[str, pl.DataFrame]
) -> dict:
    """Verify preserved judgments, isolated queries, catalog identity, and label-free metadata.

    Args:
        raw_splits: Complete source tables by official split.
        catalog: Prepared serving catalog with no judgment columns.
        judgments: Prepared evaluation-only judgment tables by official split.

    Returns:
        Named boolean checks that passed; raise ValueError on any failed contract.
    """
    if catalog.columns != CATALOG_COLUMNS:
        raise ValueError("Catalog schema must exclude labels and query/judgment associations")
    if catalog.select(PRODUCT_KEYS).is_duplicated().any():
        raise ValueError("Catalog product identities are not unique")
    for split, raw in raw_splits.items():
        if not judgments[split].equals(extract_judgments(raw)):
            raise ValueError(f"{split}: prepared judgments differ from original source")
    query_overlap = (
        raw_splits["train"]
        .select("query_id")
        .unique()
        .join(raw_splits["test"].select("query_id").unique(), on="query_id", how="inner")
        .height
    )
    if query_overlap:
        raise ValueError(f"Train/test query ID overlap: {query_overlap}")
    expected_catalog = build_catalog(pl.concat(list(raw_splits.values())))
    if not catalog.equals(expected_catalog):
        raise ValueError("Catalog metadata/text differs from deterministic label-free preparation")
    return {
        "judgments_preserved": True,
        "catalog_unique": True,
        "catalog_metadata_verified": True,
        "labels_separate": True,
        "query_splits_disjoint": True,
    }
