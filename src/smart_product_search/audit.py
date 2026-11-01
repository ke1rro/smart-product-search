import logging
from pathlib import Path

import polars as pl

from smart_product_search.preparation import PRODUCT_FIELDS, PRODUCT_KEYS
from smart_product_search.source import write_json

logger = logging.getLogger(__name__)


def summarize_missing_fields(records: pl.DataFrame) -> dict:
    """Measure nulls, blank strings, and total missingness using native expressions.

    Args:
        records: Raw listings or normalized catalog with all product metadata fields.

    Returns:
        Field names mapped to null, blank, and combined missing counts.
    """
    counts = records.select(
        *[pl.col(field).is_null().sum().alias(f"{field}_null") for field in PRODUCT_FIELDS],
        *[
            pl.col(field).str.strip_chars().eq("").sum().alias(f"{field}_blank")
            for field in PRODUCT_FIELDS
        ],
    ).row(0, named=True)
    return {
        field: {
            "null": counts[f"{field}_null"],
            "blank": counts[f"{field}_blank"],
            "missing_or_blank": counts[f"{field}_null"] + counts[f"{field}_blank"],
        }
        for field in PRODUCT_FIELDS
    }


def build_audit(
    raw_splits: dict[str, pl.DataFrame], catalog: pl.DataFrame, source_manifest: dict
) -> dict:
    """Summarize the complete source and catalog without treating published statistics as data.

    Aggregate full split distributions and missingness, compare query/product sets,
    and count conflicting nonblank metadata before catalog selection.

    Args:
        raw_splits: Full original tables keyed by official split.
        catalog: Unique normalized product catalog.
        source_manifest: Verified source identity and checksum metadata.

    Returns:
        Measured counts, schemas, missingness, distributions, overlaps, and metadata conflicts.
    """
    records = pl.concat(list(raw_splits.values()))
    split_statistics = {}
    for split, table in raw_splits.items():
        labels = table.group_by("esci_label").len().sort("esci_label")
        split_statistics[split] = {
            "rows": table.height,
            "distinct_query_ids": table["query_id"].n_unique(),
            "distinct_query_texts": table["query"].n_unique(),
            "distinct_products": table.select(PRODUCT_KEYS).n_unique(),
            "label_counts": dict(labels.iter_rows()),
            "schema": {name: str(dtype) for name, dtype in table.schema.items()},
            "duplicate_example_ids": table.height - table["example_id"].n_unique(),
            "missing_fields": summarize_missing_fields(table),
        }
    conflicts = records.group_by(PRODUCT_KEYS).agg(
        *[
            pl.col(field).filter(pl.col(field).str.strip_chars().ne("")).drop_nulls().n_unique()
            for field in PRODUCT_FIELDS
        ]
    )
    conflicting_products = conflicts.filter(pl.any_horizontal(pl.col(PRODUCT_FIELDS) > 1))
    duplicate_pairs = records.group_by(["query_id", *PRODUCT_KEYS]).len().filter(pl.col("len") > 1)
    query_id_overlap = (
        raw_splits["train"]
        .select("query_id")
        .unique()
        .join(raw_splits["test"].select("query_id").unique(), on="query_id")
        .height
    )
    query_text_overlap = (
        raw_splits["train"]
        .select("query")
        .unique()
        .join(raw_splits["test"].select("query").unique(), on="query")
        .height
    )
    product_overlap = (
        raw_splits["train"]
        .select(PRODUCT_KEYS)
        .unique()
        .join(raw_splits["test"].select(PRODUCT_KEYS).unique(), on=PRODUCT_KEYS)
        .height
    )
    return {
        "source": source_manifest,
        "splits": split_statistics,
        "total_rows": records.height,
        "distinct_queries": records["query_id"].n_unique(),
        "catalog_products": catalog.height,
        "raw_product_metadata_variants": records.select(
            [*PRODUCT_KEYS, *PRODUCT_FIELDS]
        ).n_unique(),
        "query_id_overlap": query_id_overlap,
        "query_text_overlap": query_text_overlap,
        "product_overlap": product_overlap,
        "duplicate_query_product_groups": duplicate_pairs.height,
        "extra_duplicate_query_product_rows": duplicate_pairs.select(
            (pl.col("len") - 1).sum()
        ).item(),
        "query_ids_with_multiple_texts": records.group_by("query_id")
        .agg(pl.col("query").n_unique())
        .filter(pl.col("query") > 1)
        .height,
        "raw_missing_fields": summarize_missing_fields(records),
        "catalog_missing_fields": summarize_missing_fields(catalog),
        "metadata_conflicts": {
            "products_with_conflicting_nonempty_fields": conflicting_products.height,
            "by_field": {
                field: conflicts.filter(pl.col(field) > 1).height for field in PRODUCT_FIELDS
            },
            "sample_product_keys": conflicting_products.select(PRODUCT_KEYS)
            .sort(PRODUCT_KEYS)
            .head(5)
            .to_dicts(),
            "definition": (
                "More than one distinct nonblank raw value for a field of the same product"
            ),
        },
        "normalization": (
            "NFC Unicode, HTML entity decoding, script/style/tag removal, whitespace collapse"
        ),
        "deduplication": (
            "Prefer title present, then field completeness, then normalized metadata and "
            "original title lexically; keep one whole listing"
        ),
        "catalog_scope": (
            "Transductive listing-only catalog from both official splits; "
            "no query or gold-label features"
        ),
    }


def write_audit(audit: dict, audit_dir: Path) -> None:
    """Write reproducible measured JSON and a compact Markdown EDA summary.

    Args:
        audit: Completed measured audit record.
        audit_dir: Ignored artifact directory for the audit files.
    """
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_json(audit_dir / "audit.json", audit)
    lines = [
        "# Amazon ESCI data audit",
        "",
        f"Source revision: `{audit['source']['revision']}`.",
        f"Measured judgments: {audit['total_rows']:,}; "
        f"catalog products: {audit['catalog_products']:,}.",
        "",
        "| Split | Rows | Query IDs | Products | E | S | C | I |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, stats in audit["splits"].items():
        labels = stats["label_counts"]
        lines.append(
            f"| {split} | {stats['rows']} | {stats['distinct_query_ids']} | "
            f"{stats['distinct_products']} | {labels.get('E', 0)} | {labels.get('S', 0)} | "
            f"{labels.get('C', 0)} | {labels.get('I', 0)} |"
        )
    lines.extend(
        [
            "",
            "| Field | Missing raw rows | Missing catalog products |",
            "| --- | ---: | ---: |",
        ]
    )
    for field in PRODUCT_FIELDS:
        lines.append(
            f"| {field} | {audit['raw_missing_fields'][field]['missing_or_blank']} | "
            f"{audit['catalog_missing_fields'][field]['missing_or_blank']} |"
        )
    lines.extend(
        [
            "",
            f"Query ID overlap: {audit['query_id_overlap']}; "
            f"exact query-text overlap: {audit['query_text_overlap']}.",
            f"Products shared across splits: {audit['product_overlap']} "
            "(allowed listing-only reference data).",
            "Conflicting metadata products: "
            f"{audit['metadata_conflicts']['products_with_conflicting_nonempty_fields']}.",
            f"Duplicate query-product groups: {audit['duplicate_query_product_groups']}.",
            "",
            f"Selection policy: {audit['deduplication']}.",
            "Missing-title products remain in the complete catalog with has_title=false; "
            "no titles are invented.",
            "Gold labels are retained only in separate judgments. "
            "No index or model has been built.",
            "Historical listings and incomplete relevance judgments do not describe live "
            "prices, availability, or exhaustive relevance.",
            "The pinned repack card has no explicit license field; "
            "original Amazon source specifies Apache 2.0.",
            "",
        ]
    )
    (audit_dir / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote measured audit to %s", audit_dir)
