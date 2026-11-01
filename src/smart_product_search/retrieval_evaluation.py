import json
import logging
import platform
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from smart_product_search.preparation import JUDGMENT_COLUMNS
from smart_product_search.retrieval import FIELD_WEIGHTS, TOKENIZER, open_index, retrieve_products
from smart_product_search.source import DATASET_ID, DATASET_REVISION, calculate_sha256, write_json

logger = logging.getLogger(__name__)


def select_query_sample(judgments: pl.DataFrame, sample_size: int) -> pl.DataFrame:
    """Select evenly spaced official query IDs without consulting their relevance labels.

    Args:
        judgments: Official test judgments containing query_id and original query text.
        sample_size: Positive maximum query count; values beyond available queries use them all.

    Returns:
        Deterministically ordered unique queries spanning the entire sorted query-ID range.
    """
    if sample_size <= 0:
        raise ValueError("Sample size must be positive")
    queries = judgments.select("query_id", "query").unique().sort("query_id")
    count = min(sample_size, queries.height)
    if count <= 1:
        return queries.head(count)
    indices = [number * (queries.height - 1) // (count - 1) for number in range(count)]
    return queries[indices]


def summarize_coverage(rows: pl.DataFrame) -> list[dict]:
    """Aggregate judged-positive coverage without treating unjudged products as irrelevant.

    Args:
        rows: Per-query/group/cutoff rows with positive_count and hit_count values.

    Returns:
        Macro/micro coverage, hit/positive pair counts, and eligible queries by group and K.
    """
    return (
        rows.filter(pl.col("positive_count") > 0)
        .group_by("label_group", "k")
        .agg(
            pl.len().alias("eligible_queries"),
            pl.col("positive_count").sum().alias("judged_positive_pairs"),
            pl.col("hit_count").sum().alias("retrieved_positive_pairs"),
            (pl.col("hit_count") / pl.col("positive_count")).mean().alias("macro_coverage"),
        )
        .with_columns(
            (pl.col("retrieved_positive_pairs") / pl.col("judged_positive_pairs")).alias(
                "micro_coverage"
            )
        )
        .sort("label_group", "k")
        .to_dicts()
    )


def evaluate_retrieval(
    data_dir: Path, index_path: Path, output_dir: Path, sample_size: int = 200
) -> dict:
    """Measure one fixed BM25 system on a documented official-test query sample.

    Verify the held-out judgments and matching catalog identity, choose queries independently
    of labels, then retrieve top 100 once per query. Score E, S, and E-or-S recovery at 10/100
    after inference and record production API latency, including connection and metadata fetch.

    Args:
        data_dir: Root containing the original prepared catalog/judgment manifest.
        index_path: Complete catalog index used without access to gold judgments during search.
        output_dir: Ignored directory for selected queries, ranks, query scores, and summary.
        sample_size: Number of systematically spaced query IDs, default 200; no random sampling.

    Returns:
        Measured coverage, query latency, reproducibility metadata, and explicit limitations.
    """
    processed = data_dir / "processed"
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["repo_id"], manifest["revision"]) != (DATASET_ID, DATASET_REVISION):
        raise ValueError("Evaluation requires the approved pinned dataset")
    judgment_path = processed / "judgments_test.parquet"
    expected = next(item for item in manifest["files"] if item["name"] == judgment_path.name)
    if calculate_sha256(judgment_path) != expected["sha256"]:
        raise ValueError("Official test judgment checksum mismatch")
    judgments = pl.read_parquet(judgment_path)
    if judgments.columns != JUDGMENT_COLUMNS or judgments.height != 181701:
        raise ValueError("Evaluation requires all 181701 official test judgments")
    if judgments.filter((pl.col("split") != "test") | (pl.col("product_locale") != "us")).height:
        raise ValueError("Evaluation requires official English Small test judgments only")
    with open_index(index_path) as connection:
        index_metadata = json.loads(
            connection.execute(
                "SELECT value FROM index_metadata WHERE key=?", ("build",)
            ).fetchone()[0]
        )
    catalog_record = next(item for item in manifest["files"] if item["name"] == "catalog.parquet")
    if index_metadata["catalog_sha256"] != catalog_record["sha256"]:
        raise ValueError("Index and evaluation catalog provenance differ")
    queries = select_query_sample(judgments, sample_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "selected_queries.json", {"queries": queries.to_dicts()})
    rows = []
    outcomes = []
    logger.info("Evaluating %s fixed test queries against the full catalog", queries.height)
    for number, (query_id, query) in enumerate(queries.iter_rows(), start=1):
        started = time.perf_counter()
        error_message = None
        try:
            results = retrieve_products(query, k=100, index_path=index_path)
        except (ValueError, FileNotFoundError, sqlite3.DatabaseError) as error:
            results = []
            error_message = str(error)
        latency_ms = (time.perf_counter() - started) * 1000
        ranked_ids = [result.product_id for result in results]
        gold = judgments.filter(pl.col("query_id") == query_id)
        positive_ids = {
            label: set(gold.filter(pl.col("esci_label") == label)["product_id"].to_list())
            for label in ("E", "S")
        }
        for group, positives in {
            "E": positive_ids["E"],
            "S": positive_ids["S"],
            "E_or_S": positive_ids["E"] | positive_ids["S"],
        }.items():
            for k in (10, 100):
                rows.append(
                    {
                        "query_id": query_id,
                        "label_group": group,
                        "k": k,
                        "positive_count": len(positives),
                        "hit_count": len(positives.intersection(ranked_ids[:k])),
                    }
                )
        outcomes.append(
            {
                "query_id": query_id,
                "query": query,
                "latency_ms": latency_ms,
                "error": error_message,
                "product_ids": ranked_ids,
                "bm25_scores": [result.bm25_score for result in results],
            }
        )
        if number % 20 == 0:
            logger.info("Evaluated %s/%s queries", number, queries.height)
    per_query = pl.DataFrame(rows)
    per_query.write_parquet(output_dir / "per_query_coverage.parquet")
    write_json(output_dir / "outcomes.json", {"outcomes": outcomes})
    latency = (
        pl.DataFrame(outcomes)
        .select(
            pl.col("latency_ms").mean().alias("mean_ms"),
            pl.col("latency_ms").median().alias("median_ms"),
            pl.col("latency_ms").quantile(0.95, interpolation="linear").alias("p95_ms"),
            pl.col("latency_ms").min().alias("min_ms"),
            pl.col("latency_ms").max().alias("max_ms"),
        )
        .row(0, named=True)
    )
    report = {
        "utc_time": datetime.now(UTC).isoformat(),
        "source_revision": DATASET_REVISION,
        "catalog_products": index_metadata["product_count"],
        "index_metadata": index_metadata,
        "index_sha256": calculate_sha256(index_path),
        "judgments_sha256": expected["sha256"],
        "runtime_retrieval_sha256": calculate_sha256(Path(__file__).with_name("retrieval.py")),
        "evaluation_code_sha256": calculate_sha256(Path(__file__)),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "sqlite_version": sqlite3.sqlite_version,
        "polars_version": pl.__version__,
        "field_weights": list(FIELD_WEIGHTS),
        "tokenizer": TOKENIZER,
        "query_policy": (
            "quoted literal Unicode alphanumeric OR tokens; no stemming/stopword removal"
        ),
        "selection": (
            "Equally spaced positions including endpoints in sorted unique official query IDs"
        ),
        "official_test_queries": judgments["query_id"].n_unique(),
        "sample_queries": queries.height,
        "failed_queries": sum(item["error"] is not None for item in outcomes),
        "no_hit_queries": sum(not item["product_ids"] for item in outcomes),
        "coverage": summarize_coverage(per_query),
        "latency": latency,
        "latency_protocol": (
            "One sequential top 100 production API call per selected query; no warmup; "
            "OS cache state uncontrolled; no result cache"
        ),
        "limitations": (
            "ESCI annotations are incomplete across the full catalog. Coverage recovers known "
            "judged E/S products on this systematic sample, not true corpus-wide Recall@K. "
            "Unjudged products are unknown, not irrelevant. No tuning/model comparisons/"
            "ablations were performed."
        ),
    }
    write_json(output_dir / "summary.json", report)
    logger.info("Measured retrieval coverage and latency saved to %s", output_dir)
    return report
