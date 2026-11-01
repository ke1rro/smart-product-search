import json
import logging
import math
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import polars as pl

from smart_product_search.config import SearchSettings
from smart_product_search.retrieval import retrieve_products
from smart_product_search.retrieval_evaluation import select_query_sample, summarize_coverage
from smart_product_search.semantic import TEXT_VERSION, get_reranker
from smart_product_search.source import calculate_sha256, write_json

GAINS = {"E": 3, "S": 2, "C": 1, "I": 0}
logger = logging.getLogger(__name__)


def judged_pool_ndcg(ranked_ids: list[str], gains: dict[str, int]) -> float | None:
    """Measure nDCG@10 on the judged projection of a fixed lexical candidate pool.

    Args:
        ranked_ids: Complete inferred candidate ordering, before any evaluation label lookup.
        gains: Known labels converted to direct gains only for judged products in the original pool.

    Returns:
        Conditional nDCG@10, or None for a pool with no known positive gain; unknowns are skipped.
    """
    ideal = math.fsum(
        gain / math.log2(rank + 2)
        for rank, gain in enumerate(sorted(gains.values(), reverse=True)[:10])
    )
    if ideal == 0:
        return None
    judged = [gains[product_id] for product_id in ranked_ids if product_id in gains]
    actual = math.fsum(gain / math.log2(rank + 2) for rank, gain in enumerate(judged[:10]))
    return actual / ideal


def evaluate_semantic(settings: SearchSettings, fixed_queries: Path, output_dir: Path) -> dict:
    """Compare one MiniLM reranker with unchanged BM25 on the exact Iteration 2 query set.

    Verify data/index identity, time BM25 once, then rank that same candidate pool using only
    product text. Consult gold labels afterward for coverage and conditional judged-pool nDCG.
    Retain raw rankings, per-query metrics, encoder counters, and latency for reproducibility.

    Args:
        settings: Centralized runtime settings for the index and CPU model.
        fixed_queries: Existing selected_queries.json from the accepted lexical evaluation.
        output_dir: Ignored destination for this iteration's raw and summarized measurements.

    Returns:
        Paired measured coverage, conditional nDCG, latencies, cache counts, and provenance.
    """
    if (settings.retrieval_k, settings.rerank_k) != (100, 50):
        raise ValueError("Iteration 3 validation requires BM25 top100 and semantic top50")
    baseline_dir = fixed_queries.parent
    baseline = json.loads((baseline_dir / "summary.json").read_text())
    if calculate_sha256(settings.index_path) != baseline["index_sha256"]:
        raise ValueError("Index differs from the accepted Iteration 2 index")
    judgment_path = settings.data_dir / "processed/judgments_test.parquet"
    if calculate_sha256(judgment_path) != baseline["judgments_sha256"]:
        raise ValueError("Judgments differ from the accepted official test judgments")
    judgments = pl.read_parquet(judgment_path)
    queries = pl.DataFrame(json.loads(fixed_queries.read_text())["queries"]).select(
        "query_id", "query"
    )
    if queries.height != 200 or not queries.equals(select_query_sample(judgments, 200)):
        raise ValueError("Use the unchanged 200-query Iteration 2 selection")
    previous_ranks = {
        row["query_id"]: row["product_ids"]
        for row in json.loads((baseline_dir / "outcomes.json").read_text())["outcomes"]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "selected_queries.json", {"queries": queries.to_dicts()})
    started = time.perf_counter()
    reranker = get_reranker(settings)
    model_load_seconds = time.perf_counter() - started
    initial_cache = reranker.cache_info()
    outcomes: list[dict] = []
    coverage_rows = []
    ranking_rows = []
    for number, (query_id, query) in enumerate(queries.iter_rows(), start=1):
        started = time.perf_counter()
        candidates = retrieve_products(
            query, k=settings.retrieval_k, index_path=settings.index_path
        )
        bm25_ms = (time.perf_counter() - started) * 1000
        lexical_ids = [product.product_id for product in candidates]
        if lexical_ids != previous_ranks[query_id]:
            raise ValueError(f"BM25 candidate ordering changed for query {query_id}")
        before = reranker.cache_info()
        started = time.perf_counter()
        semantic = reranker.rerank(query, candidates, k=len(candidates))
        rerank_ms = (time.perf_counter() - started) * 1000
        after = reranker.cache_info()
        semantic_ids = [product.product_id for product in semantic]
        gold = judgments.filter(pl.col("query_id") == query_id)
        pool = set(lexical_ids)
        gains = {
            product_id: GAINS[label]
            for product_id, label in gold.select("product_id", "esci_label").iter_rows()
            if product_id in pool
        }
        for mode, ranked_ids in {"bm25": lexical_ids, "semantic": semantic_ids}.items():
            ranking_rows.append(
                {
                    "query_id": query_id,
                    "mode": mode,
                    "ndcg_at_10": judged_pool_ndcg(ranked_ids, gains),
                    "judged_pool_size": len(gains),
                    "judged_in_actual_top10": sum(
                        product_id in gains for product_id in ranked_ids[:10]
                    ),
                }
            )
            for group, labels in {"E": ["E"], "S": ["S"], "E_or_S": ["E", "S"]}.items():
                positives = set(
                    gold.filter(pl.col("esci_label").is_in(labels))["product_id"].to_list()
                )
                for k in (10, 50):
                    coverage_rows.append(
                        {
                            "query_id": query_id,
                            "mode": mode,
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
                "bm25_ms": bm25_ms,
                "rerank_ms": rerank_ms,
                "semantic_total_ms": bm25_ms + rerank_ms,
                "candidate_count": len(candidates),
                "bm25_ids": lexical_ids,
                "semantic_ids": semantic_ids,
                "bm25_scores": [product.bm25_score for product in candidates],
                "semantic_scores": [product.semantic_score for product in semantic],
                "products_encoded": after["products_encoded"] - before["products_encoded"],
                "cache_hits": after["cache_hits"] - before["cache_hits"],
            }
        )
        if number % 10 == 0:
            logger.info("Validated %s/%s fixed queries; cache=%s", number, queries.height, after)
            write_json(output_dir / "progress.json", {"completed_queries": number, "cache": after})
    coverage = pl.DataFrame(coverage_rows)
    ranking = pl.DataFrame(ranking_rows)
    coverage.write_parquet(output_dir / "per_query_coverage.parquet")
    ranking.write_parquet(output_dir / "per_query_ndcg.parquet")
    write_json(output_dir / "outcomes.json", {"outcomes": outcomes})
    timings = pl.DataFrame(outcomes)
    latency = {
        mode: timings.select(
            pl.col(column).median().alias("median_ms"),
            pl.col(column).quantile(0.95, interpolation="linear").alias("p95_ms"),
            pl.col(column).mean().alias("mean_ms"),
        ).row(0, named=True)
        for mode, column in {
            "bm25": "bm25_ms",
            "semantic": "semantic_total_ms",
            "reranking": "rerank_ms",
        }.items()
    }
    report = {
        "utc_time": datetime.now(UTC).isoformat(),
        "source_revision": baseline["source_revision"],
        "catalog_products": baseline["catalog_products"],
        "index_sha256": baseline["index_sha256"],
        "judgments_sha256": baseline["judgments_sha256"],
        "fixed_queries_sha256": calculate_sha256(fixed_queries),
        "query_count": queries.height,
        "candidate_count": sum(row["candidate_count"] for row in outcomes),
        "settings": {
            key: value for key, value in asdict(settings).items() if key.startswith("embedding_")
        },
        "execution_device": str(reranker.model.device),
        "model_load_seconds": model_load_seconds,
        "initial_cache": initial_cache,
        "final_cache": reranker.cache_info(),
        "coverage": {
            mode: summarize_coverage(coverage.filter(pl.col("mode") == mode))
            for mode in ("bm25", "semantic")
        },
        "ndcg": ranking.group_by("mode")
        .agg(
            pl.col("ndcg_at_10").mean().alias("mean_ndcg_at_10"),
            pl.col("ndcg_at_10").count().alias("eligible_queries"),
            pl.col("judged_in_actual_top10").mean().alias("mean_judged_in_actual_top10"),
        )
        .sort("mode")
        .to_dicts(),
        "gain_mapping": GAINS,
        "latency": latency,
        "ndcg_protocol": (
            "Judged projection of full inferred top100 ordering, ideal over the same original "
            "judged BM25 pool; unknowns skipped, zero-ideal pools excluded"
        ),
        "cache_protocol": (
            "One sequential pass with shared bounded on-demand LRU; model initialization "
            "excluded from query latency; OS cache state uncontrolled"
        ),
        "preprocessing_version": TEXT_VERSION,
        "versions": {
            name: version(name)
            for name in ("sentence-transformers", "torch", "transformers", "numpy")
        },
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "code_sha256": {
            name: calculate_sha256(Path(__file__).with_name(name))
            for name in (
                "semantic.py",
                "search.py",
                "semantic_evaluation.py",
                "retrieval.py",
                "config.py",
            )
        },
        "limitations": (
            "Paired fixed-query component validation only; judgments incomplete, coverage "
            "not exhaustive recall and projected nDCG not catalog-wide or literal "
            "display-position nDCG. No tuning or other models."
        ),
    }
    write_json(output_dir / "summary.json", report)
    logger.info("Paired semantic evaluation saved to %s", output_dir)
    return report
