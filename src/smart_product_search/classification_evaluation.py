import json
import logging
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import polars as pl

from smart_product_search.classification import classify_products
from smart_product_search.classification_prompt import PROMPT_VERSION, SYSTEM_PROMPT
from smart_product_search.config import SearchSettings
from smart_product_search.preparation import JUDGMENT_COLUMNS
from smart_product_search.retrieval import ProductResult, open_index
from smart_product_search.retrieval_evaluation import select_query_sample
from smart_product_search.source import DATASET_ID, DATASET_REVISION, calculate_sha256, write_json

LABELS = ("E", "S", "C", "I")
SELECTION_VERSION = "esci-systematic20-first10-v1"
logger = logging.getLogger(__name__)


def select_classification_examples(judgments: pl.DataFrame) -> pl.DataFrame:
    """Select fixed held-out query/product identities without accessing relevance labels.

    Args:
        judgments: Official test rows with query and product identity columns.

    Returns:
        Up to20 evenly spaced queries and first10 lexically ID-sorted products per query.
    """
    queries = select_query_sample(judgments, 20)
    return (
        judgments.select("query_id", "query", "product_id", "product_locale")
        .join(queries, on=["query_id", "query"], how="inner")
        .sort("query_id", "product_id", "product_locale")
        .group_by("query_id", maintain_order=True)
        .head(10)
    )


def classification_metrics(gold: list[str], predicted: list[str]) -> dict:
    """Compute pooled accuracy and fixed-four-class metrics without external ML tooling.

    Args:
        gold: Known held-out labels, consulted only after prediction.
        predicted: Valid E/S/C/I predictions paired with the same examples.

    Returns:
        Accuracy, macro-F1, class precision/recall/F1/support and true-row/predicted-column matrix.
    """
    if len(gold) != len(predicted) or set(gold + predicted) - set(LABELS):
        raise ValueError("Metrics require equally sized E/S/C/I label lists")
    matrix = [
        [
            sum(
                actual == row and prediction == column
                for actual, prediction in zip(gold, predicted, strict=True)
            )
            for column in LABELS
        ]
        for row in LABELS
    ]
    per_class = {}
    for index, label in enumerate(LABELS):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {
        "accuracy": sum(matrix[index][index] for index in range(4)) / len(gold) if gold else None,
        "macro_f1": sum(row["f1"] for row in per_class.values()) / 4 if gold else None,
        "per_class": per_class,
        "label_order": list(LABELS),
        "confusion_matrix": matrix,
    }


def summarize_request_latency(attempts: list[dict], successful_only: bool = False) -> dict:
    """Summarize measured request durations without treating cache replay as inference.

    Args:
        attempts: Actual HTTP attempt records with duration and validation outcome.
        successful_only: Exclude transport/provider failures and invalid model responses.

    Returns:
        Median and linearly interpolated p95 milliseconds plus contributing request count.
    """
    latencies = pl.Series(
        "latency_ms",
        [row["latency_ms"] for row in attempts if not successful_only or row["outcome"] == "valid"],
        dtype=pl.Float64,
    )
    return {
        "median": latencies.median(),
        "p95": latencies.quantile(0.95, interpolation="linear"),
        "requests": len(latencies),
    }


def evaluate_classification(
    settings: SearchSettings, output_dir: Path, allow_api: bool = False
) -> dict:
    """Evaluate one fixed model on judged test pairs, separately from live catalog retrieval.

    Verify source and index provenance, select identities without labels, classify one batch
    per query with no evaluation retries, then score using official gold. Stop live requests
    on service/credential errors. Replay successful cached batches without network by default.

    Args:
        settings: Central model/cache/storage settings, never serialized with credentials.
        output_dir: Ignored destination for selection, raw predictions and measured summaries.
        allow_api: Explicitly enable at most20 initial POST attempts; default is offline replay.

    Returns:
        Measured classification quality, supports, baseline, abstentions and API telemetry.
    """
    processed = settings.data_dir / "processed"
    manifest = json.loads((processed / "manifest.json").read_text())
    if (manifest["repo_id"], manifest["revision"]) != (DATASET_ID, DATASET_REVISION):
        raise ValueError("Use the approved complete English Small prepared dataset")
    hashes = {}
    for name in ("judgments_test.parquet", "judgments_train.parquet"):
        path = processed / name
        expected = next(row["sha256"] for row in manifest["files"] if row["name"] == name)
        hashes[name] = calculate_sha256(path)
        if hashes[name] != expected:
            raise ValueError(f"Prepared judgment checksum mismatch: {name}")
    judgments = pl.read_parquet(processed / "judgments_test.parquet")
    if judgments.columns != JUDGMENT_COLUMNS or judgments.height != 181701:
        raise ValueError("Use all181701 official test judgments")
    if judgments.filter(pl.col("split") != "test").height:
        raise ValueError("Classification evaluation uses test judgments only")
    selection = select_classification_examples(judgments)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selected_examples.json"
    write_json(
        selection_path, {"selection_version": SELECTION_VERSION, "examples": selection.to_dicts()}
    )
    configuration = replace(settings, llm_max_retries=0)
    outcomes: list[dict] = []
    attempts = []
    can_call_api = allow_api
    last_call_started = None
    with open_index(settings.index_path) as connection:
        metadata = json.loads(
            connection.execute(
                "SELECT value FROM index_metadata WHERE key=?", ("build",)
            ).fetchone()[0]
        )
        catalog = next(row for row in manifest["files"] if row["name"] == "catalog.parquet")
        if metadata["catalog_sha256"] != catalog["sha256"]:
            raise ValueError("Evaluation index/catalog provenance mismatch")
        for number, group in enumerate(
            selection.partition_by("query_id", maintain_order=True), start=1
        ):
            query_id, query = group.select("query_id", "query").row(0)
            candidates = []
            for product_id, locale in group.select("product_id", "product_locale").iter_rows():
                row = connection.execute(
                    "SELECT product_locale,product_id,product_title,product_brand,product_color,"
                    "product_bullet_point,product_description FROM products "
                    "WHERE product_id=? AND product_locale=?",
                    (product_id, locale),
                ).fetchone()
                if row is None:
                    raise ValueError(f"Judged product missing from catalog: {product_id}")
                candidates.append(ProductResult(*row, bm25_score=0.0))
            if can_call_api and last_call_started is not None:
                time.sleep(max(0, 3.1 - (time.perf_counter() - last_call_started)))
            started = time.perf_counter()
            prediction = classify_products(
                query, candidates, settings=configuration, allow_api=can_call_api
            )
            if prediction.attempts:
                last_call_started = started
            outcomes.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "product_ids": group["product_id"].to_list(),
                    **asdict(prediction),
                    "results": [result.model_dump() for result in prediction.results],
                }
            )
            attempts.extend(prediction.attempts)
            if prediction.status != "classified" and prediction.error_code not in {
                "invalid_response",
                "cache_miss",
            }:
                can_call_api = False
            logger.info(
                "Classification evaluation %s/%s: status=%s source=%s POSTs=%s",
                number,
                selection["query_id"].n_unique(),
                prediction.status,
                prediction.source,
                len(attempts),
            )
            write_json(output_dir / "outcomes.json", {"outcomes": outcomes})
    gold_rows = selection.join(
        judgments.select("query_id", "product_id", "esci_label"),
        on=["query_id", "product_id"],
        how="left",
    )
    gold_lookup = {
        (row["query_id"], row["product_id"]): row["esci_label"] for row in gold_rows.to_dicts()
    }
    pairs: list[dict] = [
        {
            "query_id": batch["query_id"],
            "product_id": prediction["product_id"],
            "gold": gold_lookup[(batch["query_id"], prediction["product_id"])],
            "predicted": prediction["label"],
            "reason": prediction["reason"],
        }
        for batch in outcomes
        for prediction in batch["results"]
        if batch["status"] == "classified"
    ]
    gold = [row["gold"] for row in pairs]
    predicted = [row["predicted"] for row in pairs]
    metrics = classification_metrics(gold, predicted)
    train = pl.read_parquet(processed / "judgments_train.parquet", columns=["esci_label"])
    majority = (
        train.group_by("esci_label")
        .len()
        .sort(["len", "esci_label"], descending=[True, False])
        .row(0)[0]
    )
    baseline = classification_metrics(gold, [majority] * len(gold))
    query_metrics = [
        classification_metrics(
            [row["gold"] for row in pairs if row["query_id"] == batch["query_id"]],
            [row["predicted"] for row in pairs if row["query_id"] == batch["query_id"]],
        )["macro_f1"]
        for batch in outcomes
        if batch["status"] == "classified"
    ]
    support = {label: gold_rows.filter(pl.col("esci_label") == label).height for label in LABELS}
    valid = sum(batch["status"] == "classified" for batch in outcomes)
    summary = {
        "utc_time": datetime.now(UTC).isoformat(),
        "model": settings.llm_model,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "selection_version": SELECTION_VERSION,
        "allow_api_requested": allow_api,
        "query_count": len(outcomes),
        "selected_examples": selection.height,
        "selected_support": support,
        "scored_examples": len(pairs),
        "classified_queries": valid,
        "unclassified_examples": selection.height - len(pairs),
        "metrics_on_valid_predictions": metrics,
        "failure_adjusted_accuracy": (
            sum(actual == prediction for actual, prediction in zip(gold, predicted, strict=True))
            / selection.height
            if pairs
            else None
        ),
        "mean_query_macro_f1": sum(query_metrics) / len(query_metrics) if query_metrics else None,
        "train_majority_baseline": {
            "label": majority,
            "metrics_on_same_scored_examples": baseline,
            "metrics_on_full_selection": classification_metrics(
                gold_rows["esci_label"].to_list(), [majority] * selection.height
            ),
        },
        "validated_batch_rate": valid / len(outcomes),
        "api_requests": len(attempts),
        "cache_batches": sum(batch["source"] == "cache" for batch in outcomes),
        "api_valid_response_rate": (
            sum(row["outcome"] == "valid" for row in attempts) / len(attempts) if attempts else None
        ),
        "api_failure_rate": (
            sum(row["outcome"] == "api_error" for row in attempts) / len(attempts)
            if attempts
            else None
        ),
        "api_invalid_response_rate": (
            sum(row["outcome"] == "invalid_response" for row in attempts) / len(attempts)
            if attempts
            else None
        ),
        "request_latency_ms": summarize_request_latency(attempts),
        "successful_request_latency_ms": summarize_request_latency(attempts, successful_only=True),
        "api_token_usage_current_run": {
            name: (
                sum(row.get("usage", {}).get(name, 0) for row in attempts)
                if any(name in row.get("usage", {}) for row in attempts)
                else None
            )
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "usage_reporting_requests": sum("usage" in row and bool(row["usage"]) for row in attempts),
        "settings": {
            key: value
            for key, value in asdict(settings).items()
            if key.startswith("llm_") and key != "llm_cache_dir"
        },
        "evaluation_retries": 0,
        "source_revision": DATASET_REVISION,
        "judgment_sha256": hashes,
        "selection_sha256": calculate_sha256(selection_path),
        "catalog_sha256": catalog["sha256"],
        "catalog_products": metadata["product_count"],
        "code_sha256": {
            name: calculate_sha256(Path(__file__).with_name(name))
            for name in (
                "classification.py",
                "classification_prompt.py",
                "classification_evaluation.py",
                "config.py",
            )
        },
        "versions": {name: version(name) for name in ("httpx2", "pydantic", "polars")},
        "protocol": (
            "Judged test-pair classification only; no retrieval/end-to-end quality claim. "
            "Fixed labels E/S/C/I; zero-denominator class scores0. Quality scores use valid "
            "predictions only; failures retain explicit abstention counts. "
            "Cache replay has no current API latency/usage."
        ),
    }
    write_json(output_dir / "predictions.json", {"pairs": pairs})
    write_json(output_dir / "summary.json", summary)
    if attempts:
        write_json(output_dir / "live-summary.json", summary)
        write_json(output_dir / "live-outcomes.json", {"outcomes": outcomes})
    return summary
