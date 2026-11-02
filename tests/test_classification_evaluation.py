import json
from dataclasses import replace
from pathlib import Path

import httpx2
import polars as pl
import pytest

from smart_product_search.classification import classify_products
from smart_product_search.classification_evaluation import (
    classification_metrics,
    evaluate_classification,
    select_classification_examples,
    summarize_request_latency,
)
from smart_product_search.config import load_settings
from smart_product_search.preparation import CATALOG_COLUMNS, JUDGMENT_COLUMNS
from smart_product_search.retrieval import ProductResult, build_index
from smart_product_search.source import DATASET_ID, DATASET_REVISION, calculate_sha256, write_json


def test_classification_metrics_hand_calculated() -> None:
    """Verify all four classes, direct confusion counts and zero-denominator behavior."""
    metrics = classification_metrics(["E", "E", "S", "C", "I"], ["E", "S", "S", "I", "I"])
    assert metrics["accuracy"] == 0.6
    assert metrics["macro_f1"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]]
    assert metrics["per_class"]["E"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "support": 2,
    }
    assert metrics["per_class"]["C"]["f1"] == 0
    assert classification_metrics([], [])["accuracy"] is None
    with pytest.raises(ValueError):
        classification_metrics(["E"], ["unknown"])


def test_subset_is_fixed_bounded_and_label_independent() -> None:
    """Select at most20 queries and10 ID-sorted judged products without consulting labels."""
    judgments = pl.DataFrame(
        [
            {
                "query_id": qid,
                "query": f"query {qid}",
                "product_id": f"P{pid:02d}",
                "product_locale": "us",
                "esci_label": "E",
            }
            for qid in range(25)
            for pid in range(12)
        ]
    )
    first = select_classification_examples(judgments)
    second = select_classification_examples(
        judgments.reverse().with_columns(pl.lit("I").alias("esci_label"))
    )
    assert first.equals(second)
    assert first.height == 200 and first["query_id"].n_unique() == 20
    assert first["query_id"].min() == 0 and first["query_id"].max() == 24
    assert "esci_label" not in first.columns
    assert first.filter(pl.col("query_id") == 0)["product_id"].to_list() == [
        f"P{pid:02d}" for pid in range(10)
    ]


def test_successful_offline_evaluation_serializes_and_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay validated successes through the complete evaluator with real files and no network."""
    processed = tmp_path / "processed"
    processed.mkdir()
    catalog_rows = []
    for product_id, title in [("A", "Headphones"), ("B", "Headphone case")]:
        record = dict.fromkeys(CATALOG_COLUMNS)
        record.update(
            product_id=product_id,
            product_locale="us",
            product_title=title,
            has_title=True,
            index_text=title,
        )
        catalog_rows.append(record)
    schema = {name: pl.Boolean if name == "has_title" else pl.String for name in CATALOG_COLUMNS}
    catalog_path = processed / "catalog.parquet"
    pl.DataFrame(catalog_rows, schema=schema).write_parquet(catalog_path)
    test = (
        pl.DataFrame({"example_id": range(181701)})
        .with_columns(
            (pl.col("example_id") // 2).alias("query_id"),
            pl.lit("headphones").alias("query"),
            pl.lit("us").alias("product_locale"),
            pl.when(pl.col("example_id") % 2 == 0)
            .then(pl.lit("A"))
            .otherwise(pl.lit("B"))
            .alias("product_id"),
            pl.when(pl.col("example_id") % 2 == 0)
            .then(pl.lit("E"))
            .otherwise(pl.lit("C"))
            .alias("esci_label"),
            pl.lit("test").alias("split"),
        )
        .select(JUDGMENT_COLUMNS)
    )
    test.write_parquet(processed / "judgments_test.parquet")
    pl.DataFrame({"esci_label": ["E", "E", "C"]}).write_parquet(
        processed / "judgments_train.parquet"
    )
    write_json(
        processed / "manifest.json",
        {
            "repo_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "files": [
                {"name": name, "sha256": calculate_sha256(processed / name)}
                for name in ("catalog.parquet", "judgments_test.parquet", "judgments_train.parquet")
            ],
        },
    )
    index = tmp_path / "index.sqlite3"
    build_index(catalog_path, index)
    settings = replace(
        load_settings(),
        data_dir=tmp_path,
        index_path=index,
        llm_cache_dir=tmp_path / "cache",
        openrouter_api_key="dummy-unit-key",
    )

    def respond(request: httpx2.Request) -> httpx2.Response:
        """Return model-shaped predictions for this synthetic cache fixture only."""
        products = json.loads(json.loads(request.content)["messages"][1]["content"])["products"]
        return httpx2.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "results": [
                                        {
                                            "product_id": row["product_id"],
                                            "label": "E",
                                            "reason": "The listing names headphones.",
                                        }
                                        for row in products
                                    ]
                                }
                            )
                        },
                    }
                ]
            },
        )

    with httpx2.Client(transport=httpx2.MockTransport(respond)) as client:
        monkeypatch.setattr("smart_product_search.classification.get_http_client", lambda: client)
        candidates = [
            ProductResult("us", "A", "Headphones", None, None, None, None, 0),
            ProductResult("us", "B", "Headphone case", None, None, None, None, 0),
        ]
        assert classify_products("headphones", candidates, settings=settings).status == "classified"
        assert (
            classify_products("headphones", candidates[:1], settings=settings).status
            == "classified"
        )
    summary = evaluate_classification(settings, tmp_path / "evaluation")
    assert summary["api_requests"] == 0 and summary["cache_batches"] == 20
    assert summary["scored_examples"] == summary["selected_examples"] == 39
    assert summary["metrics_on_valid_predictions"]["accuracy"] == pytest.approx(20 / 39)
    saved = json.loads((tmp_path / "evaluation/outcomes.json").read_text())["outcomes"]
    assert saved[0]["results"][0]["product_id"] == "A"
    assert saved[0]["source"] == "cache"


def test_success_latency_excludes_failed_requests() -> None:
    """Keep failed request durations out of successful inference latency statistics."""
    attempts = [
        {"outcome": "valid", "latency_ms": 100.0},
        {"outcome": "api_error", "latency_ms": 10000.0},
        {"outcome": "valid", "latency_ms": 300.0},
    ]
    successful = summarize_request_latency(attempts, successful_only=True)
    assert successful == {"median": 200.0, "p95": 290.0, "requests": 2}
    assert summarize_request_latency([], successful_only=True) == {
        "median": None,
        "p95": None,
        "requests": 0,
    }
