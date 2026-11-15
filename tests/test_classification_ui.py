import json
from dataclasses import replace
from pathlib import Path

import httpx2
import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from smart_product_search import config
from smart_product_search.config import PROJECT_ROOT
from smart_product_search.preparation import CATALOG_COLUMNS
from smart_product_search.retrieval import build_index
from smart_product_search.search import search_products


@pytest.fixture
def app_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[httpx2.Request]:
    """Provide a tiny real index and record classification POSTs through mocked transport."""
    records = []
    for product_id, title in [("A", "wireless headphones"), ("B", "headphone carrying case")]:
        record = dict.fromkeys(CATALOG_COLUMNS)
        record.update(
            product_id=product_id,
            product_locale="us",
            product_title=title,
            has_title=True,
            index_text=title,
        )
        records.append(record)
    schema = {name: pl.Boolean if name == "has_title" else pl.String for name in CATALOG_COLUMNS}
    catalog = tmp_path / "catalog.parquet"
    pl.DataFrame(records, schema=schema).write_parquet(catalog)
    index = tmp_path / "products.sqlite3"
    build_index(catalog, index)
    settings = replace(
        config.load_settings(),
        index_path=index,
        llm_cache_dir=tmp_path / "cache",
        openrouter_api_key="dummy-test-key",
        llm_max_retries=0,
    )
    monkeypatch.setattr("smart_product_search.config.load_settings", lambda: settings)
    requests = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        """Predict for supplied IDs without accessing any gold labels."""
        requests.append(request)
        inputs = json.loads(json.loads(request.content)["messages"][1]["content"])
        outputs = [
            {
                "product_id": row["product_id"],
                "label": "E",
                "reason": "The listing names the requested type.",
            }
            for row in inputs["products"]
        ]
        return httpx2.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"results": outputs})},
                    }
                ]
            },
        )

    client = httpx2.Client(transport=httpx2.MockTransport(respond))
    monkeypatch.setattr("smart_product_search.classification.get_http_client", lambda: client)
    return requests


def test_classify_action_retains_results_and_never_auto_calls(app_configuration: list) -> None:
    """Classify only on a click, preserve searched query, and clear stale predictions on search."""
    app = AppTest.from_file(PROJECT_ROOT / "app/main.py").run()
    assert app.selectbox[0].value == "BM25"
    app.text_input[0].set_value("wireless headphones")
    app.button[0].click().run()
    assert not app.exception and app_configuration == []
    assert app.button[1].label == "Analyze with AI"
    app.button[1].click().run()
    assert not app.exception and len(app_configuration) == 1
    assert any("Exact Matches (E)" in element.value for element in app.subheader)
    ids = [element.value for element in app.text if "Product ID:" in element.value]
    app.run()
    assert len(app_configuration) == 1
    assert ids == [element.value for element in app.text if "Product ID:" in element.value]
    app.button[1].click().run()
    assert len(app_configuration) == 1
    assert any("cache" in element.value for element in app.caption)
    app.text_input[0].set_value("case")
    app.button[0].click().run()
    assert len(app_configuration) == 1
    assert not any("Exact Matches (E)" in element.value for element in app.subheader)


def test_failure_keeps_unclassified_search_results(
    app_configuration: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose classification_unavailable without dropping or relabeling the retrieved products."""
    client = httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(401)))
    monkeypatch.setattr("smart_product_search.classification.get_http_client", lambda: client)
    app = AppTest.from_file(PROJECT_ROOT / "app/main.py").run()
    app.text_input[0].set_value("headphones case")
    app.button[0].click().run()
    before = [element.value for element in app.text]
    app.button[1].click().run()
    assert not app.exception
    assert "AI analysis is unavailable" in app.warning[0].value
    assert [element.value for element in app.text] == before
    assert not any("Other Results (I)" in element.label for element in app.expander)


def test_bm25_is_default_without_altering_semantic_mode(app_configuration: list) -> None:
    """Retain lexical scores and never require a model or API for the new default."""
    settings = config.load_settings()
    products = search_products("headphones", settings=settings)
    assert [product.product_id for product in products] == ["A"]
    assert products == search_products("headphones", mode="bm25", settings=settings)
    assert app_configuration == []
