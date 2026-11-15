import json
from dataclasses import replace
from pathlib import Path

import httpx2
import pytest
from streamlit.testing.v1 import AppTest

from smart_product_search.config import PROJECT_ROOT, load_settings
from smart_product_search.semantic import RankedProduct


@pytest.fixture
def ui_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Control backend inputs and HTTP while exercising real classification/cache/UI behavior."""
    index_path = tmp_path / "products.sqlite3"
    index_path.touch()
    state: dict = {
        "settings": replace(
            load_settings(),
            index_path=index_path,
            llm_cache_dir=tmp_path / "cache",
            openrouter_api_key="unit-only-key",
            llm_max_retries=0,
        ),
        "products": [
            RankedProduct(
                "us",
                f"P{index:02d}",
                f"Product {index:02d}",
                "Example brand",
                "Navy",
                None,
                f"Description {index:02d}",
                -1,
                index + 1,
            )
            for index in range(25)
        ],
        "searches": [],
        "requests": [],
        "status": 200,
        "force_label": None,
    }
    monkeypatch.setattr("smart_product_search.config.load_settings", lambda: state["settings"])

    def search(query: str, *, mode: str, settings: object) -> list[RankedProduct]:
        """Return controlled ranked products without loading a model in UI unit tests."""
        state["searches"].append((query, mode))
        if query == "no matches":
            return []
        if query == "broken search":
            raise RuntimeError("Index unavailable")
        return list(state["products"]) if mode == "bm25" else list(reversed(state["products"]))

    monkeypatch.setattr("smart_product_search.search.search_products", search)

    def respond(request: httpx2.Request) -> httpx2.Response:
        """Return synthetic label predictions with exact real request IDs, without gold data."""
        state["requests"].append(request)
        if state["status"] != 200:
            return httpx2.Response(state["status"])
        inputs = json.loads(json.loads(request.content)["messages"][1]["content"])
        results = [
            {
                "product_id": product["product_id"],
                "label": state["force_label"]
                or ("E", "S", "C", "I")[int(product["product_id"][1:]) % 4],
                "reason": f"Listing evidence for {product['product_id']}.",
            }
            for product in reversed(inputs["products"])
        ]
        return httpx2.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"results": results})},
                    }
                ]
            },
        )

    client = httpx2.Client(transport=httpx2.MockTransport(respond))
    monkeypatch.setattr("smart_product_search.classification.get_http_client", lambda: client)
    return state


def open_results(query: str = "headphones") -> AppTest:
    """Submit the explicit Search action through the actual application."""
    app = AppTest.from_file(PROJECT_ROOT / "app/main.py").run()
    app.text_input[0].set_value(query)
    app.button(key="run_search").click().run()
    assert not app.exception
    return app


def product_ids(app: AppTest) -> list[str]:
    """Read stable product identities from card details, including collapsed content."""
    return [
        element.value.removeprefix("Product ID: ")
        for element in app.text
        if element.value.startswith("Product ID: ")
    ]


def test_search_cards_pagination_and_no_automatic_inference(ui_backend: dict) -> None:
    """Render ranked products and expand the visible list without rerunning retrieval or AI."""
    app = open_results()
    assert app.selectbox[0].value == "BM25"
    assert ui_backend["searches"] == [("headphones", "bm25")]
    assert ui_backend["requests"] == []
    assert product_ids(app) == [f"P{index:02d}" for index in range(20)]
    assert app.button(key="analyze_products").label == "Analyze with AI"
    app.button(key="show_more_products").click().run()
    assert product_ids(app) == [f"P{index:02d}" for index in range(25)]
    app.checkbox[0].set_value(True).run()
    app.run()
    assert len(ui_backend["searches"]) == 1 and not ui_backend["requests"]
    assert any("BM25:" in element.value for element in app.caption)


def test_ai_groups_counts_order_and_unanalysed_products(ui_backend: dict) -> None:
    """Group only validated top10 predictions and keep remaining products distinct from I."""
    app = open_results()
    app.button(key="analyze_products").click().run()
    assert not app.exception and len(ui_backend["requests"]) == 1
    inputs = json.loads(json.loads(ui_backend["requests"][0].content)["messages"][1]["content"])
    assert inputs["query"] == "headphones"
    assert [product["product_id"] for product in inputs["products"]] == [
        f"P{index:02d}" for index in range(10)
    ]
    assert all(
        "esci_label" not in product and "split" not in product for product in inputs["products"]
    )
    assert [element.value for element in app.subheader][:3] == [
        "Exact Matches (E) · 3",
        "Alternatives (S) · 3",
        "Accessories & Complements (C) · 2",
    ]
    labels = [element.label for element in app.expander]
    assert "Other Results (I) · 2" in labels
    assert "More products, not analyzed · 15" in labels
    assert product_ids(app)[:10] == [
        "P00",
        "P04",
        "P08",
        "P01",
        "P05",
        "P09",
        "P02",
        "P06",
        "P03",
        "P07",
    ]
    assert len(set(product_ids(app))) == 25
    assert any("Listing evidence for P00" in element.value for element in app.markdown)
    assert any("predictions" in element.value.lower() for element in app.caption)


def test_empty_categories_are_honest(ui_backend: dict) -> None:
    """Show zero counts and useful empty states without fabricating exact or irrelevant items."""
    ui_backend["force_label"] = "S"
    app = open_results()
    app.button(key="analyze_products").click().run()
    assert not app.exception
    assert app.subheader[0].value == "Exact Matches (E) · 0"
    assert app.subheader[2].value == "Accessories & Complements (C) · 0"
    assert any("No exact matches were predicted" in element.value for element in app.info)
    assert "Other Results (I) · 0" in [element.label for element in app.expander]
    assert len(product_ids(app)) == 25


@pytest.mark.parametrize("change", ["query", "mode"])
def test_edits_invalidate_predictions_without_expensive_work(ui_backend: dict, change: str) -> None:
    """Clear stale AI display immediately while retaining explicitly identified previous results."""
    app = open_results()
    app.button(key="analyze_products").click().run()
    if change == "query":
        app.text_input[0].set_value("different headphones").run()
    else:
        app.selectbox[0].set_value("Semantic").run()
    assert not app.exception
    assert app.button(key="analyze_products").disabled
    assert not any("Exact Matches (E)" in element.value for element in app.subheader)
    assert any("Search to apply your changes" in element.value for element in app.info)
    assert len(ui_backend["requests"]) == len(ui_backend["searches"]) == 1
    assert product_ids(app) == [f"P{index:02d}" for index in range(20)]


def test_stale_analyze_click_cannot_send_new_query_with_old_results(ui_backend: dict) -> None:
    """Guard a stale browser button event in the same rerun that commits an edited query."""
    app = open_results()
    app.text_input[0].set_value("changed query")
    app.button(key="analyze_products").click().run()
    assert not app.exception and not ui_backend["requests"]
    assert app.button(key="analyze_products").disabled


def test_mode_submission_and_cache_replay_with_no_key(ui_backend: dict) -> None:
    """Pass the chosen mode through, then reuse successful predictions without another POST."""
    app = open_results()
    app.selectbox[0].set_value("Semantic").run()
    app.button(key="run_search").click().run()
    assert ui_backend["searches"][-1] == ("headphones", "semantic")
    assert product_ids(app)[0] == "P24"
    app.button(key="analyze_products").click().run()
    first = product_ids(app)
    ui_backend["settings"] = replace(ui_backend["settings"], openrouter_api_key="")
    app.button(key="analyze_products").click().run()
    assert not app.exception and not app.warning
    assert len(ui_backend["requests"]) == 1
    assert first == product_ids(app)
    assert any("Analysis source: cache" in element.value for element in app.caption)
    app.run()
    assert len(ui_backend["searches"]) == 2 and len(ui_backend["requests"]) == 1


def test_failure_preserves_cards_without_fabricated_groups(ui_backend: dict) -> None:
    """Keep search results visible without predictions or automatic rerun retries."""
    ui_backend["status"] = 429
    app = open_results()
    ids = product_ids(app)
    app.button(key="analyze_products").click().run()
    assert not app.exception
    assert "AI analysis is unavailable" in app.warning[0].value
    assert product_ids(app) == ids
    assert not any("Exact Matches" in element.value for element in app.subheader)
    app.run()
    assert len(ui_backend["requests"]) == 1


def test_missing_and_untrusted_metadata_are_rendered_safely(ui_backend: dict) -> None:
    """Keep optional fields absent and display listing markup as literal text."""
    ui_backend["products"][0] = replace(
        ui_backend["products"][0],
        product_title=None,
        product_brand=None,
        product_color=None,
        product_description=None,
    )
    ui_backend["products"][1] = replace(
        ui_backend["products"][1],
        product_title="![tracker](https://example.test) <script>bad</script>",
        product_description='<img src="https://example.test">',
    )
    app = open_results()
    assert any("Untitled product" in element.value for element in app.markdown)
    assert any("No description available" in element.value for element in app.caption)
    content = " ".join(element.value for element in app.markdown)
    assert "![tracker](" not in content and "<script>" not in content and "<img" not in content
    assert "&lt;script&gt;" in content


def test_empty_and_failed_search_states(ui_backend: dict) -> None:
    """Show actionable empty/error states without API calls or automatic recovery searches."""
    app = open_results("no matches")
    assert any("No products matched" in element.value for element in app.info)
    assert not product_ids(app)
    app.text_input[0].set_value("")
    app.button(key="run_search").click().run()
    assert any("Enter an English product query" in element.value for element in app.info)
    app.text_input[0].set_value("broken search")
    app.button(key="run_search").click().run()
    assert "Search unavailable" in app.error[0].value
    assert not app.exception and not ui_backend["requests"]
