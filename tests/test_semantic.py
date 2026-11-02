from collections.abc import Generator
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from huggingface_hub.utils import httpx
from streamlit.testing.v1 import AppTest

from smart_product_search.config import PROJECT_ROOT, load_settings
from smart_product_search.preparation import CATALOG_COLUMNS
from smart_product_search.retrieval import ProductResult, build_index, retrieve_products
from smart_product_search.search import search_products
from smart_product_search.semantic import (
    build_product_text,
    get_reranker,
    load_reranker,
)


class ControlledEncoder:
    """Controlled unit vectors replacing only the external neural encoder in unit tests."""

    loads = 0

    def __init__(self, model_name: str, **options: object) -> None:
        ControlledEncoder.loads += 1
        self.max_seq_length = 256
        self.inputs: list[list[str]] = []

    def eval(self) -> None:
        """Accept the real model's inference-mode initialization."""

    def encode(self, texts: list[str], **options: object) -> np.ndarray:
        """Return controlled normalized vectors while retaining the actual texts submitted.

        Args:
            texts: Query followed by uncached product texts.
            options: Encoding parameters supplied by production code.

        Returns:
            Finite unit vectors with the real model's dimensionality.
        """
        self.inputs.append(texts)
        assert options["normalize_embeddings"] is True
        assert options["convert_to_numpy"] is True
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, 0 if "audio" in text.lower() else 1] = 1
        return vectors


@pytest.fixture(autouse=True)
def controlled_model(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Keep unit tests offline and isolate the process model cache between tests."""
    monkeypatch.setattr("smart_product_search.semantic.SentenceTransformer", ControlledEncoder)
    load_reranker.cache_clear()
    ControlledEncoder.loads = 0
    yield
    load_reranker.cache_clear()


def product(product_id: str, title: str | None, score: float = -1) -> ProductResult:
    """Create a metadata-only candidate without gold labels or query associations.

    Args:
        product_id: Stable fixture identity.
        title: Optional listing title.
        score: Original independent lexical score.

    Returns:
        A real serving-contract product fixture.
    """
    return ProductResult("us", product_id, title, None, None, None, None, score)


def test_cosine_order_identity_scores_and_ties() -> None:
    """Rank by cosine only while preserving metadata, lexical scores, and tied BM25 order."""
    reranker = get_reranker(load_settings())
    candidates = [product("Z", "garden", -50), product("B", "audio", -2), product("A", "audio", -1)]
    results = reranker.rerank("audio", candidates)
    assert [item.product_id for item in results] == ["B", "A", "Z"]
    assert [item.semantic_score for item in results] == [1, 1, 0]
    assert results[0].bm25_score == -2
    assert results[0].bm25_rank == 2
    assert results[0].product_title == "audio"


def test_cache_reuse_content_invalidation_and_model_reuse() -> None:
    """Load one encoder and reuse products until their represented content changes."""
    settings = load_settings()
    reranker = get_reranker(settings)
    candidates = [product("A", "audio"), product("B", "garden")]
    first = reranker.rerank("audio", candidates)
    assert get_reranker(settings) is reranker
    assert ControlledEncoder.loads == 1
    assert reranker.cache_info()["products_encoded"] == 2
    assert reranker.rerank("audio", candidates) == first
    assert reranker.cache_info()["products_encoded"] == 2
    assert reranker.cache_info()["cache_hits"] == 2
    reranker.rerank("audio", [replace(candidates[0], product_title="changed")])
    assert reranker.cache_info()["products_encoded"] == 3
    get_reranker(replace(settings, embedding_revision="a" * 40))
    assert ControlledEncoder.loads == 2


def test_bounded_cache_and_preprocessing_invalidation() -> None:
    """Bound on-demand vectors and invalidate changed text-budget/model configurations."""
    settings = replace(load_settings(), embedding_cache_size=1)
    reranker = get_reranker(settings)
    reranker.rerank("audio", [product("A", "audio"), product("B", "garden")])
    assert reranker.cache_info()["entries"] == 1
    reranker.rerank("audio", [product("A", "audio")])
    assert reranker.cache_info()["products_encoded"] == 3
    assert get_reranker(replace(settings, embedding_text_chars=20)) is not reranker


def test_missing_metadata_and_bounded_text() -> None:
    """Keep empty fields empty and prevent descriptions from crowding out identifying text."""
    empty = product("A", None)
    assert build_product_text(empty, 1200) == ""
    full = replace(product("B", "Audio X-123 without wheels"), product_description="long " * 2000)
    text = build_product_text(full, 1200)
    assert text.startswith("Audio X-123 without wheels")
    assert len(text) <= 1200
    results = get_reranker(load_settings()).rerank("audio", [empty, full])
    assert len(results) == 2
    assert results[1].product_title is None
    assert results[1].product_description is None


def test_default_top50_and_fewer_candidates() -> None:
    """Return the default top50 without unknown or duplicate IDs and accept small pools."""
    reranker = get_reranker(load_settings())
    candidates = [product(str(index), "audio") for index in range(100)]
    results = reranker.rerank("audio", candidates)
    assert len(results) == 50
    assert [item.product_id for item in results] == [str(index) for index in range(50)]
    assert len(reranker.rerank("audio", candidates[:3])) == 3
    assert reranker.rerank("", candidates) == []
    assert reranker.rerank("audio", []) == []


def test_only_serving_fields_are_encoded() -> None:
    """Exclude product IDs and lexical scores from embedding text and cache identity errors."""
    candidate = product("SECRET_IDENTIFIER", "audio", -999)
    text = build_product_text(candidate, 1200)
    assert text == "audio"
    assert "SECRET_IDENTIFIER" not in text
    reranker = get_reranker(load_settings())
    assert reranker.rerank("audio", [candidate])[0].product_id == "SECRET_IDENTIFIER"


def test_preprocessing_revision_invalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalidate vector entries when the relevant representation version changes."""
    reranker = get_reranker(load_settings())
    candidates = [product("A", "audio")]
    reranker.rerank("audio", candidates)
    monkeypatch.setattr("smart_product_search.semantic.TEXT_VERSION", "next-representation")
    reranker.rerank("audio", candidates)
    assert reranker.cache_info()["products_encoded"] == 2


@pytest.mark.parametrize("invalid", ["shape", "nonfinite", "unnormalized"])
def test_reject_incompatible_model_vectors(invalid: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject incompatible external model output instead of serving invalid ranks."""
    reranker = get_reranker(load_settings())

    def encode(texts: list[str], **options: object) -> np.ndarray:
        vectors = np.zeros((len(texts), 384))
        if invalid == "shape":
            return vectors[:, :2]
        if invalid == "nonfinite":
            vectors[:] = np.nan
        return vectors

    monkeypatch.setattr(reranker.model, "encode", encode)
    with pytest.raises(ValueError):
        reranker.rerank("audio", [product("A", "audio")])
    assert reranker.cache_info()["products_encoded"] == 0


@pytest.fixture
def semantic_index(tmp_path: Path) -> Path:
    """Build an actual small FTS index to test the unified search and UI contracts."""
    rows = []
    for product_id, title in [("A", "Audio headphones"), ("B", "Garden shovel")]:
        row = {name: None for name in CATALOG_COLUMNS}
        row.update(
            product_locale="us",
            product_id=product_id,
            product_title=title,
            product_title_original=title,
            has_title=True,
            index_text=title,
        )
        rows.append(row)
    schema = {name: pl.Boolean if name == "has_title" else pl.String for name in CATALOG_COLUMNS}
    catalog = tmp_path / "catalog.parquet"
    pl.DataFrame(rows, schema=schema).write_parquet(catalog)
    index = tmp_path / "products.sqlite3"
    build_index(catalog, index)
    return index


def test_search_modes_preserve_bm25_and_do_not_duplicate_retrieval(semantic_index: Path) -> None:
    """Keep the original lexical ordering/scores while exposing pure semantic mode."""
    settings = replace(load_settings(), index_path=semantic_index)
    original = retrieve_products("audio garden", index_path=semantic_index)
    lexical = search_products("audio garden", mode="bm25", settings=settings)
    assert [item.product_id for item in lexical] == [item.product_id for item in original]
    assert [item.bm25_score for item in lexical] == [item.bm25_score for item in original]
    assert all(item.semantic_score is None for item in lexical)
    assert ControlledEncoder.loads == 0
    semantic = search_products("audio garden", mode="semantic", settings=settings)
    assert [item.product_id for item in semantic] == ["A", "B"]
    assert [item.semantic_score for item in semantic] == [1, 0]
    assert {item.product_id for item in semantic} == {item.product_id for item in original}


def test_empty_search_does_not_initialize_model(tmp_path: Path) -> None:
    """Preserve empty/no-hit contracts and avoid loading a model unnecessarily."""
    settings = replace(load_settings(), index_path=tmp_path / "missing.sqlite3")
    assert search_products(None, settings=settings) == []
    assert search_products("!!!", settings=settings) == []
    assert ControlledEncoder.loads == 0
    with pytest.raises(ValueError):
        search_products("x" * 1001, settings=settings)


def test_semantic_ui_mode_reuses_model(
    semantic_index: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep semantic results and reuse one model/cache across Streamlit reruns."""
    settings = replace(load_settings(), index_path=semantic_index)
    monkeypatch.setattr("smart_product_search.config.load_settings", lambda: settings)
    app = AppTest.from_file(PROJECT_ROOT / "app/main.py").run()
    assert app.selectbox[0].value == "BM25"
    app.selectbox[0].set_value("Semantic")
    app.text_input[0].set_value("audio garden")
    app.button[0].click().run()
    assert len(app.exception) == 0
    app.checkbox[0].set_value(True).run()
    assert any("Cosine similarity" in element.value for element in app.caption)
    app.button[0].click().run()
    assert ControlledEncoder.loads == 1
    assert get_reranker(settings).cache_info()["products_encoded"] == 2
    app.selectbox[0].set_value("BM25")
    app.button[0].click().run()
    assert len(app.exception) == 0
    assert ControlledEncoder.loads == 1


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
def test_semantic_ui_surfaces_model_transport_failure(
    semantic_index: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[httpx.HTTPError]
) -> None:
    """Display model-download transport failures and retain explicit BM25 search."""
    settings = replace(load_settings(), index_path=semantic_index)
    monkeypatch.setattr("smart_product_search.config.load_settings", lambda: settings)

    def fail_loading(settings: object) -> None:
        """Simulate a Hub transport failure after the loader exhausts retries."""
        raise error_type("Model download interrupted")

    monkeypatch.setattr("smart_product_search.search.get_reranker", fail_loading)
    app = AppTest.from_file(PROJECT_ROOT / "app/main.py").run()
    app.selectbox[0].set_value("Semantic")
    app.text_input[0].set_value("audio garden")
    app.button[0].click().run()
    assert not app.exception
    assert "Model download interrupted" in app.error[0].value
    app.selectbox[0].set_value("BM25")
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert any("Product ID:" in element.value for element in app.text)
