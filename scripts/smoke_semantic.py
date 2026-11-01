import logging
import sys
import time

import numpy as np

from smart_product_search.config import PROJECT_ROOT, load_settings
from smart_product_search.retrieval import retrieve_products
from smart_product_search.search import search_products
from smart_product_search.semantic import get_reranker
from smart_product_search.source import write_json


def main() -> None:
    """Verify real local inference, candidate identity, and repeated-search cache reuse."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    settings = load_settings()
    started = time.perf_counter()
    reranker = get_reranker(settings)
    model_seconds = time.perf_counter() - started
    records = []
    for query in (
        "wireless noise cancelling headphones",
        "waterproof hiking backpack 35 liter",
        "WhiteCoat clipboard pink respiratory",
    ):
        candidates = retrieve_products(query, index_path=settings.index_path)
        before = reranker.cache_info()
        started = time.perf_counter()
        first = search_products(query, settings=settings)
        first_seconds = time.perf_counter() - started
        middle = reranker.cache_info()
        started = time.perf_counter()
        second = search_products(query, settings=settings)
        repeat_seconds = time.perf_counter() - started
        after = reranker.cache_info()
        first_ids = [product.product_id for product in first]
        second_ids = [product.product_id for product in second]
        if len(first) != min(50, len(candidates)) or len(set(first_ids)) != len(first_ids):
            raise RuntimeError("Real reranking returned missing or duplicate product identities")
        if not set(first_ids).issubset({product.product_id for product in candidates}):
            raise RuntimeError("Real reranking returned an unknown product")
        if first_ids != second_ids:
            raise RuntimeError("Identical repeated search changed result ordering")
        if after["products_encoded"] != middle["products_encoded"]:
            raise RuntimeError("Repeated search re-encoded unchanged product candidates")
        vectors = np.stack(list(reranker.embeddings.values()))
        if vectors.shape[1] != 384 or not np.allclose(
            np.linalg.norm(vectors, axis=1), 1, atol=1e-5
        ):
            raise RuntimeError("Real cached embeddings violated dimension/unit norm contracts")
        records.append(
            {
                "query": query,
                "returned": len(first),
                "product_ids": first_ids,
                "semantic_scores": [product.semantic_score for product in first],
                "first_seconds": first_seconds,
                "repeat_seconds": repeat_seconds,
                "products_encoded_first": middle["products_encoded"] - before["products_encoded"],
                "products_encoded_repeat": after["products_encoded"] - middle["products_encoded"],
                "cache_hits_repeat": after["cache_hits"] - middle["cache_hits"],
                "repeat_order_identical": True,
            }
        )
        logging.getLogger(__name__).info(
            "Smoke passed for %r; repeat=%s seconds", query, repeat_seconds
        )
    output = PROJECT_ROOT / ".artifacts/iteration-3/model-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output,
        {
            "device": str(reranker.model.device),
            "model_load_seconds": model_seconds,
            "queries": records,
            "cache": reranker.cache_info(),
        },
    )


if __name__ == "__main__":
    main()
