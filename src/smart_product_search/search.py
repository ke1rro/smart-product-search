from dataclasses import asdict

from smart_product_search.config import SearchSettings, load_settings
from smart_product_search.retrieval import retrieve_products
from smart_product_search.semantic import RankedProduct, get_reranker


def search_products(
    query: str | None, *, mode: str = "bm25", settings: SearchSettings | None = None
) -> list[RankedProduct]:
    """Use the existing lexical retrieval once, optionally reranking its candidates semantically.

    Args:
        query: Original English shopping text using the existing validation/empty-input contract.
        mode: bm25 by default, or semantic for optional local reranking.
        settings: Optional explicit configuration, otherwise read config/search.toml.

    Returns:
        BM25 top retrieval_k or semantically ordered top rerank_k, with separate scores.
    """
    if mode not in {"bm25", "semantic"}:
        raise ValueError("Search mode must be bm25 or semantic")
    configuration = settings if settings is not None else load_settings()
    candidates = retrieve_products(
        query, k=configuration.retrieval_k, index_path=configuration.index_path
    )
    if mode == "bm25":
        return [
            RankedProduct(**asdict(product), bm25_rank=rank)
            for rank, product in enumerate(candidates, start=1)
        ]
    if not candidates:
        return []
    return get_reranker(configuration).rerank(query, candidates, k=configuration.rerank_k)
