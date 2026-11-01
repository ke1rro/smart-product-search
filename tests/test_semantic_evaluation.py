import pytest

from smart_product_search.semantic_evaluation import judged_pool_ndcg


def test_ndcg_uses_judged_projection_and_direct_gains() -> None:
    """Exclude unknowns, retain judged irrelevant items, and score the known pool only."""
    gains = {"E": 3, "S": 2, "I": 0}
    assert judged_pool_ndcg(["unknown", "E", "S", "I"], gains) == 1.0
    assert judged_pool_ndcg(["unknown", "I", "S", "E"], gains) == pytest.approx(0.6480409555)
    assert judged_pool_ndcg(["unknown", "I"], {"I": 0}) is None
    assert judged_pool_ndcg([], {}) is None
