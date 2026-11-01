import polars as pl

from smart_product_search.retrieval_evaluation import select_query_sample, summarize_coverage


def test_sample_is_systematic_and_label_independent() -> None:
    """Select fixed query identities across the query range without consulting gold labels."""
    judgments = pl.DataFrame(
        {"query_id": [5, 1, 4, 2, 3], "query": ["five", "one", "four", "two", "three"]}
    )
    assert select_query_sample(judgments, 3)["query_id"].to_list() == [1, 3, 5]
    assert select_query_sample(judgments.reverse(), 3).equals(select_query_sample(judgments, 3))
    assert select_query_sample(judgments, 50).height == 5


def test_coverage_denominators_do_not_treat_unjudged_results_as_negatives() -> None:
    """Distinguish query-averaged and pair-weighted coverage and exclude absent-class queries."""
    rows = pl.DataFrame(
        {
            "query_id": [1, 2, 1, 2],
            "label_group": ["E", "E", "S", "S"],
            "k": [10, 10, 10, 10],
            "positive_count": [2, 1, 0, 2],
            "hit_count": [1, 1, 0, 0],
        }
    )
    report = summarize_coverage(rows)
    exact = next(row for row in report if row["label_group"] == "E")
    substitute = next(row for row in report if row["label_group"] == "S")
    assert exact["eligible_queries"] == 2
    assert exact["judged_positive_pairs"] == 3
    assert exact["macro_coverage"] == 0.75
    assert exact["micro_coverage"] == 2 / 3
    assert substitute["eligible_queries"] == 1
    assert substitute["judged_positive_pairs"] == 2
    assert substitute["micro_coverage"] == 0
