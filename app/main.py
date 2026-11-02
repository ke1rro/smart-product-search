import logging
import sqlite3
import sys
import time

import streamlit as st
from huggingface_hub.utils import httpx

from smart_product_search.classification import ClassificationBatch, classify_products
from smart_product_search.config import load_settings
from smart_product_search.presentation import escape_product_text, group_products
from smart_product_search.retrieval import MAX_QUERY_CHARACTERS
from smart_product_search.search import search_products
from smart_product_search.semantic import RankedProduct

GROUP_COPY = {
    "E": (
        "Exact Matches",
        "Predicted to directly match your request.",
        "No exact matches were predicted. Explore the alternatives below.",
    ),
    "S": (
        "Alternatives",
        "Predicted to serve a similar purpose with some differences.",
        "No alternatives were predicted for these products.",
    ),
    "C": (
        "Accessories & Complements",
        "Predicted to be used with the item you requested.",
        "No accessories or complements were predicted.",
    ),
}


def render_product(
    product: RankedProduct, rank: int, reason: str | None, should_show_scores: bool
) -> None:
    """Display one real listing with literal metadata and an optional predicted explanation.

    Args:
        product: Original serving metadata and independent retrieval scores.
        rank: One-based position in the currently selected search order.
        reason: Validated model explanation, or None for products not analysed.
        should_show_scores: Reveal optional inspection scores without affecting ranking.
    """
    with st.container(border=True):
        st.caption(f"Result {rank}")
        st.markdown(f"#### {escape_product_text(product.product_title or 'Untitled product')}")
        attributes = [
            f"{label}: {value}"
            for label, value in (("Brand", product.product_brand), ("Color", product.product_color))
            if value
        ]
        if attributes:
            st.caption(escape_product_text(" · ".join(attributes)))
        excerpt = product.product_description or product.product_bullet_point
        if excerpt:
            st.markdown(escape_product_text(excerpt[:400]) + ("…" if len(excerpt) > 400 else ""))
        else:
            st.caption("No description available.")
        if reason is None:
            st.caption("Not analyzed with AI.")
        else:
            st.markdown(f"**AI explanation:** {escape_product_text(reason)}")
        with st.expander("Product details"):
            st.text(f"Product ID: {product.product_id}")
            if product.product_bullet_point:
                st.markdown(escape_product_text(product.product_bullet_point[:3000]))
            if product.product_description:
                st.markdown(escape_product_text(product.product_description[:3000]))
            if any(
                len(value) > 3000
                for value in (product.product_bullet_point, product.product_description)
                if value
            ):
                st.caption("Long listing text is shortened for readability.")
        if should_show_scores:
            st.caption(f"BM25: {product.bm25_score:.8g} (lower ranks better)")
            if product.semantic_score is not None:
                st.caption(f"Cosine similarity: {product.semantic_score:.4f}")


def render_cards(
    products: list[RankedProduct],
    ranks: dict[str, int],
    reasons: dict[str, str],
    should_show_scores: bool,
) -> None:
    """Render products in the supplied order without changing scores or classifications.

    Args:
        products: Ordered products in one section.
        ranks: Original search positions keyed by product ID.
        reasons: Validated predicted explanations keyed by product ID.
        should_show_scores: Display optional retrieval inspection data.
    """
    for product in products:
        render_product(
            product, ranks[product.product_id], reasons.get(product.product_id), should_show_scores
        )


def show_more_products() -> None:
    """Reveal another page of saved results without running search or inference."""
    st.session_state["visible_count"] = st.session_state.get("visible_count", 20) + 20


def render_ranked_list(
    products: list[RankedProduct], ranks: dict[str, int], should_show_scores: bool
) -> None:
    """Display a bounded page of unanalysed products with an explicit local pagination action.

    Args:
        products: Ranked results that have no AI category.
        ranks: Original search positions keyed by product ID.
        should_show_scores: Reveal optional inspection scores.
    """
    visible_count = st.session_state.get("visible_count", 20)
    render_cards(products[:visible_count], ranks, {}, should_show_scores)
    if len(products) > visible_count:
        st.button("Show more products", key="show_more_products", on_click=show_more_products)
    st.caption(f"Showing {min(visible_count, len(products))} of {len(products)} products.")


def render_groups(
    products: list[RankedProduct],
    classification: ClassificationBatch,
    ranks: dict[str, int],
    should_show_scores: bool,
) -> None:
    """Show only validated predictions in counted categories and keep other products separate.

    Args:
        products: Complete saved search results in their original order.
        classification: Successful validated top-product prediction batch.
        ranks: Original search positions keyed by product ID.
        should_show_scores: Reveal optional inspection scores.
    """
    groups = group_products(products, classification.results)
    reasons = {prediction.product_id: prediction.reason for prediction in classification.results}
    for label, (title, description, empty_message) in GROUP_COPY.items():
        st.subheader(f"{title} ({label}) · {len(groups[label])}")
        st.caption(description)
        if groups[label]:
            render_cards(groups[label], ranks, reasons, should_show_scores)
        else:
            st.info(empty_message)
    with st.expander(f"Other Results (I) · {len(groups['I'])}"):
        st.caption("Predicted to be less relevant to this request. Kept here for inspection.")
        if groups["I"]:
            render_cards(groups["I"], ranks, reasons, should_show_scores)
        else:
            st.caption("No other results were predicted.")
    if groups["unclassified"]:
        with st.expander(f"More products, not analyzed · {len(groups['unclassified'])}"):
            st.caption("These search results were not included in the AI analysis.")
            render_ranked_list(groups["unclassified"], ranks, should_show_scores)


def render_search_page() -> None:
    """Coordinate explicit search/analysis actions and retain their results across reruns.

    Read drafts without invoking expensive operations. Clear predictions when a draft differs
    from the saved search, then render the previous search with clear context. Only explicit
    buttons retrieve or classify, and classification is disabled until changed drafts are searched.
    Preserve ranking and real listing information when analysis is unavailable.
    """
    st.title("Smart Product Search")
    st.markdown("Find products, compare alternatives, and discover useful accessories.")
    st.caption(
        "A research prototype using historical Amazon ESCI listings, not an Amazon storefront."
    )
    settings = load_settings()
    if not settings.index_path.is_file():
        st.error("Product catalog unavailable. Run uv run python scripts/build_index.py first.")
        return
    query_column, mode_column, search_column = st.columns([6, 2, 1.3], vertical_alignment="bottom")
    with query_column:
        query = st.text_input(
            "What are you shopping for?",
            key="search_query",
            max_chars=MAX_QUERY_CHARACTERS,
            placeholder="e.g. wireless noise cancelling headphones",
        )
    with mode_column:
        mode = st.selectbox(
            "Search mode",
            ["BM25", "Semantic"],
            key="search_mode",
            help=(
                "BM25 matches words. Semantic reranks the same candidates by meaning "
                "and may take longer."
            ),
        )
    with search_column:
        is_submitted = st.button("Search", key="run_search", type="primary", width="stretch")
    if is_submitted:
        st.session_state.pop("current_search", None)
        st.session_state.pop("classification", None)
        st.session_state.pop("analysis_seconds", None)
        st.session_state["visible_count"] = 20
        if not query or not query.strip():
            st.info("Enter an English product query to search.")
            return
        started = time.perf_counter()
        try:
            with st.spinner("Finding products..."):
                results = search_products(
                    query, mode="semantic" if mode == "Semantic" else "bm25", settings=settings
                )
        except (ValueError, OSError, RuntimeError, sqlite3.DatabaseError, httpx.HTTPError) as error:
            st.error(f"Search unavailable: {error}")
            return
        st.session_state["current_search"] = {
            "query": query,
            "mode": mode,
            "results": results,
            "search_seconds": time.perf_counter() - started,
        }
    current = st.session_state.get("current_search")
    if current is None:
        st.info("Describe a product in English, then select Search to explore the catalog.")
        return
    has_pending_changes = query != current["query"] or mode != current["mode"]
    if has_pending_changes:
        st.session_state.pop("classification", None)
        st.session_state.pop("analysis_seconds", None)
        st.info(
            "Search to apply your changes. Previous results are shown below without AI analysis."
        )
    results = current["results"]
    if not results:
        st.info("No products matched this search. Try a broader product name or different wording.")
        return
    st.markdown(f"## Results for {escape_product_text(current['query'])}")
    st.caption(f"Top {len(results)} products · {current['mode']}")
    should_analyze = st.button(
        "Analyze with AI",
        key="analyze_products",
        disabled=has_pending_changes,
        help=(
            f"Compare the top {min(settings.classify_k, len(results))} products "
            "using AI predictions."
        ),
    )
    if should_analyze and not has_pending_changes:
        started = time.perf_counter()
        with st.spinner("Analyzing your top products..."):
            st.session_state["classification"] = classify_products(
                current["query"], results[: settings.classify_k], settings=settings
            )
        st.session_state["analysis_seconds"] = time.perf_counter() - started
    classification = st.session_state.get("classification")
    if classification is not None:
        if classification.status == "classified":
            st.caption(
                "AI categories are predictions, not verified matches. Check the listing details."
            )
            st.caption(f"Analysis source: {classification.source}")
        else:
            st.warning("AI analysis is unavailable. Your search results are still available.")
            st.caption(classification.error)
    with st.expander("Search details"):
        should_show_scores = st.checkbox("Show retrieval scores", key="show_retrieval_scores")
        if "search_seconds" in current:
            st.caption(f"Search time: {current['search_seconds']:.3f} s")
        if "analysis_seconds" in st.session_state:
            st.caption(f"Analysis time: {st.session_state['analysis_seconds']:.3f} s")
        if classification is not None:
            st.caption(f"Model: {classification.model}")
            st.caption(f"Status: {classification.status}")
    ranks = {product.product_id: rank for rank, product in enumerate(results, start=1)}
    if classification is not None and classification.status == "classified":
        render_groups(results, classification, ranks, should_show_scores)
    else:
        st.subheader("Search results")
        render_ranked_list(results, ranks, should_show_scores)


def main() -> None:
    """Display the responsive product search prototype using the existing pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    st.set_page_config(page_title="Smart Product Search", layout="wide")
    with st.container(horizontal=True, horizontal_alignment="center"):
        with st.container(width=1120):
            render_search_page()


if __name__ == "__main__":
    main()
