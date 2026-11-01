import logging
import sqlite3
import sys

import streamlit as st
from huggingface_hub.utils import httpx

from smart_product_search.config import load_settings
from smart_product_search.retrieval import MAX_QUERY_CHARACTERS
from smart_product_search.search import search_products


def main() -> None:
    """Display local semantic search or the original lexical-only result ordering."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    st.set_page_config(page_title="Smart Product Search")
    st.title("Smart Product Search")
    st.caption("Local search over a historical product catalog. Results are unclassified.")
    settings = load_settings()
    if not settings.index_path.is_file():
        st.error("Search index missing. Run uv run python scripts/build_index.py first.")
        return
    with st.form("search"):
        query = st.text_input("English product search", max_chars=MAX_QUERY_CHARACTERS)
        mode = st.selectbox("Search mode", ["Semantic reranking", "BM25 only"])
        should_show_scores = st.checkbox("Show BM25 scores")
        is_submitted = st.form_submit_button("Search")
    if not is_submitted:
        return
    if not query.strip():
        st.info("Enter an English product query to search.")
        return
    try:
        with st.spinner("Searching the catalog..."):
            results = search_products(
                query,
                mode="semantic" if mode == "Semantic reranking" else "bm25",
                settings=settings,
            )
    except (ValueError, OSError, RuntimeError, sqlite3.DatabaseError, httpx.HTTPError) as error:
        st.error(f"Search unavailable: {error}")
        return
    if not results:
        st.info("No lexical matches found. Try a different product name or specification.")
        return
    st.caption(f"Showing the first {min(len(results), 20)} of {len(results)} candidates.")
    for rank, product in enumerate(results[:20], start=1):
        st.divider()
        st.text(f"{rank}. {product.product_title or 'Product title unavailable'}")
        st.text(f"Product ID: {product.product_id}")
        if product.product_brand:
            st.text(f"Brand: {product.product_brand}")
        if product.product_color:
            st.text(f"Color: {product.product_color}")
        if product.product_description:
            st.text(product.product_description[:600])
        else:
            st.caption("No description available.")
        if product.semantic_score is not None:
            st.caption(f"Cosine similarity: {product.semantic_score:.4f}")
        if should_show_scores:
            st.caption(f"BM25: {product.bm25_score:.8g} (lower ranks better)")


if __name__ == "__main__":
    main()
