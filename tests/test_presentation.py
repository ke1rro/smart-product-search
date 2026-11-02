from typing import Literal

from smart_product_search.classification import ProductClassification
from smart_product_search.presentation import escape_product_text, group_products
from smart_product_search.semantic import RankedProduct


def product(product_id: str, rank: int) -> RankedProduct:
    """Build a metadata-only serving fixture in its current retrieval order."""
    return RankedProduct(
        "us", product_id, f"Product {product_id}", None, None, None, None, -rank, rank
    )


def test_groups_follow_predicted_labels_and_preserve_search_order() -> None:
    """Partition actual products once, preserving search order regardless of response order."""
    products = [product(product_id, rank) for rank, product_id in enumerate("ABCDEFG", start=1)]
    mapping: dict[str, Literal["E", "S", "C", "I"]] = {
        "A": "S",
        "B": "E",
        "C": "I",
        "D": "S",
        "E": "C",
        "F": "E",
    }
    predictions = [
        ProductClassification(product_id=product_id, label=label, reason="Fixture evidence.")
        for product_id, label in reversed(mapping.items())
    ]
    grouped = group_products(products, predictions)
    assert {label: [item.product_id for item in group] for label, group in grouped.items()} == {
        "E": ["B", "F"],
        "S": ["A", "D"],
        "C": ["E"],
        "I": ["C"],
        "unclassified": ["G"],
    }
    assert sum(len(group) for group in grouped.values()) == len(products)
    assert grouped["E"][0] is products[1]


def test_empty_categories_do_not_fabricate_products() -> None:
    """Keep every empty category empty and distinguish unanalysed products from irrelevant."""
    products = [product("A", 1), product("B", 2)]
    grouped = group_products(
        products, [ProductClassification(product_id="A", label="S", reason="Alternative.")]
    )
    assert grouped["E"] == grouped["C"] == grouped["I"] == []
    assert grouped["unclassified"] == [products[1]]
    assert all(not group for group in group_products([], []).values())


def test_listing_text_cannot_create_markdown_images_or_html() -> None:
    """Escape untrusted markup while retaining visible literal product specifications."""
    text = '<img src="https://tracker.example/x"> ![photo](https://tracker.example) *SALE*'
    escaped = escape_product_text(text)
    assert "<img" not in escaped
    assert "![photo](" not in escaped
    assert r"\!\[photo\]\(" in escaped
    assert r"\*SALE\*" in escaped
    assert "&lt;img" in escaped
    assert escape_product_text("USB-C & 35 L") == r"USB\-C &amp; 35 L"
    assert escape_product_text("&lt;USB-C&gt;") == r"&amp;lt;USB\-C&amp;gt;"


def test_math_delimiters_remain_literal() -> None:
    """Preserve dollar-delimited listing text instead of invoking Markdown math rendering."""
    assert escape_product_text("$x_1$ costs $5") == r"\$x\_1\$ costs \$5"
