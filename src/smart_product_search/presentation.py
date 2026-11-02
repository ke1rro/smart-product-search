import html
import re

from smart_product_search.classification import ProductClassification
from smart_product_search.semantic import RankedProduct


def group_products(
    products: list[RankedProduct], predictions: list[ProductClassification]
) -> dict[str, list[RankedProduct]]:
    """Partition products by validated predicted labels, preserving current search order.

    Args:
        products: Unique metadata products in the selected retrieval mode's ranked order.
        predictions: Validated classifications for the analysed subset, without gold labels.

    Returns:
        E/S/C/I groups plus a separate unclassified group for products not analysed.
    """
    labels = {prediction.product_id: prediction.label for prediction in predictions}
    groups: dict[str, list[RankedProduct]] = {
        label: [] for label in ("E", "S", "C", "I", "unclassified")
    }
    for product in products:
        groups[labels.get(product.product_id, "unclassified")].append(product)
    return groups


def escape_product_text(value: str) -> str:
    """Escape untrusted listing or model text for literal, safe Markdown display.

    Args:
        value: Original plain text, including any literal HTML or Markdown characters.

    Returns:
        Escaped text retaining visible specifications without links, images or HTML execution.
    """
    return re.sub(r"([\\`*_{}\[\]()#+.!|~$\-])", r"\\\1", html.escape(value, quote=False))
