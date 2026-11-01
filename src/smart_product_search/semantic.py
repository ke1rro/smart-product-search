import hashlib
import logging
import unicodedata
from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import lru_cache
from threading import RLock

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from smart_product_search.config import SearchSettings
from smart_product_search.retrieval import ProductResult, build_fts_query

TEXT_VERSION = "listing-nfc-bounded-v1"
EMBEDDING_DIMENSION = 384
MODEL_LOAD_LOCK = RLock()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankedProduct(ProductResult):
    """Original serving metadata with separate lexical rank and optional cosine score.

    Attributes:
        bm25_rank: One-based rank in the original lexical candidate list.
        semantic_score: Normalized dot product, or None for the BM25-only mode.
    """

    bm25_rank: int
    semantic_score: float | None = None


def build_product_text(product: ProductResult, max_characters: int) -> str:
    """Construct bounded title-first text from real listing attributes only.

    Args:
        product: Metadata-only lexical candidate; scores and identity never enter text.
        max_characters: Total character budget before the model's wordpiece limit.

    Returns:
        NFC plain text with field caps256/96/64/512/512 for title/brand/color/bullets/description.
    """
    parts = []
    for value, limit in (
        (product.product_title, 256),
        (product.product_brand, 96),
        (product.product_color, 64),
        (product.product_bullet_point, 512),
        (product.product_description, 512),
    ):
        if value:
            parts.append(" ".join(unicodedata.normalize("NFC", value).split())[:limit])
    return " ".join(parts)[:max_characters]


class SemanticReranker:
    """One local inference model and a bounded, synchronized product embedding cache.

    Attributes:
        model: Loaded inference-only Sentence Transformer.
        embeddings: LRU vectors keyed by product identity and model/text compatibility hash.
        counters: Cumulative cache and encoding counts for verification.

    Args:
        model_name: Pretrained model repository identifier.
        revision: Pinned model commit SHA.
        device: Local execution device.
        batch_size: Maximum batch size per encoder call.
        max_seq_length: Maximum input wordpieces.
        cache_size: Maximum number of product vectors retained.
        text_chars: Maximum product text characters.
        threads: Number of CPU inference threads.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str,
        batch_size: int,
        max_seq_length: int,
        cache_size: int,
        text_chars: int,
        threads: int,
    ) -> None:
        torch.set_num_threads(threads)
        logger.info("Loading %s revision %s on %s", model_name, revision, device)
        self.model = SentenceTransformer(
            model_name, revision=revision, device=device, token=False, trust_remote_code=False
        )
        self.model.max_seq_length = max_seq_length
        self.model.eval()
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.text_chars = text_chars
        self.namespace = (model_name, revision, device, max_seq_length, text_chars)
        self.lock = RLock()
        self.embeddings: OrderedDict[tuple[str, str, str], np.ndarray] = OrderedDict()
        self.counters = {
            "cache_hits": 0,
            "cache_misses": 0,
            "products_encoded": 0,
            "queries_encoded": 0,
        }

    def cache_info(self) -> dict[str, int]:
        """Report current entry count and cumulative cache/inference counters.

        Returns:
            Cache entries, capacity, hits, misses, and encoded product/query counts.
        """
        with self.lock:
            return {"entries": len(self.embeddings), "capacity": self.cache_size, **self.counters}

    def rerank(
        self, query: str | None, candidates: list[ProductResult], k: int = 50
    ) -> list[RankedProduct]:
        """Rank lexical candidates by normalized cosine only, keeping original metadata.

        Reuse compatible cached products, encode the query and all missing product texts in
        batches, then order by descending cosine with original BM25 rank and identity ties.
        Labels and judged-pair associations are not inputs to this component.

        Args:
            query: Original natural-language query subject to the existing input contract.
            candidates: Unique metadata-only BM25 candidates in lexical order.
            k: Maximum output count, default50; evaluation may inspect the full candidate ordering.

        Returns:
            Up to k original products with separate BM25 rank/score and semantic score.
        """
        if type(k) is not int or k < 0:
            raise ValueError("Reranking limit must be a nonnegative integer")
        if not build_fts_query(query) or not candidates or k == 0:
            return []
        query_text = " ".join(unicodedata.normalize("NFC", query or "").split())
        texts = [build_product_text(product, self.text_chars) for product in candidates]
        keys = [
            (
                product.product_locale,
                product.product_id,
                hashlib.sha256(repr((self.namespace, TEXT_VERSION, text)).encode()).hexdigest(),
            )
            for product, text in zip(candidates, texts, strict=True)
        ]
        with self.lock:
            vectors_by_key = {}
            missing = []
            for key, text in zip(keys, texts, strict=True):
                if key in self.embeddings:
                    vectors_by_key[key] = self.embeddings[key]
                    self.embeddings.move_to_end(key)
                    self.counters["cache_hits"] += 1
                else:
                    missing.append((key, text))
            vectors = np.asarray(
                self.model.encode(
                    [query_text, *[text for _, text in missing]],
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            if (
                vectors.shape != (len(missing) + 1, EMBEDDING_DIMENSION)
                or not np.isfinite(vectors).all()
            ):
                raise ValueError("Encoder returned incompatible or nonfinite embeddings")
            if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
                raise ValueError("Encoder returned embeddings that are not normalized")
            self.counters["queries_encoded"] += 1
            self.counters["products_encoded"] += len(missing)
            self.counters["cache_misses"] += len(missing)
            for (key, _), vector in zip(missing, vectors[1:], strict=True):
                product_vector = np.array(vector, dtype=np.float32, copy=True)
                vectors_by_key[key] = product_vector
                self.embeddings[key] = product_vector
                if len(self.embeddings) > self.cache_size:
                    self.embeddings.popitem(last=False)
            similarities = np.clip(
                np.stack([vectors_by_key[key] for key in keys]) @ vectors[0], -1, 1
            )
        ordering = sorted(
            range(len(candidates)),
            key=lambda index: (
                -float(similarities[index]),
                index,
                candidates[index].product_locale,
                candidates[index].product_id,
            ),
        )
        return [
            RankedProduct(
                **asdict(candidates[index]),
                bm25_rank=index + 1,
                semantic_score=float(similarities[index]),
            )
            for index in ordering[:k]
        ]


@lru_cache(maxsize=1)
def load_reranker(model_options: tuple[str, str, str, int, int, int, int, int]) -> SemanticReranker:
    """Keep one current model configuration loaded across requests and application reruns.

    Args:
        model_options: Model ID/revision/device and batch/sequence/cache/text/thread options.

    Returns:
        Process-cached local reranker and its compatible product-vector cache.
    """
    return SemanticReranker(*model_options)


def get_reranker(settings: SearchSettings) -> SemanticReranker:
    """Serialize model initialization and reuse only relevant model configuration keys.

    Args:
        settings: Centralized settings; index paths and API secrets are excluded from cache keys.

    Returns:
        Process-shared reranker for the configured local model.
    """
    options = (
        settings.embedding_model,
        settings.embedding_revision,
        settings.embedding_device,
        settings.embedding_batch_size,
        settings.embedding_max_seq_length,
        settings.embedding_cache_size,
        settings.embedding_text_chars,
        settings.embedding_threads,
    )
    with MODEL_LOAD_LOCK:
        return load_reranker(options)
