import logging
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from decouple import config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchSettings:
    """Validated configuration for the planned product search pipeline.

    Attributes:
        data_dir: Absolute directory for downloaded and prepared data.
        index_path: Absolute path to the planned SQLite search index.
        retrieval_k: Maximum lexical candidate count.
        rerank_k: Maximum semantically reranked results returned.
        classify_k: Maximum candidates from the chosen mode to classify, at most10.
        embedding_model: Pretrained embedding model identifier.
        embedding_revision: Pinned model commit SHA.
        embedding_device: Local inference device, CPU by default.
        embedding_batch_size: Maximum encoding batch size.
        embedding_max_seq_length: Maximum model wordpiece sequence length.
        embedding_cache_size: Maximum on-demand product vectors held in memory.
        embedding_text_chars: Maximum bounded product text length.
        embedding_threads: Number of PyTorch CPU inference threads.
        llm_model: Explicit OpenRouter model identifier.
        llm_timeout_seconds: Maximum HTTP request timeout.
        llm_max_tokens: Maximum generated tokens per request.
        llm_temperature: Low sampling temperature.
        llm_max_retries: Maximum recoverable retries beyond the initial attempt.
        llm_retry_backoff_seconds: Initial exponential retry delay.
        llm_product_text_chars: Maximum metadata characters per product.
        llm_cache_dir: Directory for validated response replay files.
        openrouter_api_key: Optional API secret excluded from representations.
    """

    data_dir: Path
    index_path: Path
    retrieval_k: int
    rerank_k: int
    classify_k: int
    embedding_model: str
    embedding_revision: str
    embedding_device: str
    embedding_batch_size: int
    embedding_max_seq_length: int
    embedding_cache_size: int
    embedding_text_chars: int
    embedding_threads: int
    llm_model: str
    llm_timeout_seconds: int
    llm_max_tokens: int
    llm_temperature: float
    llm_max_retries: int
    llm_retry_backoff_seconds: float
    llm_product_text_chars: int
    llm_cache_dir: Path
    openrouter_api_key: str = field(repr=False)


def load_settings(path: Path = PROJECT_ROOT / "config/search.toml") -> SearchSettings:
    """Load and validate TOML settings with secrets read through decouple.

    Resolve relative storage paths from the repository root. Reject invalid types,
    unknown keys, and candidate limits outside classify <= rerank <= retrieval.

    Args:
        path: Path to the TOML configuration file.

    Returns:
        Validated search settings.
    """
    with path.open("rb") as configuration_file:
        options = tomllib.load(configuration_file)
    defaults = {
        "data_dir": "data",
        "index_path": "data/index/products.sqlite3",
        "retrieval_k": 100,
        "rerank_k": 50,
        "classify_k": 10,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "embedding_device": "cpu",
        "embedding_batch_size": 32,
        "embedding_max_seq_length": 256,
        "embedding_cache_size": 10000,
        "embedding_text_chars": 1200,
        "embedding_threads": 4,
        "llm_model": "nvidia/nemotron-3.5-super-vl:free",
        "llm_timeout_seconds": 60,
        "llm_max_tokens": 2048,
        "llm_temperature": 0.0,
        "llm_max_retries": 1,
        "llm_retry_backoff_seconds": 1.0,
        "llm_product_text_chars": 2400,
        "llm_cache_dir": ".artifacts/classification-cache",
    }
    if "openrouter_api_key" in options:
        raise ValueError("Use OPENROUTER_API_KEY in the environment or .env, not TOML")
    if options.keys() - defaults.keys():
        raise ValueError("Unknown search configuration keys")
    for name, default in defaults.items():
        value = options.setdefault(name, default)
        if type(value) is not type(default):
            raise ValueError(f"Invalid type for {name}")
        if isinstance(value, str) and name != "llm_model" and not value.strip():
            raise ValueError(f"{name} must not be blank")
    if not 0 < options["classify_k"] <= options["rerank_k"] <= options["retrieval_k"]:
        raise ValueError("Require 0 < classify_k <= rerank_k <= retrieval_k")
    for name in (
        "embedding_batch_size",
        "embedding_max_seq_length",
        "embedding_cache_size",
        "embedding_text_chars",
        "embedding_threads",
    ):
        if options[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if options["embedding_max_seq_length"] > 256:
        raise ValueError("MiniLM supports at most 256 wordpieces in this pipeline")
    if options["classify_k"] > 10:
        raise ValueError("Classify at most 10 products in one request")
    for name in ("llm_timeout_seconds", "llm_max_tokens", "llm_product_text_chars"):
        if options[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= options["llm_max_retries"] <= 2:
        raise ValueError("llm_max_retries must be between 0 and 2")
    if not 0 <= options["llm_temperature"] <= 0.3:
        raise ValueError("llm_temperature must be between 0 and 0.3")
    if not 0 < options["llm_retry_backoff_seconds"] <= 30:
        raise ValueError("llm_retry_backoff_seconds must be between 0 and 30")
    if not re.fullmatch(r"[a-f0-9]{40}", options["embedding_revision"]):
        raise ValueError("embedding_revision must be a pinned 40-character commit SHA")
    settings = SearchSettings(
        data_dir=PROJECT_ROOT / options["data_dir"],
        index_path=PROJECT_ROOT / options["index_path"],
        retrieval_k=options["retrieval_k"],
        rerank_k=options["rerank_k"],
        classify_k=options["classify_k"],
        embedding_model=options["embedding_model"],
        embedding_revision=options["embedding_revision"],
        embedding_device=options["embedding_device"],
        embedding_batch_size=options["embedding_batch_size"],
        embedding_max_seq_length=options["embedding_max_seq_length"],
        embedding_cache_size=options["embedding_cache_size"],
        embedding_text_chars=options["embedding_text_chars"],
        embedding_threads=options["embedding_threads"],
        llm_model=config("OPENROUTER_MODEL", default=options["llm_model"]),
        llm_timeout_seconds=options["llm_timeout_seconds"],
        llm_max_tokens=options["llm_max_tokens"],
        llm_temperature=options["llm_temperature"],
        llm_max_retries=options["llm_max_retries"],
        llm_retry_backoff_seconds=options["llm_retry_backoff_seconds"],
        llm_product_text_chars=options["llm_product_text_chars"],
        llm_cache_dir=PROJECT_ROOT / options["llm_cache_dir"],
        openrouter_api_key=config("OPENROUTER_API_KEY", default=""),
    )
    logger.debug("Loaded search configuration from %s", path)
    return settings
