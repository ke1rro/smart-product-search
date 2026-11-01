import logging
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
        rerank_k: Maximum lexical candidates to rerank.
        classify_k: Maximum reranked candidates to classify.
        embedding_model: Pretrained embedding model identifier.
        llm_model: OpenRouter model identifier, empty until an endpoint is verified.
        openrouter_api_key: Optional API secret excluded from representations.
    """

    data_dir: Path
    index_path: Path
    retrieval_k: int
    rerank_k: int
    classify_k: int
    embedding_model: str
    llm_model: str
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
        "llm_model": "",
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
    settings = SearchSettings(
        data_dir=PROJECT_ROOT / options["data_dir"],
        index_path=PROJECT_ROOT / options["index_path"],
        retrieval_k=options["retrieval_k"],
        rerank_k=options["rerank_k"],
        classify_k=options["classify_k"],
        embedding_model=options["embedding_model"],
        llm_model=config("OPENROUTER_MODEL", default=options["llm_model"]),
        openrouter_api_key=config("OPENROUTER_API_KEY", default=""),
    )
    logger.debug("Loaded search configuration from %s", path)
    return settings
