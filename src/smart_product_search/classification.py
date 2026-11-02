import atexit
import hashlib
import json
import logging
import math
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Literal

import httpx2
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from smart_product_search.classification_prompt import PROMPT_VERSION, SYSTEM_PROMPT
from smart_product_search.config import SearchSettings, load_settings
from smart_product_search.retrieval import ProductResult, build_fts_query

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DIAGNOSTIC_HEADERS = {
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-request-id",
    "x-openrouter-request-id",
    "date",
}
CLASSIFICATION_LOCK = RLock()
logger = logging.getLogger(__name__)


class ProductClassification(BaseModel):
    """An unverified model prediction with one stable product ID and bounded evidence."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    product_id: str = Field(min_length=1, max_length=128)
    label: Literal["E", "S", "C", "I"]
    reason: str = Field(min_length=1, max_length=400)

    @field_validator("reason")
    @classmethod
    def require_visible_reason(cls, value: str) -> str:
        """Reject explanations containing only whitespace."""
        if not value.strip():
            raise ValueError("An explanation must contain visible text")
        return value


class ClassificationOutput(BaseModel):
    """Strict batched output before exact submitted-ID matching."""

    model_config = ConfigDict(extra="forbid", strict=True)
    results: list[ProductClassification] = Field(min_length=1, max_length=10)


@dataclass
class ClassificationBatch:
    """Validated predictions or an explicit unavailable state, with safe request telemetry.

    Attributes:
        status: classified or classification_unavailable, never a fabricated label.
        results: Predictions in original submitted product order.
        source: api, cache, or none.
        model: Requested model identifier.
        cache_key: Secret-free inference-input fingerprint.
        error_code: Stable failure category, if unavailable.
        error: Safe actionable message without provider bodies or exception text.
        attempts: Actual HTTP attempt status, latency, validation, and usage records.
        usage: Token counts for the successful originating response, including replay.
    """

    status: str = "classification_unavailable"
    results: list[ProductClassification] = field(default_factory=list)
    source: str = "none"
    model: str = ""
    cache_key: str = ""
    error_code: str | None = None
    error: str | None = None
    attempts: list[dict] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


def build_classification_request(
    query: str, products: list[ProductResult], settings: SearchSettings
) -> dict:
    """Construct one bounded JSON-mode request from original query and metadata only.

    Args:
        query: Original shopping query, retaining negations and specifications.
        products: One to ten unique catalog products without judgment fields.
        settings: Model identifier, inference parameters and metadata budget.

    Returns:
        Secret-free Chat Completions request body.
    """
    if not build_fts_query(query):
        raise ValueError("Provide a nonempty English product query")
    if not 1 <= len(products) <= 10:
        raise ValueError("Submit between 1 and 10 products")
    if len({product.product_id for product in products}) != len(products):
        raise ValueError("Submit unique product IDs")
    records = []
    for product in products:
        record = {"product_id": product.product_id, "product_locale": product.product_locale}
        remaining = settings.llm_product_text_chars
        for name, limit in (
            ("product_title", 512),
            ("product_brand", 128),
            ("product_color", 128),
            ("product_bullet_point", 768),
            ("product_description", 1024),
        ):
            value = getattr(product, name)
            if value and remaining:
                text = " ".join(unicodedata.normalize("NFC", value).split())[
                    : min(limit, remaining)
                ]
                record[name] = text
                remaining -= len(text)
        records.append(record)
    return {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({"query": query, "products": records}, ensure_ascii=False),
            },
        ],
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "response_format": {"type": "json_object"},
        "provider": {"require_parameters": True},
        "reasoning": {"enabled": False, "exclude": True},
        "stream": False,
    }


def parse_classifications(content: str, product_ids: list[str]) -> list[ProductClassification]:
    """Validate strict JSON predictions and map exactly once to each submitted ID.

    Args:
        content: Raw assistant JSON content, without repair or coercion.
        product_ids: Original ordered candidate IDs.

    Returns:
        One validated prediction per original ID in original order.
    """
    results = ClassificationOutput.model_validate_json(content).results
    ids = [result.product_id for result in results]
    if len(ids) != len(set(ids)) or set(ids) != set(product_ids):
        raise ValueError("Classification IDs must match submitted IDs exactly once")
    mapped = {result.product_id: result for result in results}
    return [mapped[product_id] for product_id in product_ids]


@lru_cache(maxsize=1)
def get_http_client() -> httpx2.Client:
    """Reuse one credential-free HTTP connection pool for explicit classification calls."""
    client = httpx2.Client(follow_redirects=False)
    atexit.register(client.close)
    return client


def retry_delay(response: httpx2.Response | None, attempt: int, settings: SearchSettings) -> float:
    """Respect Retry-After when present, otherwise use exponential bounded retry spacing.

    Args:
        response: Failed HTTP response, or None for transport failures.
        attempt: Zero-based attempt number.
        settings: Configured initial retry delay.

    Returns:
        Delay in seconds; values over30 prevent retry in this synchronous client.
    """
    value = response.headers.get("Retry-After") if response is not None else None
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                delay = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
            except ValueError, TypeError, OverflowError:
                delay = settings.llm_retry_backoff_seconds * 2**attempt
        if math.isfinite(delay):
            return max(0.0, delay)
    return settings.llm_retry_backoff_seconds * 2**attempt


def read_completion(
    response: httpx2.Response, product_ids: list[str], api_key: str = ""
) -> tuple[list[ProductClassification], dict]:
    """Validate the completion envelope before accepting its strict product predictions.

    Args:
        response: Successful HTTP response whose body remains untrusted.
        product_ids: Submitted product IDs used for exact output matching.
        api_key: Credential used only to redact reflected diagnostic values.

    Returns:
        Ordered predictions and safe metadata for cache/evaluation records.
    """
    envelope = response.json()
    if not isinstance(envelope, dict) or "error" in envelope:
        raise ValueError("Provider returned an error or invalid response envelope")
    choices = envelope.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Provider returned no complete choice")
    choice = choices[0]
    message = choice.get("message")
    if choice.get("finish_reason") != "stop" or not isinstance(message, dict):
        raise ValueError("Provider response was incomplete or interrupted")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip() or message.get("tool_calls"):
        raise ValueError("Provider returned empty or unexpected tool content")
    predictions = parse_classifications(content, product_ids)
    return predictions, completion_metadata(response, api_key)


def safe_diagnostic_text(value: object, api_key: str, max_characters: int = 1000) -> str | None:
    """Bound diagnostic text after redacting credentials and bearer tokens.

    Args:
        value: Untrusted provider text or numeric error code.
        api_key: Configured credential, never included in returned text.
        max_characters: Maximum retained text length after redaction.

    Returns:
        Single-line redacted text, or None for unsupported diagnostic field types.
    """
    if type(value) not in {str, int}:
        return None
    text = str(value)
    for secret in {api_key, api_key.strip()} - {""}:
        text = text.replace(secret, "[redacted]")
    text = re.sub(r"(?i)\bBearer\s+[^\s\"'<>,;]+", "Bearer [redacted]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", text)
    text = re.sub(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|authorization)[\"']?\s*[:=]\s*"
        r"(?:\"[^\"]*\"|'[^']*'|[^,;\s]+)",
        r"\1=[redacted]",
        text,
    )
    return " ".join(text.split())[:max_characters]


def diagnostic_headers(headers: dict, api_key: str) -> dict[str, str]:
    """Retain only allowlisted response diagnostics, excluding authentication and cookies.

    Args:
        headers: Actual response headers or a provider's nested rate-limit metadata headers.
        api_key: Credential used only for redaction.

    Returns:
        Lowercase safe header names and bounded redacted values.
    """
    return {
        name.lower(): text
        for name, value in headers.items()
        if isinstance(name, str) and name.lower() in DIAGNOSTIC_HEADERS
        if (text := safe_diagnostic_text(value, api_key, 200)) is not None
    }


def provider_error_metadata(error: dict, api_key: str) -> dict:
    """Capture structured provider error fields without retaining arbitrary raw payloads.

    Args:
        error: Untrusted error object from the provider response.
        api_key: Configured credential used only for redaction.

    Returns:
        Bounded code/message/type, provider name, safe headers and structured upstream error.
    """
    result: dict = {
        name: text
        for name in ("code", "message", "type")
        if (text := safe_diagnostic_text(error.get(name), api_key)) is not None
    }
    metadata = error.get("metadata")
    if not isinstance(metadata, dict):
        return result
    provider_name = safe_diagnostic_text(metadata.get("provider_name"), api_key, 200)
    if provider_name is not None:
        result["provider_name"] = provider_name
    headers = metadata.get("headers")
    if isinstance(headers, dict):
        result["metadata_headers"] = diagnostic_headers(headers, api_key)
    raw = metadata.get("raw")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = None
    if isinstance(raw, dict):
        upstream = raw.get("error", raw)
        if isinstance(upstream, dict):
            result["upstream_error"] = {
                name: text
                for name in ("code", "message", "type")
                if (text := safe_diagnostic_text(upstream.get(name), api_key)) is not None
            }
    return result


def completion_metadata(response: httpx2.Response, api_key: str = "") -> dict:
    """Extract safe headers, provider errors, identifiers and token counts, including failures.

    Args:
        response: Provider envelope; arbitrary bodies and headers are never retained.
        api_key: Credential used only to redact reflected diagnostics.

    Returns:
        Allowlisted redacted headers/error fields, identifiers and available token counts.
    """
    metadata = {"response_headers": diagnostic_headers(dict(response.headers), api_key)}
    try:
        envelope = response.json()
    except ValueError:
        return metadata
    if not isinstance(envelope, dict):
        return metadata
    usage = envelope.get("usage")
    tokens = (
        {
            name: value
            for name, value in usage.items()
            if name in {"prompt_tokens", "completion_tokens", "total_tokens"}
            and type(value) is int
            and value >= 0
        }
        if isinstance(usage, dict)
        else {}
    )
    metadata["usage"] = tokens
    for name in ("id", "model", "provider"):
        value = envelope.get(name)
        if isinstance(value, str):
            metadata[name] = safe_diagnostic_text(value, api_key, 200)
    error = envelope.get("error")
    if isinstance(error, dict):
        metadata["provider_error"] = provider_error_metadata(error, api_key)
    return metadata


def save_classification(path: Path, record: dict) -> None:
    """Atomically save a validated response without query secrets or HTTP headers.

    Args:
        path: Secret-free hash-named cache entry.
        record: Validated output and safe originating-response metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def classify_products(
    query: str,
    products: list[ProductResult],
    *,
    settings: SearchSettings | None = None,
    allow_api: bool = True,
) -> ClassificationBatch:
    """Classify one candidate batch explicitly, replaying only validated cached successes.

    Validate inputs, fingerprint the actual inference payload, check cache before credentials,
    and make at most one initial POST plus configured recoverable retries. Reject malformed
    or incomplete output without caching or labeling it. Retain safe attempt telemetry.

    Args:
        query: Original shopping query; never a dataset judgment.
        products: Up to10 unique products from search or the separate judged evaluation pool.
        settings: Central configuration; credentials are read only through its existing loader.
        allow_api: False requires deterministic offline cache replay without network access.

    Returns:
        Ordered predictions or explicit classification_unavailable, leaving inputs unchanged.
    """
    configuration = settings if settings is not None else load_settings()
    result = ClassificationBatch(model=configuration.llm_model)
    try:
        payload = build_classification_request(query, products, configuration)
    except ValueError as error:
        result.error_code, result.error = "invalid_input", str(error)
        return result
    fingerprint = json.dumps(
        [ENDPOINT, PROMPT_VERSION, payload], sort_keys=True, ensure_ascii=False
    )
    result.cache_key = hashlib.sha256(fingerprint.encode()).hexdigest()
    path = configuration.llm_cache_dir / f"{result.cache_key}.json"
    product_ids = [product.product_id for product in products]
    with CLASSIFICATION_LOCK:
        if path.is_file():
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(record, dict) or record.get("cache_key") != result.cache_key:
                    raise ValueError("Cache identity mismatch")
                predictions = parse_classifications(json.dumps(record.get("output")), product_ids)
                metadata = record.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise ValueError("Cache metadata must be an object")
                result.results, result.usage = predictions, metadata.get("usage", {})
                result.status, result.source = "classified", "cache"
                return result
            except OSError, ValueError, AttributeError, ValidationError:
                result.error_code, result.error = (
                    "cache_error",
                    "Cached classification is invalid; remove its cache entry and retry.",
                )
                return result
        if not allow_api:
            result.error_code, result.error = (
                "cache_miss",
                "No valid cached classification; offline replay made no API request.",
            )
            return result
        if not configuration.openrouter_api_key.strip() or not configuration.llm_model.strip():
            result.error_code, result.error = (
                "missing_api_key",
                "Configure OPENROUTER_API_KEY and an explicit model to classify; "
                "retrieval is available.",
            )
            return result
        for attempt in range(configuration.llm_max_retries + 1):
            started = time.perf_counter()
            response = None
            try:
                response = get_http_client().post(
                    ENDPOINT,
                    json=payload,
                    headers={"Authorization": f"Bearer {configuration.openrouter_api_key}"},
                    timeout=configuration.llm_timeout_seconds,
                )
                status = response.status_code
                error_code = f"http_{status}"
                can_retry = status in {408, 429} or 500 <= status < 600
            except httpx2.HTTPError as error:
                status = None
                error_code = (
                    "timeout" if isinstance(error, httpx2.TimeoutException) else "network_error"
                )
                can_retry = True
            telemetry = {
                "status_code": status,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "outcome": "api_error",
                "error_code": error_code,
            }
            result.attempts.append(telemetry)
            if response is not None:
                telemetry.update(completion_metadata(response, configuration.openrouter_api_key))
            if response is not None and response.is_success:
                try:
                    predictions, metadata = read_completion(
                        response, product_ids, configuration.openrouter_api_key
                    )
                except ValueError, ValidationError:
                    telemetry.update(outcome="invalid_response", error_code="invalid_response")
                    result.error_code, result.error = (
                        "invalid_response",
                        "OpenRouter returned invalid or incomplete classifications; "
                        "products remain unclassified.",
                    )
                    logger.warning(
                        "Classification rejected: model=%s status=invalid_response",
                        configuration.llm_model,
                    )
                    return result
                telemetry.update(outcome="valid", error_code=None, **metadata)
                result.results, result.usage = predictions, metadata["usage"]
                result.status, result.source = "classified", "api"
                result.error_code, result.error = None, None
                record = {
                    "cache_key": result.cache_key,
                    "prompt_version": PROMPT_VERSION,
                    "model": configuration.llm_model,
                    "output": {"results": [row.model_dump() for row in predictions]},
                    "metadata": metadata,
                    "utc_time": datetime.now(UTC).isoformat(),
                }
                try:
                    save_classification(path, record)
                except OSError:
                    logger.warning("Validated classification could not be persisted to its cache")
                logger.info(
                    "Classification validated: model=%s products=%s latency_ms=%.2f",
                    configuration.llm_model,
                    len(products),
                    telemetry["latency_ms"],
                )
                return result
            logger.warning(
                "Classification attempt failed: model=%s status=%s latency_ms=%.2f",
                configuration.llm_model,
                error_code,
                telemetry["latency_ms"],
            )
            result.error_code = error_code
            result.error = {
                "http_401": "OpenRouter authentication failed; check the configured API key.",
                "http_403": "OpenRouter denied access to this model.",
                "http_429": "OpenRouter rate limit reached; retry later.",
                "timeout": "OpenRouter request timed out; retrieval remains available.",
            }.get(
                error_code,
                "OpenRouter is unavailable for the configured model; products remain unclassified.",
            )
            if not can_retry or attempt == configuration.llm_max_retries:
                return result
            delay = retry_delay(response, attempt, configuration)
            if delay > 30:
                return result
            time.sleep(delay)
    return result
