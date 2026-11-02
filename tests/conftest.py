import httpx2
import pytest


@pytest.fixture(autouse=True)
def block_external_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require MockTransport in normal tests so no credential or live endpoint is used."""

    def reject_network(self: httpx2.HTTPTransport, request: httpx2.Request) -> httpx2.Response:
        """Reject actual HTTP at the transport boundary, independently of caller behavior."""
        raise AssertionError("External HTTP is disabled in unit tests; use MockTransport")

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", reject_network)
