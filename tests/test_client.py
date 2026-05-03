from __future__ import annotations

import httpx
import pytest

from ddw import Client, Config, DDWAPIError


def _client_with_handler(handler: object) -> Client:
    """Build a Client whose internal httpx uses a MockTransport handler."""

    cfg = Config(base_url="http://test", tenant_id="t1")
    c = Client(cfg)
    # Replace the underlying httpx client with one wired to a mock transport.
    c._http.close()
    c._http = httpx.Client(
        base_url=cfg.base_url,
        timeout=cfg.timeout_s,
        headers={"X-Tenant-Id": cfg.tenant_id},
        transport=httpx.MockTransport(handler),
    )
    return c


def test_create_run_sends_tenant_header_and_returns_json() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["tenant"] = request.headers.get("X-Tenant-Id")
        seen["url"] = str(request.url)
        seen["method"] = request.method
        return httpx.Response(201, json={"id": "r1", "status": "succeeded"})

    c = _client_with_handler(handler)
    try:
        out = c.create_run({"model_tag": "m"})
    finally:
        c.close()

    assert seen["tenant"] == "t1"
    assert seen["url"] == "http://test/api/v1/runs"
    assert seen["method"] == "POST"
    assert out == {"id": "r1", "status": "succeeded"}


def test_create_run_raises_on_4xx_with_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": [{"loc": ["body", "x"]}]})

    c = _client_with_handler(handler)
    try:
        with pytest.raises(DDWAPIError) as ei:
            c.create_run({"x": 1})
    finally:
        c.close()

    assert ei.value.status_code == 422
    assert ei.value.detail == [{"loc": ["body", "x"]}]


def test_create_run_raises_on_5xx_with_text_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream broke")

    c = _client_with_handler(handler)
    try:
        with pytest.raises(DDWAPIError) as ei:
            c.create_run({})
    finally:
        c.close()

    assert ei.value.status_code == 500
    assert ei.value.detail in ("upstream broke", None)


def test_health_returns_true_on_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    c = _client_with_handler(handler)
    try:
        assert c.health() is True
    finally:
        c.close()


def test_health_returns_false_on_non_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    c = _client_with_handler(handler)
    try:
        assert c.health() is False
    finally:
        c.close()


def test_client_context_manager_closes() -> None:
    cfg = Config(base_url="http://test", tenant_id="t1")
    with Client(cfg) as c:
        assert c.config.tenant_id == "t1"
    # No assertion needed — just verifies __exit__ runs without error.
