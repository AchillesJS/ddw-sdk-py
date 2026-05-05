from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from ddw import Client, Config, DDWAPIError


def _client_with_handler(handler: Callable[[httpx.Request], httpx.Response]) -> Client:
    cfg = Config(base_url="http://test", tenant_id="t1")
    c = Client(cfg)
    c._http.close()
    c._http = httpx.Client(
        base_url=cfg.base_url,
        timeout=cfg.timeout_s,
        headers={"X-Tenant-Id": cfg.tenant_id},
        transport=httpx.MockTransport(handler),
    )
    return c


def test_ingest_run_posts_to_correct_path() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        return httpx.Response(
            200, json={"run_id": "r1", "external_run_id": "x", "created": True}
        )

    c = _client_with_handler(handler)
    try:
        out = c.ingest_run({"external_run_id": "x"})
    finally:
        c.close()

    assert seen["url"] == "http://test/api/v1/runs/ingest"
    assert seen["method"] == "POST"
    assert out == {"run_id": "r1", "external_run_id": "x", "created": True}


def test_ingest_run_includes_tenant_header() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["tenant"] = request.headers.get("X-Tenant-Id")
        return httpx.Response(
            200, json={"run_id": "r1", "external_run_id": "x", "created": True}
        )

    c = _client_with_handler(handler)
    try:
        c.ingest_run({"external_run_id": "x"})
    finally:
        c.close()

    assert seen["tenant"] == "t1"


def test_ingest_run_raises_api_error_on_4xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": [{"loc": ["body", "x"]}]})

    c = _client_with_handler(handler)
    try:
        with pytest.raises(DDWAPIError) as ei:
            c.ingest_run({"x": 1})
    finally:
        c.close()

    assert ei.value.status_code == 422
    assert ei.value.detail == [{"loc": ["body", "x"]}]


def test_ingest_run_raises_api_error_on_5xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream broke")

    c = _client_with_handler(handler)
    try:
        with pytest.raises(DDWAPIError) as ei:
            c.ingest_run({})
    finally:
        c.close()

    assert ei.value.status_code == 500
    assert ei.value.detail in ("upstream broke", None)
