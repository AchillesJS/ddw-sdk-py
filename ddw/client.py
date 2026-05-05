"""Thin HTTP client for DDW API.

Stateless except for httpx connection pool. One Client instance per process is fine.
All requests carry X-Tenant-Id header automatically.

This client is a Protocol-compatible target for Run, but Run can also accept
any object exposing `ingest_run(payload) -> dict[str, Any]`. This makes it
easy to inject a fake transport in tests.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

import httpx

from ddw.config import Config
from ddw.exceptions import DDWAPIError


class Client:
    """HTTP client for DDW API."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._http = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout_s,
            headers={"X-Tenant-Id": config.tenant_id},
        )

    @property
    def config(self) -> Config:
        return self._config

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """DEPRECATED in v0.2; kept for backward compatibility. Prefer ingest_run."""

        r = self._http.post("/api/v1/runs", json=payload)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = r.text
            raise DDWAPIError(r.status_code, detail)
        data: dict[str, Any] = r.json()
        return data

    def ingest_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST a RunReportV2 payload to /api/v1/runs/ingest."""

        r = self._http.post("/api/v1/runs/ingest", json=payload)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = r.text
            raise DDWAPIError(r.status_code, detail)
        data: dict[str, Any] = r.json()
        return data

    def health(self) -> bool:
        """Quick liveness check. Returns True if /health returns 200."""

        try:
            r = self._http.get("/health")
            return int(r.status_code) == 200
        except httpx.HTTPError:
            return False
