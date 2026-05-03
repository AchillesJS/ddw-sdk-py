"""SDK configuration: explicit args > env vars > error.

Env vars used:
    DDW_BASE_URL    — base URL of the DDW API (e.g. http://localhost:8000)
    DDW_TENANT_ID   — tenant id, sent as X-Tenant-Id on every request
    DDW_TIMEOUT_S   — optional request timeout in seconds (default 15.0)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ddw.exceptions import DDWConfigError


@dataclass(frozen=True)
class Config:
    """Immutable SDK configuration.

    Use Config.from_env_or_args() to construct with env-fallback.
    Direct construction is fine for tests.
    """

    base_url: str
    tenant_id: str
    timeout_s: float = 15.0

    def __post_init__(self) -> None:
        if not self.base_url:
            raise DDWConfigError("base_url must be a non-empty string")
        if not self.tenant_id:
            raise DDWConfigError("tenant_id must be a non-empty string")
        if self.timeout_s <= 0:
            raise DDWConfigError("timeout_s must be > 0")

    @classmethod
    def from_env_or_args(
        cls,
        *,
        base_url: str | None = None,
        tenant_id: str | None = None,
        timeout_s: float | None = None,
    ) -> Config:
        """Resolve config: explicit args > env vars > raise.

        Raises DDWConfigError if base_url or tenant_id cannot be resolved.
        """

        b = base_url if base_url is not None else os.getenv("DDW_BASE_URL")
        t = tenant_id if tenant_id is not None else os.getenv("DDW_TENANT_ID")

        if timeout_s is not None:
            ts = float(timeout_s)
        else:
            ts_env = os.getenv("DDW_TIMEOUT_S")
            ts = float(ts_env) if ts_env else 15.0

        if not b:
            raise DDWConfigError(
                "DDW base_url is required. "
                "Pass base_url=... or set DDW_BASE_URL env var."
            )
        if not t:
            raise DDWConfigError(
                "DDW tenant_id is required. "
                "Pass tenant_id=... or set DDW_TENANT_ID env var."
            )

        return cls(base_url=b.rstrip("/"), tenant_id=t.strip(), timeout_s=ts)
