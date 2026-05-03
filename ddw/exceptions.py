"""DDW SDK exceptions.

Hierarchy:
    DDWError                — base for all SDK errors
    ├── DDWConfigError      — invalid/missing configuration
    ├── DDWAPIError         — API returned non-2xx response
    └── DDWUsageError       — incorrect SDK usage (e.g., metric not set)
"""

from __future__ import annotations

from typing import Any


class DDWError(Exception):
    """Base exception for ddw-sdk."""


class DDWConfigError(DDWError):
    """Raised when SDK config is invalid or incomplete."""


class DDWUsageError(DDWError):
    """Raised when SDK is used incorrectly (e.g. exiting Run without metric)."""


class DDWAPIError(DDWError):
    """Raised when DDW API returns an error response."""

    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(f"DDW API error {status_code}: {detail!r}")
        self.status_code = status_code
        self.detail = detail
