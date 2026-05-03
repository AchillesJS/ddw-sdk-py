"""DDW Python SDK — report ML training runs to Data Decision Workspace.

Public API:
    Run                 — context manager that captures one ML run
    Client              — low-level HTTP client (used by Run)
    Config              — SDK configuration
    DDWError            — base exception
    DDWAPIError         — API returned non-2xx
    DDWConfigError      — invalid/missing config
    DDWUsageError       — incorrect SDK usage

Quick start:

    import os
    os.environ["DDW_BASE_URL"] = "http://localhost:8000"
    os.environ["DDW_TENANT_ID"] = "team-vision"

    from ddw import Run

    with Run(
        model_tag="resnet50",
        dataset_size=10_000,
        budget_kind="epochs",
        budget_value=10,
        metric_name="accuracy",
    ) as run:
        # ... train ...
        run.log_metric_final(0.87)

That's it. The run is POSTed to DDW on context exit.
"""

from ddw._version import __version__
from ddw.client import Client
from ddw.config import Config
from ddw.exceptions import DDWAPIError, DDWConfigError, DDWError, DDWUsageError
from ddw.run import Run

__all__ = [
    "Run",
    "Client",
    "Config",
    "DDWError",
    "DDWAPIError",
    "DDWConfigError",
    "DDWUsageError",
    "__version__",
]
