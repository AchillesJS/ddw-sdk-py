"""Run context manager: collect run metadata, POST on exit.

Typical usage in user training code:

    from ddw import Run

    with Run(
        model_tag="resnet50",
        dataset_size=10_000,
        budget_kind="epochs",
        budget_value=10,
        metric_name="accuracy",
    ) as run:
        model = train(...)
        run.log_metric_final(eval_acc)

Behaviour on __exit__:
    - if metric_final is not set:
        - if no exception in the with-block: raises DDWUsageError
        - if there was an exception: the exception propagates and no POST is sent
    - if metric_final is set:
        - status defaults to 'succeeded' (or 'failed' if exception happened)
        - POST /api/v1/runs is called
        - on HTTP error: re-raised by default; pass on_error="warn" to log instead
"""

from __future__ import annotations

import logging
import time
from types import TracebackType
from typing import Any, Literal

from ddw.client import Client
from ddw.config import Config
from ddw.exceptions import DDWAPIError, DDWUsageError

_BudgetKind = Literal["steps", "epochs", "time_sec"]
_OnError = Literal["raise", "warn"]
_Status = Literal["succeeded", "failed", "running"]

_logger = logging.getLogger("ddw.sdk")


class Run:
    """Single ML training run captured and reported to DDW.

    Strict invariants enforced on construction:
      - metric_name non-empty, dataset_size > 0, budget_value > 0
      - budget_kind ∈ {steps, epochs, time_sec}

    metric_final must be in [0, 1] when set (DDW backend constraint).
    """

    def __init__(
        self,
        *,
        model_tag: str,
        dataset_size: float,
        budget_kind: _BudgetKind,
        budget_value: float,
        metric_name: str,
        split: str | None = "val",
        dataset_version: str | None = None,
        seed: int | None = None,
        config_hash: str | None = None,
        git_commit: str | None = None,
        config: Config | None = None,
        client: Any = None,
        on_error: _OnError = "raise",
    ) -> None:
        if not model_tag or not str(model_tag).strip():
            raise DDWUsageError("model_tag must be a non-empty string")
        if not metric_name or not str(metric_name).strip():
            raise DDWUsageError("metric_name must be a non-empty string")
        if dataset_size <= 0:
            raise DDWUsageError("dataset_size must be > 0")
        if budget_value <= 0:
            raise DDWUsageError("budget_value must be > 0")
        if budget_kind not in ("steps", "epochs", "time_sec"):
            raise DDWUsageError(
                f"budget_kind must be one of steps/epochs/time_sec, got {budget_kind!r}"
            )
        if on_error not in ("raise", "warn"):
            raise DDWUsageError(f"on_error must be 'raise' or 'warn', got {on_error!r}")

        # Lazy config resolution: only if no client supplied.
        self._cfg: Config | None
        if client is None:
            self._cfg = config or Config.from_env_or_args()
            self._client: Any = Client(self._cfg)
            self._client_owned = True
        else:
            self._cfg = config  # may be None — that's fine, we don't use it
            self._client = client
            self._client_owned = False

        self._payload: dict[str, Any] = {
            "status": "running",
            "model_tag": model_tag,
            "dataset_size": float(dataset_size),
            "budget_kind": budget_kind,
            "budget_value": float(budget_value),
            "metric_name": metric_name,
            "split": split,
            "dataset_version": dataset_version,
            "seed": seed,
            "config_hash": config_hash,
            "git_commit": git_commit,
            "metric_final": None,
        }
        self._on_error = on_error
        self._started_at: float | None = None
        self._submitted: dict[str, Any] | None = None

    # ---- public API ----

    def log_metric_final(self, value: float) -> None:
        """Set the final metric value for this run.

        Must be in [0.0, 1.0]. Call before the with-block exits.
        """

        v = float(value)
        if not (0.0 <= v <= 1.0):
            raise DDWUsageError(f"metric_final must be within [0, 1], got {v}")
        self._payload["metric_final"] = v

    def set_status(self, status: _Status) -> None:
        """Override run status. Usually not needed — Run sets it automatically."""

        if status not in ("succeeded", "failed", "running"):
            raise DDWUsageError(f"status must be succeeded/failed/running, got {status!r}")
        self._payload["status"] = status

    @property
    def submitted(self) -> dict[str, Any] | None:
        """Return server response after successful POST, else None."""

        return self._submitted

    @property
    def payload(self) -> dict[str, Any]:
        """Read-only view of the current payload (useful for tests/debugging)."""

        return dict(self._payload)

    # ---- context manager ----

    def __enter__(self) -> Run:
        self._started_at = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if exc_type is not None:
                # Mark failed; user code raised. Do not swallow the exception.
                self._payload["status"] = "failed"

            if self._payload["metric_final"] is None:
                if exc_type is None:
                    # No metric, no exception — this is a usage error.
                    self._handle_error(
                        DDWUsageError(
                            "Run.metric_final was never set. "
                            "Call run.log_metric_final(value) before exiting context."
                        )
                    )
                # If there was an exception, do not POST a half-baked run.
                return

            if self._payload["status"] == "running":
                self._payload["status"] = "succeeded"

            try:
                self._submitted = self._client.create_run(self._payload)
            except DDWAPIError as e:
                self._handle_error(e)
        finally:
            if self._client_owned:
                # Owned client: close it. Do not eat close errors.
                try:
                    self._client.close()
                except Exception:  # noqa: BLE001
                    _logger.debug("Closing DDW client raised; ignoring.")

    def _handle_error(self, e: Exception) -> None:
        if self._on_error == "raise":
            raise e
        _logger.warning("DDW run report failed: %s", e)
