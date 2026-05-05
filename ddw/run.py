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
        run.log_metric("val", "epoch", 10, eval_acc)

Behaviour on __exit__:
    - if no metric data is set:
        - if no exception in the with-block: raises DDWUsageError
        - if there was an exception: the exception propagates and no POST is sent
    - if metric data is set:
        - status is submitted as 'SUCCEEDED' or 'FAILED'
        - POST /api/v1/runs/ingest is called with RunReportV2
        - on HTTP error: re-raised by default; pass on_error="warn" to log instead
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from types import TracebackType
from typing import Any, Literal

from ddw.client import Client
from ddw.config import Config
from ddw.exceptions import DDWAPIError, DDWUsageError

_BudgetKind = Literal["steps", "epochs", "time_sec"]
_OnError = Literal["raise", "warn"]
_Status = Literal["succeeded", "failed", "running"]

_logger = logging.getLogger("ddw.sdk")


def _generate_external_run_id() -> str:
    return f"sdk-{uuid.uuid4().hex[:16]}"


def _budget_kind_to_x_kind(budget_kind: str) -> str:
    return {
        "steps": "step",
        "epochs": "epoch",
        "time_sec": "time",
    }[budget_kind]


def _require_non_empty_string(value: str, field_name: str) -> str:
    text = str(value)
    if not text.strip():
        raise DDWUsageError(f"{field_name} must be a non-empty string")
    return text


def _finite_float(value: float, field_name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise DDWUsageError(f"{field_name} must be a finite float, got {v!r}")
    return v


class Run:
    """Single ML training run captured and reported to DDW.

    Strict invariants enforced on construction:
      - metric_name non-empty, dataset_size > 0, budget_value > 0
      - budget_kind ∈ {steps, epochs, time_sec}

    Metric values must be normalized quality values in [0, 1]. Diagnostic
    values are unbounded but must be finite.
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
        external_run_id: str | None = None,
        notes: str | None = None,
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
        self._metric_points: list[dict[str, Any]] = []
        self._diagnostic_signals: list[dict[str, Any]] = []
        self._notes = notes
        self._external_run_id = external_run_id or _generate_external_run_id()
        self._last_x_metric: dict[tuple[str, str], float] = {}
        self._last_x_diag: dict[tuple[str, str], float] = {}
        self._on_error = on_error
        self._started_at: float | None = None
        self._submitted: dict[str, Any] | None = None

    # ---- public API ----

    def log_metric(
        self,
        series_name: str,
        x_kind: str,
        x_value: float,
        y_value: float,
    ) -> None:
        """Append one normalized metric point.

        y_value must be in [0, 1].
        x_value must be strictly increasing within (series_name, x_kind).
        Raises DDWUsageError on violation.
        Does not contact the backend.
        """

        point = self._build_point(
            series_name=series_name,
            x_kind=x_kind,
            x_value=x_value,
            y_value=y_value,
            bounded_y=True,
            last_x=self._last_x_metric,
        )
        self._metric_points.append(point)

    def log_diagnostic(
        self,
        series_name: str,
        x_kind: str,
        x_value: float,
        y_value: float,
    ) -> None:
        """Append one diagnostic signal point.

        y_value is unbounded but must be finite.
        x_value must be strictly increasing within (series_name, x_kind).
        Raises DDWUsageError on violation.
        Does not contact the backend.
        """

        point = self._build_point(
            series_name=series_name,
            x_kind=x_kind,
            x_value=x_value,
            y_value=y_value,
            bounded_y=False,
            last_x=self._last_x_diag,
        )
        self._diagnostic_signals.append(point)

    def log_metric_final(self, value: float) -> None:
        """Set the final metric value for this run.

        Must be in [0.0, 1.0]. Kept for v0.1 compatibility; on submit this
        becomes a single val metric point when no time-series metrics exist.
        """

        v = _finite_float(value, "metric_final")
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
        """Read-only view of the current run metadata (useful for tests/debugging)."""

        return dict(self._payload)

    @property
    def metric_points(self) -> list[dict[str, Any]]:
        """Read-only copy of buffered metric points."""

        return list(self._metric_points)

    @property
    def diagnostic_signals(self) -> list[dict[str, Any]]:
        """Read-only copy of buffered diagnostic signals."""

        return list(self._diagnostic_signals)

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
                self._payload["status"] = "failed"

            has_metric_final = self._payload["metric_final"] is not None
            if exc_type is not None and not self._metric_points and not has_metric_final:
                return

            try:
                report = self._build_run_report_v2(exc_type is not None)
            except DDWUsageError as e:
                self._handle_error(e)
                return

            try:
                self._submitted = self._client.ingest_run(report)
            except DDWAPIError as e:
                self._handle_error(e)
        finally:
            if self._client_owned:
                # Owned client: close it. Do not eat close errors.
                try:
                    self._client.close()
                except Exception:  # noqa: BLE001
                    _logger.debug("Closing DDW client raised; ignoring.")

    def _build_point(
        self,
        *,
        series_name: str,
        x_kind: str,
        x_value: float,
        y_value: float,
        bounded_y: bool,
        last_x: dict[tuple[str, str], float],
    ) -> dict[str, Any]:
        series = _require_non_empty_string(series_name, "series_name")
        kind = _require_non_empty_string(x_kind, "x_kind")
        x = _finite_float(x_value, "x_value")
        y = _finite_float(y_value, "y_value")
        if bounded_y and not (0.0 <= y <= 1.0):
            raise DDWUsageError(f"y_value must be within [0, 1], got {y}")

        key = (series, kind)
        previous = last_x.get(key)
        if previous is not None and x <= previous:
            raise DDWUsageError(
                "x_value must be strictly increasing within "
                f"(series_name, x_kind), got {x} after {previous}"
            )
        last_x[key] = x

        return {
            "series_name": series,
            "x_kind": kind,
            "x_value": x,
            "y_value": y,
        }

    def _build_run_report_v2(self, failed: bool) -> dict[str, Any]:
        metric_points = list(self._metric_points)
        metric_final = self._payload["metric_final"]

        if not metric_points:
            if metric_final is None:
                raise DDWUsageError(
                    "Run metric data was never set. Call run.log_metric(...) or "
                    "run.log_metric_final(value) before exiting context."
                )
            metric_points = [self._synthetic_metric_point(float(metric_final))]
        else:
            val_point = self._last_val_metric_point(metric_points)
            if val_point is not None:
                if metric_final is not None:
                    _logger.debug(
                        "Ignoring metric_final because metric_points are the source of truth."
                    )
                metric_final = val_point["y_value"]
            elif metric_final is None:
                raise DDWUsageError(
                    "RunReportV2 requires metric_final. Log at least one 'val' metric point "
                    "or call run.log_metric_final(value)."
                )

        final_value = _finite_float(float(metric_final), "metric_final")
        if not (0.0 <= final_value <= 1.0):
            raise DDWUsageError(f"metric_final must be within [0, 1], got {final_value}")

        status = "FAILED" if failed or self._payload["status"] == "failed" else "SUCCEEDED"
        self._payload["status"] = status.lower()
        self._payload["metric_final"] = final_value

        return {
            "schema_version": 2,
            "external_run_id": self._external_run_id,
            "run": {
                "model_tag": self._payload["model_tag"],
                "dataset_size": self._payload["dataset_size"],
                "budget_kind": self._payload["budget_kind"],
                "budget_value": self._payload["budget_value"],
                "metric_name": self._payload["metric_name"],
                "metric_final": final_value,
                "split": self._payload["split"],
                "dataset_version": self._payload["dataset_version"],
                "seed": self._payload["seed"],
                "config_hash": self._payload["config_hash"],
                "git_commit": self._payload["git_commit"],
                "status": status,
            },
            "metric_points": metric_points,
            "diagnostic_signals": list(self._diagnostic_signals),
            "artifacts": [],
            "notes": self._notes,
        }

    def _synthetic_metric_point(self, metric_final: float) -> dict[str, Any]:
        return {
            "series_name": "val",
            "x_kind": _budget_kind_to_x_kind(str(self._payload["budget_kind"])),
            "x_value": self._payload["budget_value"],
            "y_value": metric_final,
        }

    def _last_val_metric_point(self, metric_points: list[dict[str, Any]]) -> dict[str, Any] | None:
        val_points = [p for p in metric_points if p["series_name"] == "val"]
        if not val_points:
            return None
        return max(val_points, key=lambda p: p["x_value"])

    def _handle_error(self, e: Exception) -> None:
        if self._on_error == "raise":
            raise e
        _logger.warning("DDW run report failed: %s", e)
