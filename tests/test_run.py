from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from ddw import Config, DDWAPIError, DDWUsageError, Run


def _ok_client(response: dict[str, Any] | None = None) -> MagicMock:
    """A fake client matching the Client.ingest_run interface."""

    c = MagicMock()
    c.ingest_run.return_value = response or {
        "run_id": "run-uuid",
        "external_run_id": "sdk-test",
        "created": True,
    }
    return c


def _make_run(client: Any, cfg: Config, **overrides: Any) -> Run:
    defaults: dict[str, Any] = dict(
        model_tag="m1",
        dataset_size=1000,
        budget_kind="epochs",
        budget_value=10,
        metric_name="accuracy",
        config=cfg,
        client=client,
        external_run_id="sdk-test",
    )
    defaults.update(overrides)
    return Run(**defaults)


# ---------------- happy path ----------------


def test_happy_path_posts_run(cfg: Config) -> None:
    client = _ok_client()
    with _make_run(client, cfg) as run:
        run.log_metric_final(0.87)

    client.ingest_run.assert_called_once()
    payload = client.ingest_run.call_args.args[0]

    assert payload["schema_version"] == 2
    assert payload["external_run_id"] == "sdk-test"
    assert payload["run"]["model_tag"] == "m1"
    assert payload["run"]["dataset_size"] == 1000.0
    assert payload["run"]["metric_final"] == 0.87
    assert payload["run"]["status"] == "SUCCEEDED"
    assert payload["run"]["split"] == "val"
    assert payload["run"]["budget_kind"] == "epochs"
    assert payload["metric_points"] == [
        {"series_name": "val", "x_kind": "epoch", "x_value": 10.0, "y_value": 0.87}
    ]
    assert run.submitted == {"run_id": "run-uuid", "external_run_id": "sdk-test", "created": True}


def test_payload_carries_all_optional_metadata(cfg: Config) -> None:
    client = _ok_client()
    with _make_run(
        client,
        cfg,
        seed=42,
        dataset_version="v3",
        config_hash="abc123",
        git_commit="deadbeef",
        split="test",
    ) as run:
        run.log_metric_final(0.5)

    p = client.ingest_run.call_args.args[0]["run"]
    assert p["seed"] == 42
    assert p["dataset_version"] == "v3"
    assert p["config_hash"] == "abc123"
    assert p["git_commit"] == "deadbeef"
    assert p["split"] == "test"


def test_payload_property_is_readonly_view(cfg: Config) -> None:
    client = _ok_client()
    run = _make_run(client, cfg)
    p1 = run.payload
    p1["model_tag"] = "TAMPERED"
    assert run.payload["model_tag"] == "m1"


# ---------------- exception handling ----------------


def test_exception_in_block_marks_failed_and_posts_if_metric_exists(cfg: Config) -> None:
    client = _ok_client()

    with pytest.raises(RuntimeError, match="boom"):
        with _make_run(client, cfg) as run:
            run.log_metric_final(0.5)
            raise RuntimeError("boom")

    client.ingest_run.assert_called_once()
    p = client.ingest_run.call_args.args[0]
    assert p["run"]["status"] == "FAILED"
    assert p["run"]["metric_final"] == 0.5


def test_exception_without_metric_skips_post(cfg: Config) -> None:
    client = _ok_client()

    with pytest.raises(RuntimeError):
        with _make_run(client, cfg):
            raise RuntimeError("crashed before metric")

    client.ingest_run.assert_not_called()


def test_no_metric_no_exception_raises_usage(cfg: Config) -> None:
    client = _ok_client()
    with pytest.raises(DDWUsageError, match="Run metric data was never set"):
        with _make_run(client, cfg):
            pass

    client.ingest_run.assert_not_called()


# ---------------- validation ----------------


def test_metric_out_of_range_raises(cfg: Config) -> None:
    client = _ok_client()
    with pytest.raises(DDWUsageError, match=r"\[0, 1\]"):
        with _make_run(client, cfg) as run:
            run.log_metric_final(1.5)


def test_metric_negative_raises(cfg: Config) -> None:
    client = _ok_client()
    with pytest.raises(DDWUsageError):
        with _make_run(client, cfg) as run:
            run.log_metric_final(-0.1)


def test_invalid_budget_kind_raises(cfg: Config) -> None:
    client = _ok_client()
    with pytest.raises(DDWUsageError, match="budget_kind"):
        Run(
            model_tag="m1",
            dataset_size=100,
            budget_kind="hours",  # type: ignore[arg-type]
            budget_value=1,
            metric_name="accuracy",
            config=cfg,
            client=client,
        )


def test_empty_model_tag_raises(cfg: Config) -> None:
    client = _ok_client()
    with pytest.raises(DDWUsageError, match="model_tag"):
        Run(
            model_tag="   ",
            dataset_size=100,
            budget_kind="epochs",
            budget_value=1,
            metric_name="accuracy",
            config=cfg,
            client=client,
        )


def test_zero_dataset_size_raises(cfg: Config) -> None:
    client = _ok_client()
    with pytest.raises(DDWUsageError, match="dataset_size"):
        Run(
            model_tag="m1",
            dataset_size=0,
            budget_kind="epochs",
            budget_value=1,
            metric_name="accuracy",
            config=cfg,
            client=client,
        )


# ---------------- on_error="warn" ----------------


def test_on_error_warn_does_not_raise_on_api_error(
    cfg: Config, caplog: pytest.LogCaptureFixture
) -> None:
    client = MagicMock()
    client.ingest_run.side_effect = DDWAPIError(500, "boom")

    with caplog.at_level(logging.WARNING, logger="ddw.sdk"):
        with _make_run(client, cfg, on_error="warn") as run:
            run.log_metric_final(0.7)

    assert run.submitted is None
    assert any("DDW run report failed" in rec.message for rec in caplog.records)


def test_on_error_raise_propagates_api_error(cfg: Config) -> None:
    client = MagicMock()
    client.ingest_run.side_effect = DDWAPIError(422, [{"loc": ["body", "x"]}])

    with pytest.raises(DDWAPIError) as ei:
        with _make_run(client, cfg) as run:
            run.log_metric_final(0.7)

    assert ei.value.status_code == 422


# ---------------- status overrides ----------------


def test_set_status_can_force_failed(cfg: Config) -> None:
    client = _ok_client()
    with _make_run(client, cfg) as run:
        run.log_metric_final(0.4)
        run.set_status("failed")

    p = client.ingest_run.call_args.args[0]
    assert p["run"]["status"] == "FAILED"


def test_set_status_invalid_raises(cfg: Config) -> None:
    client = _ok_client()
    run = _make_run(client, cfg)
    with pytest.raises(DDWUsageError, match="status"):
        run.set_status("weird")  # type: ignore[arg-type]


# ---------------- client lifecycle ----------------


def test_supplied_client_is_not_closed(cfg: Config) -> None:
    """If user passed a client explicitly, Run must not close it
    (user owns its lifecycle)."""

    client = _ok_client()
    with _make_run(client, cfg) as run:
        run.log_metric_final(0.5)

    client.close.assert_not_called()
