from __future__ import annotations

from typing import Any

import pytest

from ddw import Config, DDWUsageError, Run


class FakeClient:
    def __init__(self) -> None:
        self.ingest_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []

    def ingest_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ingest_calls.append(payload)
        return {
            "run_id": "r_test",
            "external_run_id": payload["external_run_id"],
            "created": True,
        }

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.create_calls.append(payload)
        return {"id": "r_legacy"}

    def close(self) -> None:
        pass


def _make_run(client: FakeClient, cfg: Config, **overrides: Any) -> Run:
    defaults: dict[str, Any] = dict(
        model_tag="m1",
        dataset_size=1000,
        budget_kind="epochs",
        budget_value=10,
        metric_name="accuracy",
        config=cfg,
        client=client,
    )
    defaults.update(overrides)
    return Run(**defaults)


def _one_payload(client: FakeClient) -> dict[str, Any]:
    assert len(client.ingest_calls) == 1
    return client.ingest_calls[0]


def test_log_metric_appends_point(cfg: Config) -> None:
    client = FakeClient()
    run = _make_run(client, cfg)

    run.log_metric("val", "epoch", 1, 0.7)

    assert run.metric_points == [
        {"series_name": "val", "x_kind": "epoch", "x_value": 1.0, "y_value": 0.7}
    ]


def test_log_metric_rejects_y_above_one(cfg: Config) -> None:
    client = FakeClient()
    run = _make_run(client, cfg)

    with pytest.raises(DDWUsageError, match=r"\[0, 1\]"):
        run.log_metric("val", "epoch", 1, 1.4)


def test_log_metric_rejects_y_below_zero(cfg: Config) -> None:
    client = FakeClient()
    run = _make_run(client, cfg)

    with pytest.raises(DDWUsageError, match=r"\[0, 1\]"):
        run.log_metric("val", "epoch", 1, -0.1)


def test_log_metric_rejects_non_increasing_x(cfg: Config) -> None:
    client = FakeClient()
    run = _make_run(client, cfg)
    run.log_metric("val", "epoch", 2, 0.7)

    with pytest.raises(DDWUsageError, match="strictly increasing"):
        run.log_metric("val", "epoch", 2, 0.8)


def test_log_metric_allows_independent_series_with_different_x_progressions(cfg: Config) -> None:
    client = FakeClient()
    run = _make_run(client, cfg)

    run.log_metric("train", "epoch", 10, 0.8)
    run.log_metric("val", "epoch", 1, 0.7)

    assert [p["series_name"] for p in run.metric_points] == ["train", "val"]


def test_log_metric_separate_x_kinds_in_same_series_are_independent(cfg: Config) -> None:
    client = FakeClient()
    run = _make_run(client, cfg)

    run.log_metric("val", "epoch", 10, 0.8)
    run.log_metric("val", "step", 1, 0.7)

    assert [p["x_kind"] for p in run.metric_points] == ["epoch", "step"]


def test_log_diagnostic_appends_point(cfg: Config) -> None:
    client = FakeClient()
    run = _make_run(client, cfg)

    run.log_diagnostic("loss", "epoch", 1, 1.23)

    assert run.diagnostic_signals == [
        {"series_name": "loss", "x_kind": "epoch", "x_value": 1.0, "y_value": 1.23}
    ]


def test_log_diagnostic_accepts_unbounded_y(cfg: Config) -> None:
    client = FakeClient()
    run = _make_run(client, cfg)

    run.log_diagnostic("grad_norm", "step", 1, 12345.0)

    assert run.diagnostic_signals[0]["y_value"] == 12345.0


def test_log_diagnostic_rejects_non_finite_y(cfg: Config) -> None:
    client = FakeClient()
    run = _make_run(client, cfg)

    with pytest.raises(DDWUsageError, match="finite"):
        run.log_diagnostic("loss", "epoch", 1, float("inf"))


def test_log_diagnostic_rejects_non_increasing_x(cfg: Config) -> None:
    client = FakeClient()
    run = _make_run(client, cfg)
    run.log_diagnostic("loss", "epoch", 2, 1.0)

    with pytest.raises(DDWUsageError, match="strictly increasing"):
        run.log_diagnostic("loss", "epoch", 1, 0.9)


def test_exit_with_no_data_raises(cfg: Config) -> None:
    client = FakeClient()

    with pytest.raises(DDWUsageError, match="Run metric data was never set"):
        with _make_run(client, cfg):
            pass

    assert client.ingest_calls == []


def test_exit_with_only_metric_final_synthesizes_one_point(cfg: Config) -> None:
    client = FakeClient()

    with _make_run(client, cfg, external_run_id="provided") as run:
        run.log_metric_final(0.8)

    payload = _one_payload(client)
    assert payload["run"]["metric_final"] == 0.8
    assert payload["metric_points"] == [
        {"series_name": "val", "x_kind": "epoch", "x_value": 10.0, "y_value": 0.8}
    ]


def test_exit_with_metric_points_ignores_metric_final(cfg: Config) -> None:
    client = FakeClient()

    with _make_run(client, cfg) as run:
        run.log_metric_final(0.1)
        run.log_metric("val", "epoch", 1, 0.6)
        run.log_metric("val", "epoch", 2, 0.7)

    payload = _one_payload(client)
    assert payload["run"]["metric_final"] == 0.7
    assert len(payload["metric_points"]) == 2


def test_exit_posts_run_report_v2_payload_shape(cfg: Config) -> None:
    client = FakeClient()

    with _make_run(client, cfg, external_run_id="run-x", notes="note") as run:
        run.log_metric("val", "epoch", 1, 0.6)
        run.log_diagnostic("loss", "epoch", 1, 2.5)

    payload = _one_payload(client)
    assert payload == {
        "schema_version": 2,
        "external_run_id": "run-x",
        "run": {
            "model_tag": "m1",
            "dataset_size": 1000.0,
            "budget_kind": "epochs",
            "budget_value": 10.0,
            "metric_name": "accuracy",
            "metric_final": 0.6,
            "split": "val",
            "dataset_version": None,
            "seed": None,
            "config_hash": None,
            "git_commit": None,
            "status": "SUCCEEDED",
        },
        "metric_points": [
            {"series_name": "val", "x_kind": "epoch", "x_value": 1.0, "y_value": 0.6}
        ],
        "diagnostic_signals": [
            {"series_name": "loss", "x_kind": "epoch", "x_value": 1.0, "y_value": 2.5}
        ],
        "artifacts": [],
        "notes": "note",
    }


def test_exit_posts_to_ingest_endpoint_not_create_run(cfg: Config) -> None:
    client = FakeClient()

    with _make_run(client, cfg) as run:
        run.log_metric_final(0.8)

    assert len(client.ingest_calls) == 1
    assert client.create_calls == []


def test_exit_status_failed_when_exception(cfg: Config) -> None:
    client = FakeClient()

    with pytest.raises(RuntimeError):
        with _make_run(client, cfg) as run:
            run.log_metric("val", "epoch", 1, 0.6)
            raise RuntimeError("boom")

    payload = _one_payload(client)
    assert payload["run"]["status"] == "FAILED"


def test_external_run_id_generated_when_not_provided(cfg: Config) -> None:
    client = FakeClient()

    with _make_run(client, cfg) as run:
        run.log_metric_final(0.8)

    payload = _one_payload(client)
    assert payload["external_run_id"].startswith("sdk-")
    assert len(payload["external_run_id"]) == len("sdk-") + 16


def test_external_run_id_used_when_provided(cfg: Config) -> None:
    client = FakeClient()

    with _make_run(client, cfg, external_run_id="ext-123") as run:
        run.log_metric_final(0.8)

    assert _one_payload(client)["external_run_id"] == "ext-123"


def test_metric_final_derived_from_last_val_point_by_x_value(cfg: Config) -> None:
    client = FakeClient()

    with _make_run(client, cfg) as run:
        run.log_metric("val", "step", 100, 0.7)
        run.log_metric("val", "epoch", 2, 0.6)
        run.log_metric("train", "step", 1000, 0.9)

    payload = _one_payload(client)
    assert payload["run"]["metric_final"] == 0.7


def test_invalid_local_logging_raises_even_with_on_error_warn(cfg: Config) -> None:
    client = FakeClient()
    run = _make_run(client, cfg, on_error="warn")

    with pytest.raises(DDWUsageError):
        run.log_metric("val", "epoch", 1, 1.1)


def test_log_metric_final_backward_compatible_sends_valid_v2_report(cfg: Config) -> None:
    client = FakeClient()

    with _make_run(client, cfg, external_run_id="old-style", on_error="raise") as run:
        run.log_metric_final(0.8)

    payload = _one_payload(client)
    assert payload["schema_version"] == 2
    assert payload["run"]["metric_final"] == 0.8
    assert payload["metric_points"] == [
        {"series_name": "val", "x_kind": "epoch", "x_value": 10.0, "y_value": 0.8}
    ]
    assert payload["run"]["status"] == "SUCCEEDED"
    assert run.submitted == {"run_id": "r_test", "external_run_id": "old-style", "created": True}
