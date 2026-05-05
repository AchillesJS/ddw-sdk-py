from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")


class FakeClient:
    def __init__(self) -> None:
        self.ingest_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []

    def ingest_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ingest_calls.append(payload)
        return {
            "run_id": "r_example",
            "external_run_id": payload["external_run_id"],
            "created": True,
        }

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.create_calls.append(payload)
        return {"id": "r_legacy"}

    def close(self) -> None:
        pass


def _load_basic_pytorch_example() -> Any:
    path = Path(__file__).resolve().parents[1] / "examples" / "basic_pytorch.py"
    spec = importlib.util.spec_from_file_location("basic_pytorch_example", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_basic_pytorch_example_emits_v2_metrics_and_diagnostics() -> None:
    module = _load_basic_pytorch_example()
    client = FakeClient()

    submitted = module.main(client=client, epochs=4)

    assert submitted == {
        "run_id": "r_example",
        "external_run_id": client.ingest_calls[0]["external_run_id"],
        "created": True,
    }
    assert client.create_calls == []
    assert len(client.ingest_calls) == 1

    payload = client.ingest_calls[0]
    assert payload["schema_version"] == 2
    assert payload["run"]["status"] == "SUCCEEDED"

    metric_series = {point["series_name"] for point in payload["metric_points"]}
    diagnostic_series = {point["series_name"] for point in payload["diagnostic_signals"]}

    assert {"val", "train"}.issubset(metric_series)
    assert {"grad_norm_global", "lr"}.issubset(diagnostic_series)
    assert len([p for p in payload["metric_points"] if p["series_name"] == "val"]) >= 4
    assert len([p for p in payload["metric_points"] if p["series_name"] == "train"]) >= 4
    assert len(payload["diagnostic_signals"]) >= 8
