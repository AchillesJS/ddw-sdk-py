from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

import ddw.frameworks.pytorch as pytorch_helpers
from ddw.exceptions import DDWUsageError


class _DummyModel:
    def parameters(self):  # type: ignore[no-untyped-def]
        return []

    def named_modules(self):  # type: ignore[no-untyped-def]
        return []


class _DummyOptimizer:
    param_groups = [{"lr": 0.1}]


def test_pytorch_helpers_raise_usage_error_when_torch_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str):  # type: ignore[no-untyped-def]
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return importlib.import_module(name)

    monkeypatch.setattr(pytorch_helpers.importlib, "import_module", fake_import_module)

    helpers = [
        lambda: pytorch_helpers.compute_grad_norm_global(_DummyModel()),
        lambda: pytorch_helpers.compute_grad_norm_layer(_DummyModel(), "layer"),
        lambda: pytorch_helpers.compute_weight_norm_global(_DummyModel()),
        lambda: pytorch_helpers.get_current_lr(_DummyOptimizer()),
    ]

    for helper in helpers:
        with pytest.raises(DDWUsageError, match="require torch"):
            helper()


def test_importing_pytorch_helper_module_does_not_require_torch() -> None:
    assert isinstance(pytorch_helpers.__name__, str)
    assert SimpleNamespace is not None
