from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from ddw.frameworks.pytorch import (  # noqa: E402
    compute_grad_norm_global,
    compute_grad_norm_layer,
    compute_weight_norm_global,
    get_current_lr,
)


class TwoLayerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def _backward(model: nn.Module) -> None:
    x = torch.randn(5, 3)
    y = torch.tensor([0, 1, 0, 1, 0])
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()


def _manual_norm(tensors: list[torch.Tensor], norm_type: float = 2.0) -> float:
    if not tensors:
        return 0.0
    p = float(norm_type)
    if math.isinf(p):
        return max(float(torch.linalg.vector_norm(t.detach(), ord=p).item()) for t in tensors)
    total = sum(float(torch.linalg.vector_norm(t.detach(), ord=p).item()) ** p for t in tensors)
    return float(total ** (1.0 / p))


def test_compute_grad_norm_global_zero_before_backward() -> None:
    model = TwoLayerModel()

    assert compute_grad_norm_global(model) == 0.0


def test_compute_grad_norm_global_matches_manual_computation() -> None:
    model = TwoLayerModel()
    _backward(model)

    grads = [p.grad for p in model.parameters() if p.grad is not None]

    assert compute_grad_norm_global(model) == pytest.approx(_manual_norm(grads))
    assert isinstance(compute_grad_norm_global(model), float)


def test_compute_grad_norm_global_handles_partial_grads() -> None:
    model = TwoLayerModel()
    _backward(model)
    model.encoder[0].weight.grad = None

    grads = [p.grad for p in model.parameters() if p.grad is not None]

    assert compute_grad_norm_global(model) == pytest.approx(_manual_norm(grads))


def test_compute_grad_norm_layer_returns_value_for_known_layer() -> None:
    model = TwoLayerModel()
    _backward(model)

    grads = [p.grad for p in model.head.parameters(recurse=False) if p.grad is not None]

    assert compute_grad_norm_layer(model, "head") == pytest.approx(_manual_norm(grads))


def test_compute_grad_norm_layer_recurse_false_excludes_subtree() -> None:
    model = TwoLayerModel()
    _backward(model)

    assert compute_grad_norm_layer(model, "encoder") == 0.0


def test_compute_grad_norm_layer_raises_on_unknown_layer() -> None:
    model = TwoLayerModel()

    with pytest.raises(KeyError):
        compute_grad_norm_layer(model, "missing")


def test_compute_weight_norm_global_matches_manual_computation() -> None:
    model = TwoLayerModel()

    weights = [p for p in model.parameters()]

    assert compute_weight_norm_global(model) == pytest.approx(_manual_norm(weights))
    assert isinstance(compute_weight_norm_global(model), float)


class DummyOptimizer:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 0.01}, {"lr": 0.02}]


def test_get_current_lr_default_group() -> None:
    optimizer = DummyOptimizer()

    assert get_current_lr(optimizer) == pytest.approx(0.01)
    assert isinstance(get_current_lr(optimizer), float)


def test_get_current_lr_with_group_index() -> None:
    optimizer = DummyOptimizer()

    assert get_current_lr(optimizer, 1) == pytest.approx(0.02)


def test_get_current_lr_raises_on_bad_index() -> None:
    optimizer = DummyOptimizer()

    with pytest.raises(IndexError):
        get_current_lr(optimizer, 2)
