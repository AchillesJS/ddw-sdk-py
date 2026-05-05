"""Optional PyTorch helper utilities for DDW diagnostics.

The functions in this module are explicit helpers only. They do not register
hooks, mutate model or optimizer state, or log anything automatically.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable
from typing import Any

from ddw.exceptions import DDWUsageError


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as e:
        if e.name == "torch":
            raise DDWUsageError(
                'PyTorch helpers require torch. Install with: pip install "ddw-sdk[torch]"'
            ) from e
        raise


def _finite_result(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise DDWUsageError(f"{name} must be a finite float, got {result!r}")
    return result


def _combined_norm(tensors: Iterable[Any], norm_type: float) -> float:
    torch = _torch()
    values = list(tensors)
    if not values:
        return 0.0

    p = float(norm_type)
    if math.isinf(p):
        result = max(float(torch.linalg.vector_norm(t.detach(), ord=p).item()) for t in values)
    else:
        total = 0.0
        for tensor in values:
            norm = float(torch.linalg.vector_norm(tensor.detach(), ord=p).item())
            total += norm**p
        result = total ** (1.0 / p)
    return _finite_result(result, "norm")


def compute_grad_norm_global(model: Any, norm_type: float = 2.0) -> float:
    """L_p norm of all parameter gradients flattened together.

    Returns 0.0 if no parameter has a non-None grad.
    Skips parameters where grad is None.
    Returns a plain Python float.
    """

    _torch()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return _combined_norm(grads, norm_type)


def compute_grad_norm_layer(model: Any, layer_name: str, norm_type: float = 2.0) -> float:
    """Gradient norm for exactly one named module.

    Use model.named_modules().
    Use parameters(recurse=False).
    Raise KeyError if layer_name is missing.
    Return 0.0 if the layer has no gradients.
    """

    _torch()
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise KeyError(layer_name)
    layer = modules[layer_name]
    grads = [parameter.grad for parameter in layer.parameters(recurse=False) if parameter.grad is not None]
    return _combined_norm(grads, norm_type)


def compute_weight_norm_global(model: Any, norm_type: float = 2.0) -> float:
    """L_p norm of all model parameters.

    Returns a plain Python float.
    """

    _torch()
    return _combined_norm(model.parameters(), norm_type)


def get_current_lr(optimizer: Any, group_index: int = 0) -> float:
    """Return optimizer.param_groups[group_index]['lr'] as float.

    Raise IndexError for invalid group_index.
    """

    _torch()
    if group_index < 0 or group_index >= len(optimizer.param_groups):
        raise IndexError(group_index)
    return _finite_result(float(optimizer.param_groups[group_index]["lr"]), "learning rate")
