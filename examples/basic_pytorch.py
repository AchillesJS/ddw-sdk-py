# This example is the canonical 'how to use DDW v0.2 from PyTorch' reference. Update it whenever the SDK API changes.
"""Self-contained PyTorch training loop with DDW v0.2 metrics and diagnostics."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ddw import Run
from ddw.frameworks.pytorch import compute_grad_norm_global, get_current_lr


class TinySGD:
    def __init__(self, parameters: Any, lr: float) -> None:
        self.param_groups = [{"params": list(parameters), "lr": float(lr)}]

    def zero_grad(self) -> None:
        for parameter in self.param_groups[0]["params"]:
            parameter.grad = None

    def step(self) -> None:
        lr = self.param_groups[0]["lr"]
        with torch.no_grad():
            for parameter in self.param_groups[0]["params"]:
                if parameter.grad is not None:
                    parameter -= lr * parameter.grad


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))


def make_dataset(seed: int = 42) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(256, 4, generator=generator)
    y = ((x[:, 0] + 0.7 * x[:, 1] - 0.4 * x[:, 2]) > 0).long()
    return x[:192], y[:192], x[192:], y[192:]


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        return float((pred == y).float().mean().item())


def main(client: Any = None, epochs: int = 10) -> dict[str, Any] | None:
    seed = 42
    torch.manual_seed(seed)
    x_train, y_train, x_val, y_val = make_dataset(seed)
    model = make_model()
    optimizer = TinySGD(model.parameters(), lr=0.2)
    loss_fn = nn.CrossEntropyLoss()

    with Run(
        client=client,
        model_tag="tiny-pytorch-classifier",
        dataset_size=float(len(x_train) + len(x_val)),
        budget_kind="epochs",
        budget_value=float(epochs),
        metric_name="accuracy",
        split="val",
        seed=seed,
    ) as run:
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(x_train), y_train)
            loss.backward()

            grad_norm = compute_grad_norm_global(model)
            lr = get_current_lr(optimizer)
            optimizer.step()

            train_acc = accuracy(model, x_train, y_train)
            val_acc = accuracy(model, x_val, y_val)
            run.log_metric("train", "epoch", epoch, train_acc)
            run.log_metric("val", "epoch", epoch, val_acc)
            run.log_diagnostic("grad_norm_global", "epoch", epoch, grad_norm)
            run.log_diagnostic("lr", "epoch", epoch, lr)
            print(
                f"epoch={epoch:02d} loss={loss.item():.4f} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
                f"grad_norm={grad_norm:.3f} lr={lr:.3f}"
            )

    print(f"DDW recorded run: {run.submitted}")
    return run.submitted


if __name__ == "__main__":
    main()
