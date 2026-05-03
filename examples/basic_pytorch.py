"""PyTorch-style training loop with DDW SDK.

Usage:
    export DDW_BASE_URL=http://localhost:8000
    export DDW_TENANT_ID=team-vision
    python examples/basic_pytorch.py

This is a self-contained example with a stub trainer. Replace `train_one_epoch`
and `evaluate` with your real training code.
"""

from __future__ import annotations

import random

from ddw import Run


def train_one_epoch(epoch: int) -> float:
    """Return a fake training loss."""

    return max(0.05, 1.0 / (epoch + 1) + random.uniform(-0.05, 0.05))


def evaluate() -> float:
    """Return a fake validation accuracy in [0, 1]."""

    return 0.82 + random.uniform(-0.03, 0.03)


def main() -> None:
    epochs = 10
    dataset_size = 10_000
    seed = 42

    random.seed(seed)

    with Run(
        model_tag="resnet50",
        dataset_size=dataset_size,
        budget_kind="epochs",
        budget_value=epochs,
        metric_name="accuracy",
        split="val",
        seed=seed,
        # In production, pass on_error="warn" so that DDW failures
        # don't crash your training:
        # on_error="warn",
    ) as run:
        for ep in range(epochs):
            loss = train_one_epoch(ep)
            print(f"epoch {ep:>2}  loss={loss:.4f}")

        val_acc = evaluate()
        print(f"final val_acc={val_acc:.4f}")
        run.log_metric_final(val_acc)

    print(f"DDW recorded run: {run.submitted}")


if __name__ == "__main__":
    main()
