# DDW Python SDK

Report ML training runs to **Data Decision Workspace** from your training code in 3 lines.

## Install (local dev)

The SDK is not on PyPI yet. From the SDK repo root:

```bash
pip install -e .
```

This installs `ddw-sdk` in editable mode with only one runtime dependency: `httpx`.

To install with dev tools (pytest, ruff, mypy):

```bash
pip install -e ".[dev]"
```

## Configure

Two ways: env vars (recommended) or explicit args.

```bash
export DDW_BASE_URL=http://localhost:8000
export DDW_TENANT_ID=team-vision
# optional:
export DDW_TIMEOUT_S=15
```

Or:

```python
from ddw import Config, Run

cfg = Config(base_url="http://localhost:8000", tenant_id="team-vision")
with Run(..., config=cfg) as run:
    ...
```

## Use

```python
from ddw import Run

with Run(
    model_tag="resnet50",
    dataset_size=10_000,
    budget_kind="epochs",       # "steps" | "epochs" | "time_sec"
    budget_value=10,
    metric_name="accuracy",
    split="val",                # default "val"
    seed=42,                    # optional but recommended
) as run:
    model = train(...)
    run.log_metric_final(eval_accuracy)
```

On context exit, v0.2 run reports are POSTed to `DDW_BASE_URL/api/v1/runs/ingest` with the `X-Tenant-Id` header.

## Production-safe usage

In real training scripts, set `on_error="warn"` so DDW reporting failures
never crash your training:

```python
with Run(..., on_error="warn") as run:
    ...
```

Failures are logged via the `ddw.sdk` logger (level WARNING).

## Behaviour reference

| Situation                                              | Behaviour                                  |
| ------------------------------------------------------ | ------------------------------------------ |
| Exit with `metric_final` set, no exception             | Status `succeeded`, run POSTed             |
| Exit with `metric_final` set, exception in block       | Status `failed`, run POSTed; exc reraised  |
| Exit without `metric_final`, no exception              | `DDWUsageError` raised                     |
| Exit without `metric_final`, exception in block        | No POST; original exception reraised       |
| HTTP 4xx/5xx from DDW                                  | `DDWAPIError` raised (or warned)           |

## What this SDK does NOT do

- No background threads, no retries, no async.
- No model artifact upload.
- No PyTorch/sklearn auto-instrumentation. Explicit API only.
- No automatic git/config_hash inference. You pass them yourself.

## Tests

```bash
pytest -q
```

All tests use mocked HTTP — no DDW backend required.

## Examples

See `examples/`:
- `basic_pytorch.py` — single training run
- `basic_sklearn.py` — multiple runs at different dataset sizes (feeds DDW analytics)

## What's new in v0.2

v0.2 adds direct RunReportV2 ingestion through `POST /api/v1/runs/ingest`.

New run logging methods:

- `run.log_metric(series, x_kind, x_value, y_value)` for normalized time-series quality metrics in `[0, 1]`.
- `run.log_diagnostic(series, x_kind, x_value, y_value)` for unbounded finite diagnostic signals such as loss, gradient norm, or latency.
- `run.log_metric_final(value)` still works and is submitted as a single `val` metric point.

```python
from ddw import Run

with Run(model_tag="resnet50", dataset_size=10_000, budget_kind="epochs", budget_value=3, metric_name="accuracy") as run:
    for epoch in range(1, 4):
        run.log_metric("val", "epoch", epoch, validate(epoch))
        run.log_diagnostic("loss", "epoch", epoch, train_loss(epoch))
```

## PyTorch helpers

PyTorch helpers are optional. The base SDK still installs with only `httpx`.
Install the optional PyTorch extra when you want framework-specific diagnostics:

```bash
pip install "ddw-sdk[torch]"
# local editable dev install:
pip install -e ".[torch]"
```

Helpers do not register hooks, do not mutate model or optimizer state, and do not
auto-log anything. You explicitly compute values and pass them to
`run.log_diagnostic(...)`.

```python
from ddw import Run
from ddw.frameworks.pytorch import compute_grad_norm_global, get_current_lr

with Run(model_tag="tiny", dataset_size=256, budget_kind="epochs", budget_value=10, metric_name="accuracy") as run:
    for epoch in range(1, 11):
        loss.backward()
        run.log_diagnostic("grad_norm_global", "epoch", epoch, compute_grad_norm_global(model))
        run.log_diagnostic("lr", "epoch", epoch, get_current_lr(optimizer))
        optimizer.step()
        run.log_metric("val", "epoch", epoch, val_accuracy)
```

See `examples/basic_pytorch.py` for the canonical v0.2 PyTorch reference.
