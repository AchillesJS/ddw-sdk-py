# DDW Python SDK

A synchronous Python client for reporting ML training evidence to [Data Decision Workspace](https://github.com/AchillesJS/data-decision-layer). **Status: MVP; package version 0.2.0.** It collects explicitly logged metrics and diagnostic signals, then submits a RunReport v2 when a `Run` context exits.

## Install from source

Requires Python 3.11 or newer. These instructions use a checkout; PyPI availability has not been independently verified.

```bash
git clone https://github.com/AchillesJS/ddw-sdk-py.git
cd ddw-sdk-py
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows use `.venv\Scripts\Activate.ps1`. The base SDK has one runtime dependency, `httpx`.

## Minimal example

Start the backend using its README first. This complete example submits synthetic input, not training results:

```python
from ddw import Config, Run

config = Config(base_url="http://127.0.0.1:8000", tenant_id="portfolio-demo")
with Run(
    model_tag="synthetic-baseline",
    dataset_size=100,
    budget_kind="epochs",
    budget_value=1,
    metric_name="accuracy",
    config=config,
) as run:
    run.log_metric("val", "epoch", 1, 0.75)
    run.log_diagnostic("loss_train", "epoch", 1, 0.5)
print(run.submitted)
```

`log_metric_final(0.75)` can replace the time-series metric call; it creates a single `val` point if no metric points exist. Metric values must be in `[0, 1]`; diagnostic values must be finite. Coordinates must increase strictly within each series and coordinate kind.

## Configuration

Pass `Config` directly or omit it to resolve environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `DDW_BASE_URL` | Backend base URL | Required |
| `DDW_TENANT_ID` | Sent as `X-Tenant-Id` | Required |
| `DDW_TIMEOUT_S` | HTTP timeout in seconds | `15.0` |

`Config.from_env_or_args(...)` gives explicit arguments precedence over environment variables. Direct `Config(...)` construction does not read the environment. Tenant headers are routing identifiers, not authentication credentials.

## Backend compatibility

`Run` sends `schema_version: 2` to `POST /api/v1/runs/ingest`. The audited backend revision is [4ea325e](https://github.com/AchillesJS/data-decision-layer/commit/4ea325ec92e5d17cb605df0f657f58419869f5a4); no minimum released backend version is established. API `/api/v1` and report schema v2 are separate version concepts.

Use at least one `val` metric series and metric `x_kind` of `epoch`, `step`, or `time` for compatibility. The SDK accepts some inputs that backend validation rejects, including unsupported metric coordinate kinds and reports without a `val` point. The deprecated `Client.create_run` remains available; `Client.ingest_run` is the current submission path. `Client.health()` returns false on HTTP transport failure or a non-200 health response.

## Error behavior

- Missing configuration raises `DDWConfigError`; invalid logging input raises `DDWUsageError`. A malformed timeout environment value can raise `ValueError`.
- HTTP error responses raise `DDWAPIError`, exposing `status_code` and `detail`. Transport errors from httpx propagate directly.
- `on_error="warn"` logs report-building usage errors and API response errors via `ddw.sdk`; it does **not** suppress network/timeout errors or errors raised by logging calls inside the context.
- A block exception marks a report failed if metric data exists. Without metric data, it propagates without submission. A reporting exception can supersede a training exception in raise mode.
- Successful context exit without metric data raises `DDWUsageError` by default. Submission occurs synchronously at exit; there are no retries or background uploads.

## Optional PyTorch helpers

```bash
python -m pip install -e '.[torch]'
```

`ddw.frameworks.pytorch` provides explicit diagnostics such as `compute_grad_norm_global(model)` and `get_current_lr(optimizer)`. Compute gradient norms after backpropagation and log the returned values yourself. Helpers do not install hooks or automatically upload metrics. See [PyTorch example](examples/basic_pytorch.py); it uses synthetic training data. The [multi-run example](examples/basic_sklearn.py) illustrates reporting across dataset sizes; inspect its requirements before running it.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check .
python -m mypy ddw
```

Tests use fake/mocked transports and do not require a live backend. PyTorch-dependent tests skip when torch is absent. Install `.[dev,torch]` to run those tests. See [contribution guidance](CONTRIBUTING.md).

## Limitations and licensing

No async API, retry queue, model artifact upload, automatic training instrumentation, or automatic git/config-hash inference is implemented. The SDK reports evidence; backend analysis and decision records live in DDW. No production reliability or benchmark claims are made.

`pyproject.toml` declares `Proprietary`; no LICENSE file is present. This audit preserves those terms. Public visibility does not establish an open-source license.
