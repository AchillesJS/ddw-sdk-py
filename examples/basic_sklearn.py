"""sklearn-style example demonstrating multiple runs at different dataset sizes.

This example creates 4 runs at different dataset_size values.
After running, you can hit the DDW analytics endpoint:

    curl -H "X-Tenant-Id: team-vision" \\
      "http://localhost:8000/api/v1/analytics/learning?model_tag=logreg&metric_name=f1&x_kind=dataset_size&split=val"

and DDW will return saturation_estimate + regime_estimate based on these 4 points.

Then save it as a DataPotential:

    curl -X POST -H "X-Tenant-Id: team-vision" \\
      -H "Content-Type: application/json" \\
      "http://localhost:8000/api/v1/analytics/save-potential" \\
      -d '{"model_tag":"logreg","metric_name":"f1","x_kind":"dataset_size","split":"val"}'

Usage:
    export DDW_BASE_URL=http://localhost:8000
    export DDW_TENANT_ID=team-vision
    python examples/basic_sklearn.py
"""

from __future__ import annotations

import math
import random

from ddw import Run


def train_and_eval(dataset_size: int, seed: int) -> float:
    """Stub: pretend to train logistic regression at given dataset size.

    Returns f1 score in [0, 1] following a saturating curve.
    """

    random.seed(seed)
    # Simple log-saturation curve with noise.
    base = 0.4 + 0.25 * math.log10(max(dataset_size, 10))
    return max(0.0, min(1.0, base + random.uniform(-0.02, 0.02)))


def main() -> None:
    dataset_sizes = [500, 1_000, 4_000, 16_000]
    seed = 7

    for ds in dataset_sizes:
        with Run(
            model_tag="logreg",
            dataset_size=ds,
            budget_kind="epochs",
            budget_value=1,
            metric_name="f1",
            split="val",
            seed=seed,
            dataset_version="v1",
        ) as run:
            f1 = train_and_eval(ds, seed=seed)
            print(f"dataset_size={ds:>6}  f1={f1:.4f}")
            run.log_metric_final(f1)

        print(f"  → run id: {run.submitted['id'] if run.submitted else 'NOT POSTED'}")


if __name__ == "__main__":
    main()
