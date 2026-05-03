from __future__ import annotations

import pytest

from ddw import Config


@pytest.fixture()
def cfg() -> Config:
    return Config(base_url="http://test", tenant_id="t1", timeout_s=5.0)
