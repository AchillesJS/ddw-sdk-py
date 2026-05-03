from __future__ import annotations

import pytest

from ddw import Config, DDWConfigError


def test_explicit_args_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DDW_BASE_URL", "http://env")
    monkeypatch.setenv("DDW_TENANT_ID", "env-tenant")

    cfg = Config.from_env_or_args(base_url="http://arg", tenant_id="arg-tenant")
    assert cfg.base_url == "http://arg"
    assert cfg.tenant_id == "arg-tenant"


def test_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DDW_BASE_URL", "http://env")
    monkeypatch.setenv("DDW_TENANT_ID", "env-tenant")

    cfg = Config.from_env_or_args()
    assert cfg.base_url == "http://env"
    assert cfg.tenant_id == "env-tenant"
    assert cfg.timeout_s == 15.0


def test_env_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DDW_BASE_URL", "http://env")
    monkeypatch.setenv("DDW_TENANT_ID", "env-tenant")
    monkeypatch.setenv("DDW_TIMEOUT_S", "3.5")

    cfg = Config.from_env_or_args()
    assert cfg.timeout_s == 3.5


def test_missing_base_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DDW_BASE_URL", raising=False)
    monkeypatch.setenv("DDW_TENANT_ID", "t")

    with pytest.raises(DDWConfigError, match="base_url"):
        Config.from_env_or_args()


def test_missing_tenant_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DDW_BASE_URL", "http://x")
    monkeypatch.delenv("DDW_TENANT_ID", raising=False)

    with pytest.raises(DDWConfigError, match="tenant_id"):
        Config.from_env_or_args()


def test_strips_trailing_slash() -> None:
    cfg = Config.from_env_or_args(base_url="http://x/", tenant_id="t")
    assert cfg.base_url == "http://x"


def test_direct_construction_validates() -> None:
    with pytest.raises(DDWConfigError):
        Config(base_url="", tenant_id="t")
    with pytest.raises(DDWConfigError):
        Config(base_url="http://x", tenant_id="")
    with pytest.raises(DDWConfigError):
        Config(base_url="http://x", tenant_id="t", timeout_s=0)
