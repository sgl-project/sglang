"""ROCm serving defaults hipBLASLt's workspace to 128 MiB unless the user sets one."""

from unittest.mock import patch

import pytest

from sglang.srt.entrypoints import engine
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_KEYS = ("HIPBLASLT_WORKSPACE_SIZE", "CUBLASLT_WORKSPACE_SIZE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)


def test_rocm_sets_default_workspace(monkeypatch):
    with patch.object(engine, "is_hip", return_value=True):
        engine._set_rocm_blas_envs()
    assert engine.os.environ["HIPBLASLT_WORKSPACE_SIZE"] == str(128 * 1024)


@pytest.mark.parametrize("key", _KEYS)
def test_rocm_keeps_user_workspace(monkeypatch, key):
    monkeypatch.setenv(key, "4096")
    with patch.object(engine, "is_hip", return_value=True):
        engine._set_rocm_blas_envs()
    assert engine.os.environ[key] == "4096"
    other = [k for k in _KEYS if k != key][0]
    assert other not in engine.os.environ


def test_non_rocm_leaves_env_unset():
    with patch.object(engine, "is_hip", return_value=False):
        engine._set_rocm_blas_envs()
    assert not any(k in engine.os.environ for k in _KEYS)
