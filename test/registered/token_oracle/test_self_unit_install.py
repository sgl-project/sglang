from __future__ import annotations

from types import SimpleNamespace

from sglang.srt.kv_canary.token_oracle.install import install_mock_model_sampler
from sglang.srt.layers.sampler import _CUSTOM_SAMPLER_FACTORIES
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")


def _make_server_args(*, sampling_backend: str) -> SimpleNamespace:
    return SimpleNamespace(sampling_backend=sampling_backend)


def test_install_mock_model_sampler_disabled_returns_none() -> None:
    server_args = _make_server_args(sampling_backend="auto")
    hook = install_mock_model_sampler(server_args=server_args, vocab_size=1000)
    assert hook is None


def test_install_mock_model_sampler_enabled_registers_oracle_backend() -> None:
    server_args = _make_server_args(sampling_backend="oracle")
    hook = install_mock_model_sampler(server_args=server_args, vocab_size=512)
    assert hook is not None
    assert "oracle" in _CUSTOM_SAMPLER_FACTORIES


def test_install_mock_model_sampler_enabled_returns_hook_with_hash_oracle() -> None:
    from sglang.srt.kv_canary.token_oracle.oracle import HashOracle

    server_args = _make_server_args(sampling_backend="oracle")
    hook = install_mock_model_sampler(server_args=server_args, vocab_size=256)
    assert hook is not None
    assert isinstance(hook.oracle, HashOracle)
    assert hook.oracle.vocab_size == 256
