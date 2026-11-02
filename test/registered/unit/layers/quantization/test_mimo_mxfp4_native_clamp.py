"""Reject unsupported clamps before constructing the native MiMo MoE runner."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.quantization import mxfp4_aiter_a16w4_moe as module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-b-test-cpu")


@pytest.mark.parametrize("field", ["swiglu_limit", "gemm1_clamp_limit"])
@pytest.mark.parametrize("limit", [1.0, 7.0, -1.0, float("inf"), float("nan")])
def test_native_clamp_is_rejected_before_runner_setup(monkeypatch, field, limit):
    def unexpected(*args, **kwargs):
        pytest.fail("An unsupported clamp must fail before constructing a runner")

    monkeypatch.setattr(module, "MoeRunner", unexpected)
    method = module.Mxfp4AiterA16W4MoEMethod(None, "test.native_clamp")
    config = MoeRunnerConfig(**{field: limit})
    with pytest.raises(ValueError, match="does not support activation clamps"):
        method.create_moe_runner(torch.nn.Module(), config)
    assert not hasattr(method, "runner")
    assert not hasattr(method, "moe_runner_config")


@pytest.mark.parametrize("swiglu_limit", [None, 0.0])
@pytest.mark.parametrize("gemm1_clamp_limit", [None, 0.0])
def test_native_unclamped_configuration_is_preserved(
    monkeypatch, swiglu_limit, gemm1_clamp_limit
):
    calls = []
    runner = SimpleNamespace(runner_backend=MoeRunnerBackend.AITER)

    def make_runner(backend, config):
        calls.append((backend, config))
        return runner

    monkeypatch.setattr(module, "MoeRunner", make_runner)
    config = MoeRunnerConfig(
        swiglu_limit=swiglu_limit, gemm1_clamp_limit=gemm1_clamp_limit
    )
    method = module.Mxfp4AiterA16W4MoEMethod(None, "test.native_clamp")
    method.create_moe_runner(torch.nn.Module(), config)
    assert calls == [(MoeRunnerBackend.AITER, config)]
    assert method.moe_runner_config is config
    assert method.runner is runner
    assert method.get_moe_runner_backend(None) == MoeRunnerBackend.AITER
