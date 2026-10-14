import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.aiter as aiter_runner
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="stage-b-test-cpu-intel")


def _runner_input():
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    return AiterRunnerInput(
        hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
        topk_ids=topk_ids,
        topk_weights=torch.ones(topk_ids.shape, dtype=torch.float32),
        quant_type=AiterQuantType.PER_1X32,
    )


def _quant_info(**overrides):
    kwargs = {
        "w13_weight": torch.empty((2, 8, 2)),
        "w2_weight": torch.empty((2, 4, 2)),
        "quant_type": AiterQuantType.PER_1X32,
    }
    kwargs.update(overrides)
    return AiterMoeQuantInfo(**kwargs)


def _install_fake_aiter(monkeypatch, fused_moe):
    fake_aiter = ModuleType("aiter")
    fake_aiter.__path__ = []
    fake_aiter.ActivationType = SimpleNamespace(Silu="Silu")
    fake_aiter.QuantType = SimpleNamespace(per_1x32="per_1x32")

    fake_fused_moe = ModuleType("aiter.fused_moe")
    fake_fused_moe.fused_moe = fused_moe

    fake_ops = ModuleType("aiter.ops")
    fake_ops.__path__ = []
    fake_flydsl = ModuleType("aiter.ops.flydsl")
    fake_flydsl.__path__ = []
    fake_moe_common = ModuleType("aiter.ops.flydsl.moe_common")
    fake_moe_common.GateMode = SimpleNamespace(
        INTERLEAVE=SimpleNamespace(value="INTERLEAVE")
    )

    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fake_fused_moe)
    monkeypatch.setitem(sys.modules, "aiter.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", fake_flydsl)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.moe_common", fake_moe_common)


def test_aiter_runner_forwards_no_combine_and_extra_fused_moe_kwargs(monkeypatch):
    captured = {}

    def fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: True
    )

    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu", no_combine=True))

    runner.run(
        _runner_input(),
        _quant_info(fused_moe_kwargs={"custom_fused_moe_kwarg": "enabled"}),
        running_state={},
    )

    assert captured["activation"] == "Silu"
    assert captured["quant_type"] == "per_1x32"
    assert captured["no_combine"] is True
    assert captured["custom_fused_moe_kwarg"] == "enabled"


def test_aiter_runner_rejects_no_combine_when_fused_moe_does_not_support_it(
    monkeypatch,
):
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: False
    )
    runner = AiterRunnerCore(MoeRunnerConfig(no_combine=True))

    with pytest.raises(NotImplementedError, match="no_combine=True"):
        runner.run(_runner_input(), _quant_info(), running_state={})


def test_aiter_runner_preserves_no_combine_rank_for_empty_input(monkeypatch):
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: True
    )
    runner = AiterRunnerCore(MoeRunnerConfig(no_combine=True))
    runner_input = _runner_input()
    runner_input.hidden_states = torch.zeros((0, 4), dtype=torch.bfloat16)
    runner_input.topk_ids = torch.zeros((0, 2), dtype=torch.int32)
    runner_input.topk_weights = torch.zeros((0, 2), dtype=torch.float32)

    output = runner.run(runner_input, _quant_info(), running_state={})

    assert output.hidden_states.shape == (0, 2, 4)


@pytest.mark.parametrize("supports_output", [False, True])
def test_aiter_runner_uses_epv2_output_when_kernel_supports_it(
    monkeypatch, supports_output
):
    def fused_moe(hidden_states, output=None, **kwargs):
        result = hidden_states + 2
        if output is not None:
            output.copy_(result)
            return output
        return result

    _install_fake_aiter(monkeypatch, fused_moe)
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_output", lambda: supports_output
    )
    runner_input = _runner_input()
    runner_input.hidden_states.fill_(1)
    runner_input.output = torch.zeros_like(runner_input.hidden_states)
    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu"))

    result = runner.run(runner_input, _quant_info(), running_state={}).hidden_states

    torch.testing.assert_close(result, torch.full_like(result, 3))
    assert (result.data_ptr() == runner_input.output.data_ptr()) == supports_output
    torch.testing.assert_close(
        runner_input.output,
        torch.full_like(result, 3 if supports_output else 0),
    )


@pytest.mark.parametrize(
    "low_latency,recv_cap,expected_rows",
    [(False, 0, 16), (False, 32, 16), (True, 5, 5)],
)
def test_mori_pre_permute_consumes_dispatcher_cap(
    monkeypatch, low_latency, recv_cap, expected_rows
):
    from sglang.srt.layers.moe.token_dispatcher.moriep import (
        MoriEPLLDispatchOutput,
        MoriEPNormalDispatchOutput,
    )

    fake_utils = ModuleType("sglang.kernels.ops.moe.rocm_moe_utils")
    fake_utils.upscale = None
    fake_utils.upscale_mxfp4 = None
    monkeypatch.setitem(sys.modules, fake_utils.__name__, fake_utils)
    output_type = MoriEPLLDispatchOutput if low_latency else MoriEPNormalDispatchOutput
    hidden = torch.zeros((16, 4), dtype=torch.bfloat16)
    scales = torch.ones((16, 1))
    ids = torch.zeros((16, 2), dtype=torch.int32)
    weights = torch.ones((16, 2))
    kwargs = {} if low_latency else {"expert_output": torch.empty_like(hidden)}
    dispatched = output_type(
        hidden_states=hidden,
        hidden_states_scale=scales,
        topk_ids=ids,
        topk_weights=weights,
        num_recv_tokens_per_expert=torch.tensor([5]),
        origin_topk_ids=ids[:2],
        origin_topk_weights=weights[:2],
        out_dtype=torch.bfloat16,
        recv_cap=recv_cap,
        **kwargs,
    )
    quant_info = _quant_info(quant_type=AiterQuantType.NONE)
    state = {}
    # The dispatcher owns the override. The runner must not read it again.
    with envs.SGLANG_MORI_MOE_MAX_INPUT_TOKENS.override(3):
        result = aiter_runner._pre_permute_deepep_to_aiter(
            dispatched, quant_info, MoeRunnerConfig(), state
        )
    assert result.hidden_states.shape[0] == expected_rows
    assert result.topk_ids.shape[0] == expected_rows
    assert result.topk_weights.shape[0] == expected_rows
    assert result.a1_scale.shape[0] == expected_rows
    assert result.hidden_states.data_ptr() == hidden.data_ptr()
    assert state["aiter_combine_topk_ids"] is dispatched.origin_topk_ids
    if not low_latency:
        assert result.output.shape[0] == expected_rows
        assert result.output.data_ptr() == dispatched.expert_output.data_ptr()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
