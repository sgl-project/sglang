import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.aiter as aiter_runner
from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


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
    fake_aiter.ActivationType = SimpleNamespace(Silu="Silu", Swiglu="Swiglu")
    fake_aiter.QuantType = SimpleNamespace(per_1x32="per_1x32")

    fake_fused_moe = ModuleType("aiter.fused_moe")
    fake_fused_moe.fused_moe = fused_moe

    fake_ops = ModuleType("aiter.ops")
    fake_ops.__path__ = []
    fake_flydsl = ModuleType("aiter.ops.flydsl")
    fake_flydsl.__path__ = []
    fake_moe_common = ModuleType("aiter.ops.flydsl.moe_common")
    fake_moe_common.GateMode = SimpleNamespace(
        INTERLEAVE=SimpleNamespace(value="INTERLEAVE"),
        SEPARATED=SimpleNamespace(value="SEPARATED"),
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
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_fake_topk_slot", lambda: True
    )

    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu", no_combine=True))

    runner.run(
        _runner_input(),
        _quant_info(
            expert_mask=torch.ones(2, dtype=torch.int32),
            fused_moe_kwargs={"custom_fused_moe_kwarg": "enabled"},
        ),
        running_state={},
    )

    assert captured["activation"] == "Silu"
    assert captured["quant_type"] == "per_1x32"
    assert captured["no_combine"] is True
    assert captured["has_fake_topk_slot"] is False
    assert captured["custom_fused_moe_kwarg"] == "enabled"


def test_standard_ep_rejects_unpaired_aiter(monkeypatch):
    def fused_moe(**kwargs):
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_fake_topk_slot", lambda: False
    )
    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu"))

    with pytest.raises(NotImplementedError, match="paired AITER v0.1.19.post2"):
        runner.run(
            _runner_input(),
            _quant_info(expert_mask=torch.ones(2, dtype=torch.int32)),
            running_state={},
        )


def test_aiter_runner_maps_minimax_gated_silu_to_clamped_swiglu(monkeypatch):
    from sglang.srt.environ import envs

    captured = {}

    def fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    monkeypatch.setattr(envs.SGLANG_USE_AITER_MOE_GU_ITLV, "get", lambda: False)
    runner = AiterRunnerCore(
        MoeRunnerConfig(
            activation="silu",
            gemm1_alpha=1.702,
            gemm1_clamp_limit=7.0,
        )
    )

    runner.run(
        _runner_input(),
        _quant_info(
            w13_weight=torch.empty((2, 8, 2), dtype=torch.float8_e4m3fn)
        ),
        running_state={},
    )

    assert captured["activation"] == "Swiglu"
    assert captured["gate_mode"] == "INTERLEAVE"
    assert captured["swiglu_limit"] == 7.0
    # EP1 has no expert mask and must use the raw 128-expert/top-k-4 lookup key.
    assert "has_fake_topk_slot" not in captured


def test_mxfp8_aiter_quant_info_uses_per_1x32(monkeypatch):
    import sglang.srt.layers.quantization.fp8 as fp8_quant

    monkeypatch.setattr(fp8_quant, "_use_aiter", True)
    method = fp8_quant.Fp8MoEMethod.__new__(fp8_quant.Fp8MoEMethod)
    method.block_quant = True
    method.is_fp4_expert = False
    method.use_mxfp8 = True
    method.moe_runner_config = SimpleNamespace(swiglu_limit=None)

    layer = SimpleNamespace(
        w13_weight=torch.empty((2, 8, 4)),
        w2_weight=torch.empty((2, 4, 4)),
        w13_weight_scale_inv=torch.empty((2, 8, 1), dtype=torch.uint8),
        w2_weight_scale_inv=torch.empty((2, 4, 1), dtype=torch.uint8),
        dispatcher=SimpleNamespace(expert_mask_gpu=None),
        hidden_pad=0,
        intermediate_pad=0,
    )

    quant_info = method.maybe_get_hip_aiter_quant_info(layer)

    assert quant_info.quant_type is AiterQuantType.PER_1X32


def test_mxfp8_aiter_moe_preshuffle_forces_gate_up_interleave(monkeypatch):
    import sglang.srt.layers.quantization.fp8 as fp8_quant

    scale_calls = []
    weight_calls = []

    def shuffle_scale(value, experts_cnt, is_guinterleave, gate_up):
        scale_calls.append((experts_cnt, is_guinterleave, gate_up))
        return value.clone()

    def shuffle_weight(value, *, is_guinterleave, gate_up):
        weight_calls.append((is_guinterleave, gate_up))
        return value.detach().clone()

    monkeypatch.setattr(fp8_quant, "_use_aiter", True)
    monkeypatch.setattr(fp8_quant, "_is_hip", True)
    monkeypatch.setattr(fp8_quant, "_is_gfx95_supported", True)
    monkeypatch.setattr(fp8_quant, "shuffle_scale", shuffle_scale, raising=False)
    monkeypatch.setattr(fp8_quant, "shuffle_weight", shuffle_weight, raising=False)

    method = fp8_quant.Fp8MoEMethod.__new__(fp8_quant.Fp8MoEMethod)
    method.use_mxfp8 = True
    method.runner = SimpleNamespace(
        runner_backend=SimpleNamespace(is_aiter=lambda: True)
    )

    layer = torch.nn.Module()
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(
            torch.empty((2, 8, 4), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight",
        torch.nn.Parameter(
            torch.empty((2, 4, 4), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w13_weight_scale_inv",
        torch.nn.Parameter(
            torch.empty((2, 8, 1), dtype=torch.uint8), requires_grad=False
        ),
    )
    layer.register_parameter(
        "w2_weight_scale_inv",
        torch.nn.Parameter(
            torch.empty((2, 4, 1), dtype=torch.uint8), requires_grad=False
        ),
    )

    method._process_mxfp8_moe_weights(layer, quantize=False)

    assert scale_calls == [(2, True, True), (2, True, False)]
    assert weight_calls == [(True, True), (True, False)]
    assert layer.w13_weight.is_shuffled is True
    assert layer.w2_weight.is_shuffled is True


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
