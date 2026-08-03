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
    fake_aiter.ActivationType = SimpleNamespace(Silu="Silu")
    fake_aiter.QuantType = SimpleNamespace(per_1x32="per_1x32")

    fake_fused_moe = ModuleType("aiter.fused_moe")
    fake_fused_moe.fused_moe = fused_moe
    fake_fused_moe.fused_moe_multi_b = fused_moe

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


def test_aiter_runner_injects_rocm_dwdp_partitions(monkeypatch):
    from types import SimpleNamespace

    from sglang.srt.runtime_context import set_global_dwdp_manager

    captured = {}

    def fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    weights1 = torch.empty((2, 8, 2))
    weights2 = torch.empty((2, 4, 2))
    scale1 = torch.empty((2, 8, 1), dtype=torch.uint8)
    scale2 = torch.empty((2, 4, 1), dtype=torch.uint8)
    refs = {
        weights1.untyped_storage().data_ptr(): "w13_weight",
        weights2.untyped_storage().data_ptr(): "w2_weight",
        scale1.untyped_storage().data_ptr(): "w13_weight_scale",
        scale2.untyped_storage().data_ptr(): "w2_weight_scale",
    }

    class FakeManager:
        weight_backend = "ipc"

        def find_partitioned_name(self, layer_idx, reference):
            assert layer_idx == 3
            return refs[reference.untyped_storage().data_ptr()]

        def get_partition_view(self, layer_idx, name, reference=None):
            assert layer_idx == 3
            if reference is None:
                reference = {
                    "w13_weight_scale": scale1,
                    "w2_weight_scale": scale2,
                }[name]
            return SimpleNamespace(tensors=(reference, reference.clone()))

    set_global_dwdp_manager(FakeManager())
    try:
        runner = AiterRunnerCore(MoeRunnerConfig(activation="silu", layer_id=3))
        runner.run(
            _runner_input(),
            _quant_info(
                w13_weight=weights1,
                w2_weight=weights2,
                w13_scale=scale1,
                w2_scale=scale2,
                expert_mask=torch.ones(3, dtype=torch.bool),
            ),
            running_state={},
        )
    finally:
        set_global_dwdp_manager(None)

    assert len(captured["w1_partitions"]) == 2
    assert len(captured["w2_partitions"]) == 2
    assert len(captured["w1_scale_partitions"]) == 2
    assert len(captured["w2_scale_partitions"]) == 2
    assert captured["w1_scale_partitions"][0].shape == (16, 1)
    assert captured["w2_scale_partitions"][0].shape == (8, 1)
    assert captured["expert_mask"] is None


def test_aiter_runner_preserves_fp32_block_scale_layout():
    fp8_scale = torch.empty((2, 4, 3), dtype=torch.float32)
    normalized = aiter_runner._normalize_dwdp_scale_partition(fp8_scale)
    assert normalized.shape == (2, 4, 3)
    assert normalized.dtype == torch.float32
    assert normalized.is_contiguous()


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
