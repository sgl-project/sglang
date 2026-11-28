import argparse
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.moe import fuseep
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="stage-a-unit-test-npu")


def test_fuseep_mode_three_cli():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)

    args = parser.parse_args(["--model-path", "dummy", "--fuseep-mode", "3"])

    assert args.fuseep_mode == 3


def test_fuseep_mode_three_routes_to_mega_moe(monkeypatch):
    received = {}

    class FakeBuffer:
        def fused_deep_moe(self, **kwargs):
            received.update(kwargs)
            return kwargs["x"], None

    monkeypatch.setattr(fuseep, "_get_fuseep_buffer", lambda _layer: FakeBuffer())
    monkeypatch.setattr(
        fuseep,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(fuseep_mode=3)),
    )

    hidden_states = torch.randn(2, 16, dtype=torch.bfloat16)
    layer = SimpleNamespace(
        moe_runner_config=SimpleNamespace(
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
        ),
        w13_weight=[torch.empty(16, 6, dtype=torch.int32)],
        w13_weight_scale=[torch.empty(48, dtype=torch.uint64)],
        w2_weight=[torch.empty(12, 2, dtype=torch.int32)],
        w2_weight_scale=[torch.empty(16, dtype=torch.uint64)],
        w13_scale_bias=[torch.empty(48, dtype=torch.float32)],
        w2_scale_bias=[torch.empty(16, dtype=torch.float32)],
        num_experts=8,
    )
    topk_output = SimpleNamespace(
        topk_ids=torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
        topk_weights=torch.tensor([[0.6, 0.4], [0.7, 0.3]]),
    )

    output = fuseep.forward_fuseep(layer, hidden_states, topk_output)

    assert output is hidden_states
    assert received["backend"] == "mega_moe"
    assert received["activation"] == "situ"
    assert received["beta"] == 4.0
    assert received["linear_beta"] == 25.0
    assert received["topk_idx"].dtype == torch.int32
    assert received["topk_weights"].dtype == torch.float32


def test_mega_moe_w4a8_weights_use_gemm_layout(monkeypatch):
    monkeypatch.setattr(fuseep, "npu_format_cast", lambda tensor: tensor)
    experts, hidden, intermediate = 2, 16, 12
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.empty(experts, intermediate, hidden, dtype=torch.int8),
            requires_grad=False,
        ),
        w2_weight=torch.nn.Parameter(
            torch.empty(experts, hidden // 2, intermediate, dtype=torch.int8),
            requires_grad=False,
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.empty(experts, 2 * intermediate, 1),
            requires_grad=False,
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.empty(experts, hidden, 1),
            requires_grad=False,
        ),
        w13_scale_bias=torch.nn.Parameter(
            torch.empty(experts, 2 * intermediate, 1),
            requires_grad=False,
        ),
        w2_scale_bias=torch.nn.Parameter(
            torch.arange(experts * hidden * 2, dtype=torch.float32).reshape(
                experts, hidden, 2
            ),
            requires_grad=False,
        ),
    )
    expected_w2_bias = layer.w2_scale_bias.sum(dim=-1)

    fuseep._process_mega_moe_weights(layer)

    assert len(layer.w13_weight) == experts
    assert layer.w13_weight[0].shape == (hidden, intermediate // 4)
    assert layer.w2_weight[0].shape == (intermediate, hidden // 8)
    assert layer.w13_weight[0].dtype == torch.int32
    assert layer.w2_weight[0].dtype == torch.int32
    assert layer.w13_weight_scale[0].dtype == torch.uint64
    assert layer.w2_weight_scale[0].dtype == torch.uint64
    assert layer.w13_weight_scale[0].shape == (2 * intermediate,)
    assert layer.w2_weight_scale[0].shape == (hidden,)
    assert layer.w13_scale_bias[0].dtype == torch.float32
    assert layer.w2_scale_bias[0].dtype == torch.float32
    assert layer.w13_scale_bias[0].shape == (2 * intermediate,)
    assert layer.w2_scale_bias[0].shape == (hidden,)
    torch.testing.assert_close(layer.w2_scale_bias[0], expected_w2_bias[0])
