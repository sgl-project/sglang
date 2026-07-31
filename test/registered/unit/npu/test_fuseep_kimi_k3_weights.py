from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.moe import fuseep
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="stage-a-unit-test-npu")


def test_mega_moe_w4a8_weights_use_gemm_layout(monkeypatch):
    monkeypatch.setattr(fuseep, "npu_format_cast", lambda tensor: tensor)
    experts, hidden, intermediate = 2, 16, 12
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.empty(experts, intermediate, hidden, dtype=torch.int8),
            requires_grad=False,
        ),
        w2_weight=torch.nn.Parameter(
            torch.empty(
                experts, hidden // 2, intermediate, dtype=torch.int8
            ),
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
