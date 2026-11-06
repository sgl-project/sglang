from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from sglang.kernels.ops import diffusion
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    ScaleResidualLayerNormScaleShift,
)
from sglang.multimodal_gen.runtime.models.dits import wanvideo


@pytest.mark.parametrize(
    "layer_cls", [LayerNormScaleShift, ScaleResidualLayerNormScaleShift]
)
@pytest.mark.parametrize(
    "default,override,expected",
    [
        (False, None, False),
        (True, None, True),
        (True, "0", False),
        (False, "1", True),
    ],
)
def test_rocm_flydsl_default_and_environment_override(
    monkeypatch, layer_cls, default, override, expected
):
    if override is None:
        monkeypatch.delenv("SGLANG_USE_ROCM_FLYDSL", raising=False)
    else:
        monkeypatch.setenv("SGLANG_USE_ROCM_FLYDSL", override)

    layer = layer_cls(5120, rocm_flydsl_default=default)

    assert layer.use_rocm_flydsl is expected


@pytest.mark.parametrize(
    "layer_cls,num_inputs,fused_name",
    [
        (LayerNormScaleShift, 3, "flydsl_norm_scale_shift"),
        (
            ScaleResidualLayerNormScaleShift,
            5,
            "flydsl_fused_residual_norm_scale_shift",
        ),
    ],
)
def test_wan_aligned_norm_uses_flydsl(monkeypatch, layer_cls, num_inputs, fused_name):
    monkeypatch.delenv("SGLANG_USE_ROCM_FLYDSL", raising=False)
    layer = layer_cls(5120, rocm_flydsl_default=True)
    inputs = [torch.empty(1, 5120) for _ in range(num_inputs)]
    expected = (
        torch.empty_like(inputs[0])
        if num_inputs == 3
        else (torch.empty_like(inputs[0]), torch.empty_like(inputs[0]))
    )
    monkeypatch.setitem(vars(diffusion), "FLYDSL_NORM_MIN_ALIGNED_DIM", 5120)
    monkeypatch.setitem(vars(diffusion), fused_name, lambda *args: expected)

    with patch.object(layer, "forward_native") as native:
        actual = layer.forward_hip(*inputs)

    assert actual is expected
    native.assert_not_called()


@pytest.mark.parametrize(
    "layer_cls,num_inputs",
    [(LayerNormScaleShift, 3), (ScaleResidualLayerNormScaleShift, 5)],
)
@pytest.mark.parametrize("hidden_size,enabled", [(5120, False), (128, True)])
def test_wan_norm_uses_native_when_disabled_or_unaligned(
    monkeypatch, layer_cls, num_inputs, hidden_size, enabled
):
    monkeypatch.delenv("SGLANG_USE_ROCM_FLYDSL", raising=False)
    layer = layer_cls(hidden_size, rocm_flydsl_default=enabled)
    inputs = [torch.empty(1, hidden_size) for _ in range(num_inputs)]
    expected = object()
    monkeypatch.setitem(vars(diffusion), "FLYDSL_NORM_MIN_ALIGNED_DIM", 5120)

    with patch.object(layer, "forward_native", return_value=expected) as native:
        actual = layer.forward_hip(*inputs)

    assert actual is expected
    native.assert_called_once_with(*inputs)


@pytest.mark.parametrize("override,expected", [(None, True), ("0", False)])
def test_wan_transformer_block_selects_flydsl_by_default(
    monkeypatch, override, expected
):
    if override is None:
        monkeypatch.delenv("SGLANG_USE_ROCM_FLYDSL", raising=False)
    else:
        monkeypatch.setenv("SGLANG_USE_ROCM_FLYDSL", override)

    def empty_module(*args, **kwargs):
        return nn.Identity()

    for name in (
        "ColumnParallelLinear",
        "RowParallelLinear",
        "USPAttention",
        "WanT2VCrossAttention",
        "_WanGELUMLP",
    ):
        monkeypatch.setattr(wanvideo, name, empty_module)
    monkeypatch.setattr(wanvideo, "get_tp_world_size", lambda: 1)

    block = wanvideo.WanTransformerBlock(
        dim=128,
        ffn_dim=256,
        num_heads=4,
        cross_attn_norm=True,
        supported_attention_backends=set(),
    )

    for name in (
        "norm1",
        "self_attn_residual_norm",
        "cross_attn_residual_norm",
        "ffn_residual_norm1",
    ):
        assert getattr(block, name).use_rocm_flydsl is expected
