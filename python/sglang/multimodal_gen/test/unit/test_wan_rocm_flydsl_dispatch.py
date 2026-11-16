from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from sglang.kernels.ops import diffusion
from sglang.kernels.ops.diffusion import WAN_ROCM_FLYDSL_NORM_SCALE_SHIFT
from sglang.kernels.ops.diffusion.sites import wan_rocm_flydsl_norm_scale_shift_site
from sglang.multimodal_gen.runtime.layers import layernorm
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    ScaleResidualLayerNormScaleShift,
)
from sglang.multimodal_gen.runtime.models.dits import wanvideo
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    _QUALITY_FUSION_HANDLERS,
    toggle_quality_fusions,
)


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
def test_wan_high_quality_aligned_norm_uses_flydsl(
    monkeypatch, layer_cls, num_inputs, fused_name
):
    monkeypatch.setattr(layernorm, "_use_rocm_flydsl", True)
    layer = layer_cls(5120, rocm_flydsl_high_only=True)
    inputs = [torch.empty(1, 5120) for _ in range(num_inputs)]
    expected = (
        torch.empty_like(inputs[0])
        if num_inputs == 3
        else (torch.empty_like(inputs[0]), torch.empty_like(inputs[0]))
    )
    monkeypatch.setitem(vars(diffusion), "FLYDSL_NORM_MIN_ALIGNED_DIM", 5120)
    monkeypatch.setitem(vars(diffusion), fused_name, lambda *args: expected)

    monkeypatch.setattr(
        wan_rocm_flydsl_norm_scale_shift_site.current_platform, "is_rocm", lambda: True
    )
    toggle_quality_fusions([layer], "high")

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
    monkeypatch.setattr(layernorm, "_use_rocm_flydsl", enabled)
    layer = layer_cls(hidden_size, rocm_flydsl_high_only=True)
    inputs = [torch.empty(1, hidden_size) for _ in range(num_inputs)]
    expected = object()
    monkeypatch.setitem(vars(diffusion), "FLYDSL_NORM_MIN_ALIGNED_DIM", 5120)

    with patch.object(layer, "forward_native", return_value=expected) as native:
        actual = layer.forward_hip(*inputs)

    assert actual is expected
    native.assert_called_once_with(*inputs)


def test_wan_transformer_block_marks_high_quality_sites(monkeypatch):

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
        layer = getattr(block, name)
        assert WAN_ROCM_FLYDSL_NORM_SCALE_SHIFT.metadata(layer) == 128
        assert not WAN_ROCM_FLYDSL_NORM_SCALE_SHIFT.is_enabled(layer)


@pytest.mark.parametrize(
    "layer_cls,num_inputs",
    [(LayerNormScaleShift, 3), (ScaleResidualLayerNormScaleShift, 5)],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_wan_quality_transitions_restore_native(
    monkeypatch, layer_cls, num_inputs, enabled
):
    monkeypatch.setattr(layernorm, "_use_rocm_flydsl", enabled)
    monkeypatch.setattr(
        wan_rocm_flydsl_norm_scale_shift_site.current_platform, "is_rocm", lambda: True
    )
    monkeypatch.setitem(vars(diffusion), "FLYDSL_NORM_MIN_ALIGNED_DIM", 5120)
    layer = layer_cls(5120, rocm_flydsl_high_only=True)
    inputs = [torch.empty(1, 5120) for _ in range(num_inputs)]
    assert ("high", WAN_ROCM_FLYDSL_NORM_SCALE_SHIFT) in _QUALITY_FUSION_HANDLERS
    for quality in ("exact", "lossless", "high", "lossless", "exact"):
        toggle_quality_fusions([layer], quality)
        assert WAN_ROCM_FLYDSL_NORM_SCALE_SHIFT.is_enabled(layer) == (
            enabled and quality == "high"
        )
        if not enabled or quality != "high":
            with patch.object(
                layer, "forward_native", return_value="reference"
            ) as native:
                assert layer.forward_hip(*inputs) == "reference"
                native.assert_called_once_with(*inputs)


@pytest.mark.parametrize(
    "reason", ["disabled", "unaligned", "epsilon", "platform", "missing_backend"]
)
def test_wan_fusion_is_all_or_nothing(monkeypatch, reason):
    monkeypatch.setattr(layernorm, "_use_rocm_flydsl", True)
    monkeypatch.setattr(
        wan_rocm_flydsl_norm_scale_shift_site.current_platform,
        "is_rocm",
        lambda: reason != "platform",
    )
    monkeypatch.setitem(vars(diffusion), "FLYDSL_NORM_MIN_ALIGNED_DIM", 5120)
    model = nn.ModuleList(
        [LayerNormScaleShift(5120, rocm_flydsl_high_only=True) for _ in range(2)]
    )
    if reason == "disabled":
        monkeypatch.setattr(layernorm, "_use_rocm_flydsl", False)
    elif reason == "unaligned":
        WAN_ROCM_FLYDSL_NORM_SCALE_SHIFT.mark(model[1], 128)
    elif reason == "epsilon":
        model[1].eps = 1e-5
    elif reason == "missing_backend":
        monkeypatch.delitem(vars(diffusion), "FLYDSL_NORM_MIN_ALIGNED_DIM")
        original = diffusion.__getattr__

        def unavailable(name):
            if name == "FLYDSL_NORM_MIN_ALIGNED_DIM":
                raise ImportError("FlyDSL unavailable")
            return original(name)

        monkeypatch.setattr(diffusion, "__getattr__", unavailable)
    toggle_quality_fusions([model], "high")
    assert not any(
        WAN_ROCM_FLYDSL_NORM_SCALE_SHIFT.is_enabled(layer) for layer in model
    )


@pytest.mark.parametrize(
    "layer_cls", [LayerNormScaleShift, ScaleResidualLayerNormScaleShift]
)
def test_environment_permission_is_cached_before_layer_construction(
    monkeypatch, layer_cls
):
    permission = layernorm._use_rocm_flydsl
    monkeypatch.setenv("SGLANG_USE_ROCM_FLYDSL", "0" if permission else "1")
    layer = layer_cls(5120, rocm_flydsl_high_only=True)
    assert layernorm._use_rocm_flydsl is permission
    assert not WAN_ROCM_FLYDSL_NORM_SCALE_SHIFT.is_enabled(layer)
