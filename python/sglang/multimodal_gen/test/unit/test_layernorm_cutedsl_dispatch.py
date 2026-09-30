import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNormScaleShift,
    ScaleResidualRMSNormScaleShift,
)

_CUTEDSL_MODULE = "sglang.kernels.ops.diffusion.norm.scale_residual_norm_cutedsl"


@pytest.mark.parametrize("hidden_size", [257, 8448])
@pytest.mark.parametrize(
    "layer_cls,num_inputs",
    [(RMSNormScaleShift, 3), (ScaleResidualRMSNormScaleShift, 5)],
)
def test_cuda_falls_back_for_unsupported_hidden_size(
    hidden_size, layer_cls, num_inputs
):
    layer = layer_cls(hidden_size)
    inputs = [torch.empty(1, 1, hidden_size) for _ in range(num_inputs)]
    expected = object()

    with (
        patch.object(layer, "forward_native", return_value=expected) as native,
        pytest.warns(UserWarning, match="native fallback"),
    ):
        actual = layer.forward_cuda(*inputs)

    assert actual is expected
    native.assert_called_once_with(*inputs)


def test_norm_scale_shift_cuda_uses_cutedsl_for_supported_hidden_size(monkeypatch):
    hidden_size = 256
    layer = RMSNormScaleShift(hidden_size)
    x = torch.empty(1, 1, hidden_size)
    shift = torch.empty(1, 1, hidden_size)
    scale = torch.empty(1, 1, hidden_size)
    expected = object()

    def fused_norm_scale_shift(*args):
        return expected

    monkeypatch.setitem(
        sys.modules,
        _CUTEDSL_MODULE,
        SimpleNamespace(fused_norm_scale_shift=fused_norm_scale_shift),
    )

    assert layer.forward_cuda(x, shift, scale) is expected
