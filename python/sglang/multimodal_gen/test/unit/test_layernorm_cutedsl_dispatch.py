import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNormScaleShift,
    ScaleResidualRMSNormScaleShift,
)

_CUTEDSL_MODULE = "sglang.kernels.ops.diffusion.cutedsl.scale_residual_norm_scale_shift"


@pytest.mark.parametrize("hidden_size", [257, 8448])
def test_norm_scale_shift_cuda_falls_back_for_unsupported_hidden_size(hidden_size):
    layer = RMSNormScaleShift(hidden_size)
    x = torch.empty(1, 1, hidden_size)
    shift = torch.empty(1, 1, hidden_size)
    scale = torch.empty(1, 1, hidden_size)
    expected = object()

    with (
        patch.object(layer, "forward_native", return_value=expected) as native,
        pytest.warns(UserWarning, match="native fallback"),
    ):
        actual = layer.forward_cuda(x, shift, scale)

    assert actual is expected
    native.assert_called_once_with(x, shift, scale)


@pytest.mark.parametrize("hidden_size", [257, 8448])
def test_scale_residual_cuda_falls_back_for_unsupported_hidden_size(hidden_size):
    layer = ScaleResidualRMSNormScaleShift(hidden_size)
    residual = torch.empty(1, 1, hidden_size)
    x = torch.empty(1, 1, hidden_size)
    gate = torch.empty(1, 1, hidden_size)
    shift = torch.empty(1, 1, hidden_size)
    scale = torch.empty(1, 1, hidden_size)
    expected = object()

    with (
        patch.object(layer, "forward_native", return_value=expected) as native,
        pytest.warns(UserWarning, match="native fallback"),
    ):
        actual = layer.forward_cuda(residual, x, gate, shift, scale)

    assert actual is expected
    native.assert_called_once_with(residual, x, gate, shift, scale)


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
