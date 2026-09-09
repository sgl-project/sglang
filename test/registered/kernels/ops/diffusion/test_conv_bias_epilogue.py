"""``conv_bias_epilogue``: bit-exact conv bias (+ residual) epilogue.

Oracle: aten itself. ``F.conv3d`` adds its bias as ``out.add_(bias.view(...))``
(fp32 opmath, one rounding), and the eager residual chain is ``conv(x) + h``
with a second rounding. The kernel must reproduce both exactly, keep the
input's dense channels_last(_3d) strides, and refuse anything else through
its predicate (never by raising from ``can_use_``).
"""

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_conv_bias_epilogue,
    conv_bias_epilogue,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"


def _dense(shape, dtype):
    fmt = torch.channels_last_3d if len(shape) == 5 else torch.channels_last
    return torch.randn(shape, device=DEVICE, dtype=dtype).contiguous(memory_format=fmt)


def _bias_view(bias, ndim):
    return bias.view(1, -1, *([1] * (ndim - 2)))


@torch.no_grad()
@pytest.mark.parametrize(
    "x_dtype,bias_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
        (torch.float32, torch.float32),
    ],
)
@pytest.mark.parametrize(
    "shape", [(1, 96, 2, 30, 52), (2, 192, 1, 15, 26), (4, 96, 30, 52), (1, 3, 4, 8, 8)]
)
def test_conv_bias_epilogue_matches_aten(x_dtype, bias_dtype, shape):
    x = _dense(shape, x_dtype)
    c = shape[1]
    bias = torch.randn(c, device=DEVICE, dtype=bias_dtype)
    # autocast casts the bias to the activation dtype before the conv call.
    ref = x.clone().add_(_bias_view(bias.to(x_dtype), x.dim()))
    assert can_use_conv_bias_epilogue(x, bias)
    out = conv_bias_epilogue(x, bias)
    assert out.stride() == x.stride()
    assert torch.equal(out, ref)

    h = _dense(shape, x_dtype)
    ref_res = ref + h
    assert can_use_conv_bias_epilogue(x, bias, h)
    out_res = conv_bias_epilogue(x, bias, h)
    assert out_res.stride() == x.stride()
    assert torch.equal(out_res, ref_res)


@torch.no_grad()
def test_conv_bias_epilogue_equals_conv_with_bias():
    # End-to-end oracle: F.conv3d with bias vs conv without bias + epilogue.
    conv = torch.nn.Conv3d(48, 64, 3, padding=1).to(DEVICE, torch.bfloat16)
    conv.weight.data = conv.weight.data.to(memory_format=torch.channels_last_3d)
    x = _dense((1, 48, 3, 12, 20), torch.bfloat16)
    ref = conv(x)
    raw = torch.nn.functional.conv3d(x, conv.weight, None, padding=1)
    assert can_use_conv_bias_epilogue(raw, conv.bias)
    assert torch.equal(conv_bias_epilogue(raw, conv.bias), ref)


@torch.no_grad()
def test_conv_bias_epilogue_rejects_unsupported_inputs():
    x = _dense((1, 96, 2, 6, 6), torch.bfloat16)
    bias = torch.randn(96, device=DEVICE, dtype=torch.bfloat16)
    assert not can_use_conv_bias_epilogue(x.contiguous(), bias)  # NCDHW
    assert not can_use_conv_bias_epilogue(x, bias[:95])  # wrong width
    assert not can_use_conv_bias_epilogue(x, bias.to(torch.float16))  # dtype
    assert not can_use_conv_bias_epilogue(x, bias, x.contiguous())  # residual layout
    assert not can_use_conv_bias_epilogue(
        x, bias, x.to(torch.float32)
    )  # residual dtype
    assert not can_use_conv_bias_epilogue(x[:, :, :0], bias)  # empty
    x1 = torch.randn(1, 1, 2, 3, 4, device=DEVICE, dtype=torch.bfloat16)
    assert x1.is_contiguous(memory_format=torch.channels_last_3d)
    assert not can_use_conv_bias_epilogue(x1, bias[:1])  # C == 1 is ambiguous
    with pytest.raises(ValueError):
        conv_bias_epilogue(x.contiguous(), bias)
    with torch.enable_grad():
        xg = x.clone().requires_grad_(True)
        assert not can_use_conv_bias_epilogue(xg, bias)
        with pytest.raises(ValueError):
            conv_bias_epilogue(xg, bias)
