# SPDX-License-Identifier: Apache-2.0
"""Online MXFP8 GEMM path (``MXFP8Config()`` on a bf16 checkpoint): load-time
block quant, the prequantized (e4m3, swizzled scales) input from the fused
SwiGLU kernel, the per-layer fallback to the per-channel fp8 path, and the Intel XPU
block-scaled ``scaled_mm`` path."""

import sys

import pytest
import torch

_XPU = torch.xpu.is_available()
_DEVICE = "cuda" if torch.cuda.is_available() else "xpu"
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or _XPU), reason="CUDA or XPU required"
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _layer(
    in_f: int, out_f: int, bias: bool, params_dtype: torch.dtype = torch.bfloat16
):
    from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear
    from sglang.multimodal_gen.runtime.layers.quantization.mxfp8 import MXFP8Config

    return RowParallelLinear(
        in_f,
        out_f,
        bias=bias,
        params_dtype=params_dtype,
        quant_config=MXFP8Config(),
        prefix="mlp.fc2",
    ).to(_DEVICE)


@requires_cuda
def test_mxfp8_linear_matches_bf16_and_accepts_prequantized(request) -> None:
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("cuBLASLt MXFP8 block scaling requires Blackwell or newer")
    request.getfixturevalue("single_process_model_parallel")
    from sglang.kernels.ops.activation.swiglu_mxfp8_swizzled import silu_mul_mxfp8

    in_f, out_f, rows = 512, 384, 200
    layer = _layer(in_f, out_f, bias=False)
    g = torch.Generator(device="cpu").manual_seed(1)
    weight = (torch.randn(out_f, in_f, generator=g) * 0.02).to("cuda", torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.mxfp8 and layer.quant_method.accepts_mxfp8_input(layer)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.float8_e8m0fnu

    x = torch.randn(rows, in_f, generator=g).to("cuda", torch.bfloat16)
    out, _ = layer(x)
    ref = x.float() @ weight.float().t()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.05, rel

    hidden = torch.randn(rows, 2 * in_f, generator=g).to("cuda", torch.bfloat16)
    act = torch.nn.functional.silu(hidden[:, :in_f]) * hidden[:, in_f:]
    out_tensor, _ = layer(act)
    out_tuple, _ = layer(silu_mul_mxfp8(hidden))
    assert torch.equal(out_tensor, out_tuple)


@pytest.mark.skipif(not _XPU, reason="XPU required")
@pytest.mark.parametrize(
    "x_dtype, out_f, strided_x",
    [
        (torch.bfloat16, 384, False),
        (torch.float32, 384, False),
        # oneDNN's block-scaled GEMM has no N % 16 rule (unlike the cuBLASLt path).
        (torch.bfloat16, 100, False),
        # A row-strided slice, as from a fused QKV projection; the quant kernel
        # needs contiguous rows, so the helper copies only in this case.
        (torch.bfloat16, 384, True),
    ],
)
def test_xpu_mxfp8_linear_matches_bf16(request, x_dtype, out_f, strided_x) -> None:
    request.getfixturevalue("single_process_model_parallel")
    in_f, rows = 512, 200
    layer = _layer(in_f, out_f, bias=True)
    g = torch.Generator(device="cpu").manual_seed(1)
    weight = (torch.randn(out_f, in_f, generator=g) * 0.02).to("xpu", torch.bfloat16)
    bias = torch.randn(out_f, generator=g).to("xpu", torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.xpu_mxfp8 and not layer.quant_method.accepts_mxfp8_input(layer)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.float8_e8m0fnu
    assert layer.weight_scale.shape == (out_f, in_f // 32)

    x = torch.randn(rows, in_f, generator=g).to("xpu", x_dtype)
    if strided_x:
        x = torch.cat([x, x], dim=1)[:, :in_f]
        assert not x.is_contiguous()
    out, _ = layer(x)
    assert out.dtype == x_dtype
    ref = x.float() @ weight.float().t() + bias.float()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.05, rel


def _assert_aligned_layer_falls_back(params_dtype: torch.dtype) -> None:
    layer = _layer(512, 384, bias=False, params_dtype=params_dtype)
    g = torch.Generator(device="cpu").manual_seed(1)
    weight = (torch.randn(384, 512, generator=g) * 0.02).to(_DEVICE, params_dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.quant_method.process_weights_after_loading(layer)
    assert not layer.mxfp8 and not layer.quant_method.accepts_mxfp8_input(layer)
    assert not layer.xpu_mxfp8
    x = torch.randn(200, 512, generator=g).to(_DEVICE, params_dtype)
    out, _ = layer(x)
    ref = x.float() @ weight.float().t()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.05, rel


@requires_cuda
def test_pre_blackwell_aligned_layer_falls_back_to_channelwise(request) -> None:
    if torch.cuda.get_device_capability()[0] >= 10:
        pytest.skip("requires a pre-Blackwell GPU")
    request.getfixturevalue("single_process_model_parallel")
    _assert_aligned_layer_falls_back(torch.bfloat16)


def test_fp16_layer_falls_back_to_channelwise(request) -> None:
    """The swizzled quantizer takes bf16 only; an fp16 layer must not fail at load."""
    request.getfixturevalue("single_process_model_parallel")
    _assert_aligned_layer_falls_back(torch.float16)


def test_unaligned_layer_falls_back_to_channelwise(request) -> None:
    """The block-scaled GEMM needs K % 32 == 0; such a layer keeps the
    per-channel fp8 path and still answers a forward. K is 16 rather than a
    smaller odd size because the fallback's scaled GEMM still wants K % 16 == 0
    (ROCm rejects anything else outright)."""
    request.getfixturevalue("single_process_model_parallel")
    layer = _layer(16, 128, bias=True)
    layer.quant_method.process_weights_after_loading(layer)
    assert not layer.xpu_mxfp8
    assert not layer.mxfp8 and not layer.quant_method.accepts_mxfp8_input(layer)
    out, _ = layer(torch.randn(4, 16, device=_DEVICE, dtype=torch.bfloat16))
    assert out.shape == (4, 128)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
