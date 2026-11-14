"""MiniMax-H3's fused RMSNorm + indexed AdaLN (quality lossless/high) against the
same math in fp32 with one rounding, and against the reference rounding chain."""

import sys

import pytest
import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion import (
    can_use_rmsnorm_indexed_scale_shift,
    can_use_rmsnorm_indexed_scale_shift_fp8,
    indexed_scale_shift_bf16_,
    rmsnorm_indexed_scale_shift,
    rmsnorm_indexed_scale_shift_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

register_cuda_ci(est_time=30, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")

from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

EPS = 1e-5


def _inputs(rows: int, hidden: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (2 * torch.randn(rows, hidden, device="cuda", generator=g)).to(torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(hidden, device="cuda", generator=g)).to(
        torch.bfloat16
    )
    # H3's shift/scale are chunks of one [M, 6 * hidden] AdaLN output
    adaln = (0.1 * torch.randn(3, 6 * hidden, device="cuda", generator=g)).to(
        torch.bfloat16
    )
    shift, scale = adaln[:, :hidden], adaln[:, hidden : 2 * hidden]
    indices = torch.randint(0, 3, (rows,), device="cuda", generator=g)
    return x, weight, shift, scale, indices


@pytest.mark.parametrize("hidden", [1024, 5376])
@pytest.mark.parametrize("rows", [1, 37, 1025])
def test_matches_fp32_single_rounding(rows: int, hidden: int) -> None:
    x, weight, shift, scale, indices = _inputs(rows, hidden, rows + hidden)
    assert can_use_rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices)
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()
    ref = normed * (1 + scale[indices].float()) + shift[indices].float()
    got = rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, EPS)
    # only the reduction order differs: at most one bf16 rounding step apart
    torch.testing.assert_close(got.float(), ref, rtol=2**-7, atol=2**-7)


def test_stays_within_rounding_of_the_reference_chain() -> None:
    x, weight, shift, scale, indices = _inputs(1025, 5376, 7)
    norm = nn.RMSNorm(5376, eps=EPS, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(weight)
    chain = indexed_scale_shift_bf16_(norm(x), shift, scale, indices)
    got = rmsnorm_indexed_scale_shift(x, weight, shift, scale, indices, EPS)
    torch.testing.assert_close(got.float(), chain.float(), rtol=2**-6, atol=2**-6)


def assert_quant_equal(actual, expected):
    aq, asc = actual
    eq, esc = expected
    assert torch.equal(aq.view(torch.uint8), eq.view(torch.uint8))
    assert torch.equal(asc, esc)


@pytest.mark.parametrize(
    "rows,index_dtype",
    [(4096, torch.int64), (4144, torch.int64), (16432, torch.int64)]
    + [(4096, dtype) for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32)],
)
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="per-token quantized output targets SM90",
)
def test_norm_modulate_fp8_bits(rows, index_dtype):
    torch.manual_seed(42)
    x = torch.randn(rows, 5376, device="cuda", dtype=torch.bfloat16)
    # Include zero and tiny rows, plus outliers. Use strided modulation views
    # and indices, as H3's split AdaLN tables are not contiguous across rows.
    x[0].zero_()
    x[1] *= 1e-12
    x[2, ::127] *= 100
    w = torch.randn(5376, device="cuda", dtype=torch.bfloat16)
    table = torch.randn(6, 5376 * 6, device="cuda", dtype=torch.bfloat16)
    shift, scale = table[:, :5376], table[:, 5376:10752]
    shift[0].zero_()
    indices = torch.randint(0, 6, (rows * 2,), device="cuda").to(index_dtype)[::2]
    indices[0] = 0
    strided_weight = torch.empty(10752, device="cuda", dtype=w.dtype)[::2]
    strided_weight.copy_(w)
    assert not can_use_rmsnorm_indexed_scale_shift_fp8(
        x, strided_weight, shift, scale, indices
    )
    with pytest.raises(ValueError, match="weight must be contiguous"):
        rmsnorm_indexed_scale_shift_fp8(
            x,
            strided_weight,
            shift,
            scale,
            indices,
            EPS,
        )
    reference = rmsnorm_indexed_scale_shift(x, w, shift, scale, indices, 1e-5)
    expected = sglang_per_token_quant_fp8(reference)
    y, actual = rmsnorm_indexed_scale_shift_fp8(
        x,
        w,
        shift,
        scale,
        indices,
        1e-5,
        keep_bf16=True,
    )
    assert torch.equal(y, reference)
    assert_quant_equal(actual, expected)
    y, actual = rmsnorm_indexed_scale_shift_fp8(x, w, shift, scale, indices, 1e-5)
    assert y is None
    assert_quant_equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="per-token quantized output targets SM90",
)
@pytest.mark.parametrize(
    "invalid", ["short_rows", "fp32", "bool_indices", "cpu_indices"]
)
def test_fp8_rejects_unsupported_inputs(invalid):
    x, weight, shift, scale, indices = _inputs(4096, 5376, 93)
    if invalid == "short_rows":
        x, indices = x[:4095], indices[:4095]
    elif invalid == "fp32":
        x = x.float()
    elif invalid == "bool_indices":
        indices = indices.bool()
    elif invalid == "cpu_indices":
        indices = indices.cpu()
    assert not can_use_rmsnorm_indexed_scale_shift_fp8(x, weight, shift, scale, indices)
    with pytest.raises(ValueError, match="unsupported indexed RMSNorm FP8"):
        rmsnorm_indexed_scale_shift_fp8(x, weight, shift, scale, indices, EPS)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
