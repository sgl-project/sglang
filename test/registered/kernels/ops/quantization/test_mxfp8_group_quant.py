"""MXFP8 group quantization (1x32 block, ROUND_UP E8M0 scale) JIT kernel."""

import functools
import sys

import pytest
import torch

from sglang.kernels.jit.utils.compile.parallel import parallel_compile
from sglang.kernels.ops.quantization.mxfp8_group_quant import (
    _jit_mxfp8_group_quant_module,
    mxfp8_group_quant,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
SHAPES = [(1, 32), (7, 64), (128, 2880), (333, 5376), (1024, 7168)]


@pytest.fixture(scope="module", autouse=True)
def _build_all():
    parallel_compile(
        functools.partial(_jit_mxfp8_group_quant_module, dtype) for dtype in DTYPES
    )


def _make_input(shape, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape, device="cuda", generator=g) * 4
    # Spread the block amax over many binades so every exponent path is hit.
    x = x * torch.exp2(torch.randint(-20, 20, (shape[0], 1), device="cuda"))
    return x.to(dtype)


def _reference_scale(x: torch.Tensor) -> torch.Tensor:
    """Biased E8M0 exponent of 2^ceil(log2(amax / 448)), plus whether the IEEE
    quotient amax / 448 is itself a power of two (where an approximate division
    may legitimately land on either side)."""
    m, k = x.shape
    dequant = x.float().view(m, k // 32, 32).abs().amax(dim=-1) / 448.0
    bits = dequant.view(torch.int32)
    scale = ((bits + 0x007FFFFF) & 0x7F800000) >> 23
    exact_pow2 = (bits & 0x007FFFFF) == 0
    return scale.to(torch.uint8), exact_pow2


def _dequant_scale(scale_u8: torch.Tensor) -> torch.Tensor:
    return (scale_u8.to(torch.int32) << 23).view(torch.float32)


def _reference_values(x: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    d = _dequant_scale(scale_u8)
    quant_scale = torch.where(d == 0, torch.zeros_like(d), 1.0 / d)
    q = x.float().view(m, k // 32, 32) * quant_scale.unsqueeze(-1)
    return q.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(m, k)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_matches_reference(dtype, shape):
    x = _make_input(shape, dtype)
    q, scale = mxfp8_group_quant(x)
    assert q.shape == shape and q.dtype == torch.float8_e4m3fn
    assert scale.shape == (shape[0], shape[1] // 32) and scale.dtype == torch.uint8

    ref_scale, exact_pow2 = _reference_scale(x)
    mismatch = scale != ref_scale
    assert not (mismatch & ~exact_pow2).any()
    # On an exact power of two the approximate division may round one binade up.
    assert (scale[mismatch].int() - ref_scale[mismatch].int() == 1).all()

    # Values: bit-exact against the kernel's own scale.
    expected_q = _reference_values(x, scale)
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))


@pytest.mark.parametrize("dtype", DTYPES)
def test_edge_blocks(dtype):
    x = torch.randn((4, 128), device="cuda").to(dtype)
    x[0, :32] = 0  # all-zero block: scale byte 0, zero payload
    x[1, :32] = 448.0  # amax exactly at the e4m3 max
    x[2, :32] = -1.0
    x[2, 5] = 3.5
    q, scale = mxfp8_group_quant(x)
    assert scale[0, 0].item() == 0
    assert (q[0, :32].view(torch.uint8) == 0).all()
    # The block amax never clips: amax / scale <= 448.
    d = _dequant_scale(scale).repeat_interleave(32, dim=1)
    nonzero = d > 0
    assert ((x.float().abs() / d)[nonzero] <= 448.0).all()


def test_tiny_fp32_values():
    # Block amax below 448 * 2^-126 makes the dequant scale an fp32 subnormal.
    x = (torch.randn((16, 64), device="cuda") * 1e-38).float()
    q, scale = mxfp8_group_quant(x)
    ref_scale, exact_pow2 = _reference_scale(x)
    assert not ((scale != ref_scale) & ~exact_pow2).any()
    assert torch.equal(
        q.view(torch.uint8), _reference_values(x, scale).view(torch.uint8)
    )


def test_bitwise_matches_triton_kernels():
    mxfp = pytest.importorskip("triton_kernels.numerics_details.mxfp")
    for dtype in DTYPES:
        for shape in SHAPES:
            x = _make_input(shape, dtype, seed=1)
            # Include blocks whose amax is exactly 448 * 2^k.
            x[0, :32] = 448.0
            x[-1, -32:] = 1.75
            q, scale = mxfp8_group_quant(x)
            ref_q, ref_scale = mxfp.downcast_to_mxfp(x, torch.float8_e4m3fn, axis=1)
            assert torch.equal(scale, ref_scale.contiguous())
            assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))


def test_rejects_bad_hidden():
    x = torch.randn((4, 48), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(Exception, match="multiple of 32"):
        mxfp8_group_quant(x)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
