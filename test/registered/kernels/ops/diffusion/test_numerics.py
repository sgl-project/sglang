"""``sglang.kernels.numerics``: the inline-asm primitives behind the bit-exact
kernels.

ROCm-only: the CUDA build hands PTX to ``tl.inline_asm_elementwise``, while the
AMD backend assembles GCN mnemonics with ``v`` constraints and rejects the PTX
form with ``couldn't allocate output register for constraint 'f'``. Compiling
each primitive is the regression test; the assertions pin it to torch's math.
"""

import sys

import pytest
import torch
import triton
import triton.language as tl

from sglang.kernels import numerics
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="exercises the GCN inline-asm path",
)

DEVICE = "cuda"
BLOCK = 256


@triton.jit
def _mul_kernel(x_ptr, y_ptr, out_ptr):
    offs = tl.arange(0, 256)
    x = tl.load(x_ptr + offs).to(tl.float32)
    y = tl.load(y_ptr + offs).to(tl.float32)
    tl.store(out_ptr + offs, numerics.mul_rn_f32(x, y))


@triton.jit
def _div_kernel(x_ptr, y_ptr, out_ptr):
    offs = tl.arange(0, 256)
    x = tl.load(x_ptr + offs).to(tl.float32)
    y = tl.load(y_ptr + offs).to(tl.float32)
    tl.store(out_ptr + offs, numerics.div_rn_f32(x, y))


@triton.jit
def _rsqrt_kernel(x_ptr, out_ptr, CUDA_STYLE: tl.constexpr):
    offs = tl.arange(0, 256)
    x = tl.load(x_ptr + offs).to(tl.float32)
    result = numerics.cuda_rsqrtf(x) if CUDA_STYLE else numerics.rsqrt_approx_f32(x)
    tl.store(out_ptr + offs, result)


def _run_binary(kernel, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    kernel[(1,)](x, y, out, num_warps=4)
    torch.cuda.synchronize()
    return out


def _run_rsqrt(x: torch.Tensor, cuda_style: bool) -> torch.Tensor:
    out = torch.empty_like(x)
    _rsqrt_kernel[(1,)](x, out, cuda_style, num_warps=4)
    torch.cuda.synchronize()
    return out


def _binary_inputs():
    g = torch.Generator(device=DEVICE).manual_seed(0)
    x = torch.randn(BLOCK, device=DEVICE, generator=g)
    y = torch.randn(BLOCK, device=DEVICE, generator=g)
    return x, y


def _positive_inputs():
    g = torch.Generator(device=DEVICE).manual_seed(1)
    return torch.rand(BLOCK, device=DEVICE, generator=g) + 0.25


def test_picks_the_amd_asm_family() -> None:
    assert numerics._AMD is (torch.version.hip is not None)


def test_mul_rn_f32_is_a_correctly_rounded_multiply() -> None:
    x, y = _binary_inputs()
    # A product that only differs from the fused one once it is rounded.
    x[0], y[0] = -0.1484375, 14.25
    x[1], y[1] = 1.0000001192092896, 1.0000001192092896
    assert torch.equal(_run_binary(_mul_kernel, x, y), x * y)


def test_div_rn_f32_is_ieee_division() -> None:
    x, y = _binary_inputs()
    # A subnormal quotient and a near-overflow one, both correctly rounded.
    x[0], y[0] = 1e-38, 3.0
    x[1], y[1] = 3.0e38, 2.0
    assert torch.equal(_run_binary(_div_kernel, x, y), x / y)


def test_rsqrt_approximation_stays_within_its_error_bound() -> None:
    x = _positive_inputs()
    got = _run_rsqrt(x, cuda_style=False)
    assert torch.isfinite(got).all()
    rel = ((got - torch.rsqrt(x)).abs() / torch.rsqrt(x)).max().item()
    assert rel < 1e-6, rel


def test_cuda_rsqrtf_rescales_subnormals() -> None:
    x = _positive_inputs()
    # Inputs fine enough to be subnormal; the rescaled path has to stay accurate.
    x[:32] = torch.arange(1, 33, device=DEVICE, dtype=torch.float32) * 1e-40
    got = _run_rsqrt(x, cuda_style=True)
    assert torch.isfinite(got).all()
    rel = ((got - torch.rsqrt(x)).abs() / torch.rsqrt(x)).max().item()
    assert rel < 1e-5, rel


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
