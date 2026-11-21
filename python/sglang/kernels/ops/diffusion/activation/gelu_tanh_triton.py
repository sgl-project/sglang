"""In-place, bit-exact tanh-approximate GELU for large bf16 activations.

``F.gelu(x, approximate="tanh")`` on the Wan2.2 FFN intermediate
(``[90000, 13824]`` bf16 at 720p) is slow on gfx1250: TensorIterator splits
any tensor whose byte offsets pass 2**31 into two launches, and each reaches
only ~2.2 TB/s. This kernel uses int64 offsets, so it is one launch.

It computes aten's ``GeluCUDAKernelImpl`` formula in fp32. Checked over all
65536 bf16 inputs, that formula differs from aten only for ``x < -4``
(Triton's tanh rounds to -1 where aten's does not) and in NaN payloads; the
former get an exp-based form that matches aten there, and non-finite inputs
get aten's canonical NaN. The result is bit-identical to aten for every bf16
input (pinned by a unit test).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK = 16384
_NUM_WARPS = 8


@triton.jit
def _gelu_tanh_bf16_inplace_kernel(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x_bf16 = tl.load(x_ptr + offs, mask=mask)
    x = x_bf16.to(tl.float32)
    x_cube = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x_cube)
    y_main = 0.5 * x * (1.0 + libdevice.tanh(inner))
    # For x < -4, aten's 1 + tanh(inner) lies on the 2**-24 grid of tanh near -1,
    # where Triton's tanh rounds to -1; 2 * exp(2 * inner) rounded to that grid matches.
    one_plus_tanh = (
        libdevice.rint(tl.exp2(2.8853900817779268 * inner) * 33554432.0)
        * 5.9604644775390625e-08
    )
    y = tl.where(x < -4.0, 0.5 * x * one_plus_tanh, y_main)
    y_bits = y.to(tl.bfloat16).to(tl.int16, bitcast=True)
    # aten returns the canonical quiet NaN for every NaN input and for -inf.
    y_bits = tl.where((x != x) | (x == float("-inf")), 0x7FC0, y_bits)
    tl.store(x_ptr + offs, y_bits.to(tl.bfloat16, bitcast=True), mask=mask)


def _torch_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def gelu_tanh_inplace_(x: torch.Tensor) -> torch.Tensor:
    """``x.copy_(F.gelu(x, approximate="tanh"))``, bit-identical to aten."""
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        return x.copy_(_torch_gelu_tanh(x))
    n = x.numel()
    if n:
        _gelu_tanh_bf16_inplace_kernel[(triton.cdiv(n, _BLOCK),)](
            x, n, BLOCK=_BLOCK, num_warps=_NUM_WARPS
        )
    return x
