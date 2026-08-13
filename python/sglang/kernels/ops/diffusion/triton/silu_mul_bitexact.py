# SPDX-License-Identifier: Apache-2.0
"""Bit-exact fused ``silu(a) * b`` over two same-shape tensors.

For SwiGLU MLPs whose gate/up projections are separate GEMMs (so the
concatenated-input ``silu_and_mul`` kernels don't apply without an extra
full-width ``cat`` pass), this fuses the eager pair

    ``s = F.silu(a)`` (one kernel) ``out = s * b`` (another kernel)

into one pass while reproducing both aten bf16 rounding boundaries:
``silu`` is a single aten op (fp32 opmath, one round), the multiply rounds
once more.  ``tl.sigmoid`` lowers to the same fp32 sigmoid aten uses, which
makes the replication exact (verified ``torch.equal`` on 1M random bf16
values); callers still verify the first call and fall back on mismatch.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.triton.numerics import round_bf16_to_fp32
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _silu_mul_kernel(
    out_ptr,
    a_ptr,
    b_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = round_bf16_to_fp32(a * tl.sigmoid(a))
    tl.store(out_ptr + offs, s * b, mask=mask)  # store rounds the multiply


def can_use_fused_silu_mul(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.dtype is torch.bfloat16
        and b.dtype is torch.bfloat16
        and a.is_cuda
        and b.is_cuda
        and a.device == b.device
        and a.shape == b.shape
        and a.is_contiguous()
        and b.is_contiguous()
        and a.numel() > 0
    )


def _fake_silu_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a)


@register_custom_op(
    op_name="triton_fused_silu_mul_bitexact",
    mutates_args=[],
    fake_impl=_fake_silu_mul,
)
def fused_silu_mul_bitexact(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``silu(a) * b``, bit-exact vs the eager two-kernel chain."""
    out = torch.empty_like(a)
    numel = a.numel()
    with torch.cuda.device(a.device):
        _silu_mul_kernel[(triton.cdiv(numel, 1024),)](
            out,
            a,
            b,
            numel,
            BLOCK=1024,
        )
    return out
