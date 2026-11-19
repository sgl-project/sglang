# SPDX-License-Identifier: Apache-2.0
"""Split FP32 weights into three BF16 parts for BF16x3 tensor-core GEMMs."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _split_fp32_to_bf16x3_kernel(w_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    w = tl.load(w_ptr + offs, mask=mask)
    hi = w.to(tl.bfloat16)
    residual = w - hi.to(tl.float32)
    mid = residual.to(tl.bfloat16)
    lo = (residual - mid.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + offs, hi, mask=mask)
    tl.store(out_ptr + numel + offs, mid, mask=mask)
    tl.store(out_ptr + 2 * numel + offs, lo, mask=mask)


def split_fp32_to_bf16x3(
    w: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """[N, K] fp32 -> [3N, K] bf16 (hi, mid, lo) blocks, each rounding what remains.

    hi + mid + lo == w for 2**-110 <= |w| < 2**128 - 2**119: below that the last
    remainder drops under BF16's subnormals, above it hi rounds to infinity.
    """
    if not (
        w.is_cuda and w.ndim == 2 and w.dtype == torch.float32 and w.is_contiguous()
    ):
        raise ValueError("w must be a contiguous CUDA fp32 [N, K] tensor")
    n, k = w.shape
    if out is None:
        out = torch.empty(3 * n, k, dtype=torch.bfloat16, device=w.device)
    elif not (
        out.shape == (3 * n, k)
        and out.dtype == torch.bfloat16
        and out.device == w.device
        and out.is_contiguous()
    ):
        raise ValueError("out must be a contiguous bf16 [3N, K] tensor on w's device")
    block = 1024
    _split_fp32_to_bf16x3_kernel[(triton.cdiv(w.numel(), block),)](
        w, out, w.numel(), BLOCK=block
    )
    return out
