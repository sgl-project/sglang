# SPDX-License-Identifier: Apache-2.0
"""Interleaved RoPE with the eager FP32 matrix-product rounding boundaries."""

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _matrix_rope_kernel(
    x,
    rope,
    out,
    pairs,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BATCH_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    PAIR_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < pairs
    token = idx // (HEADS * HALF_DIM)
    pos = (
        token // SEQ * BATCH_STRIDE
        + token % SEQ * SEQ_STRIDE
        + idx % HALF_DIM * PAIR_STRIDE
    )
    # preserve the input cast before FP32 products when fusing a dtype conversion
    x0 = tl.load(x + 2 * idx, valid, other=0).to(out.dtype.element_ty).to(tl.float32)
    x1 = (
        tl.load(x + 2 * idx + 1, valid, other=0).to(out.dtype.element_ty).to(tl.float32)
    )
    r00 = tl.load(rope + pos, valid, other=0)
    r01 = tl.load(rope + pos + COL_STRIDE, valid, other=0)
    r10 = tl.load(rope + pos + ROW_STRIDE, valid, other=0)
    r11 = tl.load(rope + pos + ROW_STRIDE + COL_STRIDE, valid, other=0)
    # do not contract the products: eager materializes them before the pair sum
    y0 = x0 * r00 + x1 * r01
    y1 = x0 * r10 + x1 * r11
    tl.store(out + 2 * idx, y0, valid)
    tl.store(out + 2 * idx + 1, y1, valid)


def _fake_apply_matrix_rope(
    x: torch.Tensor, rope: torch.Tensor, dtype: torch.dtype | None = None
) -> torch.Tensor:
    return torch.empty_like(x, dtype=dtype)


@register_custom_op(
    op_name="diffusion_apply_matrix_rope",
    mutates_args=[],
    fake_impl=_fake_apply_matrix_rope,
)
def apply_matrix_rope(
    x: torch.Tensor, rope: torch.Tensor, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Apply FP32 matrices to [B,S,H,D], optionally casting x to dtype first.

    The result has the cast dtype; the input rounding precedes FP32 products.
    """
    dtype = x.dtype if dtype is None else dtype
    if not (
        x.is_cuda
        and torch.version.hip is None
        and x.ndim == 4
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise ValueError(
            "matrix RoPE requires contiguous FP16/BF16/FP32 CUDA [B,S,H,D]"
        )
    batch, seq, heads, dim = x.shape
    if not (
        dim % 2 == 0
        and rope.device == x.device
        and rope.dtype == torch.float32
        and rope.ndim == 6
        and rope.shape[0] in (1, batch)
        and rope.shape[1:] == (seq, 1, dim // 2, 2, 2)
    ):
        raise ValueError("matrix RoPE requires matching FP32 [B or 1,S,1,D/2,2,2]")
    output = torch.empty_like(x, dtype=dtype)
    pairs = x.numel() // 2
    if pairs:
        with torch.cuda.device(x.device):
            _matrix_rope_kernel[(triton.cdiv(pairs, 512),)](
                x,
                rope,
                output,
                pairs,
                SEQ=seq,
                HEADS=heads,
                HALF_DIM=dim // 2,
                BATCH_STRIDE=rope.stride(0) if rope.shape[0] != 1 else 0,
                SEQ_STRIDE=rope.stride(1),
                PAIR_STRIDE=rope.stride(3),
                ROW_STRIDE=rope.stride(4),
                COL_STRIDE=rope.stride(5),
                BLOCK=512,
                enable_fp_fusion=False,
            )
    return output
