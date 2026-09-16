"""Dots-specific FP8 helpers for absorbed MLA batched matmuls."""

from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import ceil_align

_FP8_MAX = 224.0 if is_fp8_fnuz() else torch.finfo(torch.float8_e4m3fn).max


@triton.jit
def _per_token_group_quant_einsum_fp8(
    x_ptr,
    x_q_ptr,
    x_s_ptr,
    group_size,
    num_b,
    num_k,
    total_rows,
    x_stride_m,
    x_stride_b,
    x_q_stride_m,
    x_q_stride_b,
    x_s_stride_m,
    x_s_stride_b,
    x_s_stride_g,
    eps,
    quant_min,
    quant_max,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    group_id = tl.program_id(1)
    m_ids = row_ids // num_b
    b_ids = row_ids - m_ids * num_b
    k_offsets = tl.arange(0, BLOCK_K)
    k_ids = group_id * group_size + k_offsets
    mask = (row_ids[:, None] < total_rows) & (
        (k_offsets[None, :] < group_size) & (k_ids[None, :] < num_k)
    )
    x_ptrs = (
        x_ptr
        + m_ids[:, None] * x_stride_m
        + b_ids[:, None] * x_stride_b
        + k_ids[None, :]
    )
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), eps)
    scale = absmax / quant_max
    quant = tl.clamp(x / scale[:, None], quant_min, quant_max).to(
        x_q_ptr.dtype.element_ty
    )
    q_ptrs = (
        x_q_ptr
        + m_ids[:, None] * x_q_stride_m
        + b_ids[:, None] * x_q_stride_b
        + k_ids[None, :]
    )
    s_ptrs = (
        x_s_ptr + m_ids * x_s_stride_m + b_ids * x_s_stride_b + group_id * x_s_stride_g
    )
    tl.store(q_ptrs, quant, mask=mask)
    tl.store(s_ptrs, scale, mask=row_ids < total_rows)


def per_token_group_quant_einsum_fp8(
    x: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[m, b, k]`` in the scale layout required by FP8 einsum."""
    assert x.ndim == 3 and x.stride(-1) == 1
    assert group_size == 128
    m, b, k = x.shape
    num_groups = (k + group_size - 1) // group_size
    aligned_m = ceil_align(m, 4)
    x_q = x.new_empty((m, b, k), dtype=torch.float8_e4m3fn)
    scale_storage = x.new_empty((b, num_groups, aligned_m), dtype=torch.float32)
    x_s = scale_storage.permute(2, 0, 1)[:m]
    if m == 0 or b == 0 or num_groups == 0:
        return x_q, x_s
    block_m = 16
    block_k = triton.next_power_of_2(group_size)
    _per_token_group_quant_einsum_fp8[(triton.cdiv(m * b, block_m), num_groups)](
        x,
        x_q,
        x_s,
        group_size,
        b,
        k,
        m * b,
        x.stride(0),
        x.stride(1),
        x_q.stride(0),
        x_q.stride(1),
        x_s.stride(0),
        x_s.stride(1),
        x_s.stride(2),
        eps,
        -_FP8_MAX,
        _FP8_MAX,
        block_m,
        block_k,
        num_warps=4,
        num_stages=1,
    )
    return x_q, x_s
