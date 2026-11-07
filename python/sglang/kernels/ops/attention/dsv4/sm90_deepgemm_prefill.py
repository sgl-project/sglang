"""Gather Hopper Indexer operands and score a request with DeepGEMM."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .fp4_indexer import (
    INDEX_HEAD_DIM,
)
from .fp4_indexer import INDEX_K_PAYLOAD_BYTES as PAYLOAD_BYTES
from .fp4_indexer import INDEX_K_SCALE_BYTES as SCALE_BYTES
from .fp4_indexer import (
    _e2m1_decode,
)

FP8_E4M3_MAX = 448.0


@triton.jit
def _unpack_fp4_index_keys_to_fp8_kernel(
    slots_ptr,
    table_ptr,
    out_ptr,
    page_size,
    row_stride,
    HALF_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Decode block-scaled E2M1 values directly into E4M3."""
    row = tl.program_id(0)
    offs_i = tl.arange(0, HALF_D)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    page = slot // page_size
    off = slot % page_size
    row_base = page * row_stride
    pay = tl.load(table_ptr + row_base + off * PAYLOAD_BYTES + offs_i)
    scale_block = offs_i // 16
    exps = tl.load(
        table_ptr
        + row_base
        + page_size * PAYLOAD_BYTES
        + off * SCALE_BYTES
        + scale_block
    )
    scale = tl.exp2(exps.to(tl.float32) - 127.0)
    low = tl.clamp(_e2m1_decode(pay & 0x0F) * scale, -FP8_MAX, FP8_MAX).to(
        out_ptr.dtype.element_ty
    )
    high = tl.clamp(
        _e2m1_decode((pay >> 4) & 0x0F) * scale,
        -FP8_MAX,
        FP8_MAX,
    ).to(out_ptr.dtype.element_ty)
    out = out_ptr + row * (HALF_D * 2)
    tl.store(out + 2 * offs_i, low)
    tl.store(out + 2 * offs_i + 1, high)


def unpack_fp4_index_keys_to_fp8(
    slots: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Gather FP4 index-K rows and decode them directly into E4M3."""
    assert slots.dim() == 1
    assert table.dtype == torch.uint8 and table.dim() == 2
    assert table.stride(1) == 1 and table.shape[1] == page_size * 68
    slots = slots.to(torch.int64).contiguous()
    values = torch.empty(
        (slots.shape[0], INDEX_HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=slots.device,
    )
    if slots.numel() > 0:
        _unpack_fp4_index_keys_to_fp8_kernel[(slots.shape[0],)](
            slots,
            table,
            values,
            page_size,
            table.stride(0),
            HALF_D=INDEX_HEAD_DIM // 2,
            FP8_MAX=FP8_E4M3_MAX,
            num_warps=4,
        )
    return values


@triton.jit
def _quantize_bf16_index_queries_fp8_kernel(
    q_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_ob,
    stride_oh,
    H: tl.constexpr,
    D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    q = tl.load(
        q_ptr + row * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :]
    ).to(tl.float32)
    q_fp8 = tl.clamp(q, -FP8_MAX, FP8_MAX).to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + row * stride_ob + offs_h[:, None] * stride_oh + offs_d[None, :],
        q_fp8,
    )


def quantize_bf16_index_queries_fp8(
    q: torch.Tensor,
) -> torch.Tensor:
    """Cast each BF16 query head to E4M3 for one K=128 tensor-core dot."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    q = q.contiguous()
    rows, heads, _ = q.shape
    values = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    if rows > 0:
        _quantize_bf16_index_queries_fp8_kernel[(rows,)](
            q,
            values,
            q.stride(0),
            q.stride(1),
            values.stride(0),
            values.stride(1),
            H=heads,
            D=INDEX_HEAD_DIM,
            FP8_MAX=FP8_E4M3_MAX,
            num_warps=8,
        )
    return values


def deepgemm_fp8_index_logits_prefill(
    q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    lens: torch.Tensor,
    *,
    key_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score one request's contiguous K; each row sees ``keys[:lens[row]]``.

    ``q`` is [rows, 32|64, 128] E4M3, ``keys`` is [keys, 128] E4M3,
    ``weights`` is [rows, heads], and lengths are nonnegative and <= key count.
    The shared FP4-to-FP8 conversion already applies block scales, so runtime
    passes unit FP32 key scales. No MXFP4 or packed UE8M0 operand reaches DG.

    Each request has its own gathered K, hence all starts are zero. Use full
    width plus cleaning: candidate reduction needs -inf causal tails, whereas
    the compact-output API forbids ``clean_logits=True``. DG's padded backing
    stride also satisfies ragged top-k's vector-read alignment.
    """
    assert q.ndim == 3 and q.dtype == torch.float8_e4m3fn
    rows, heads, dim = q.shape
    assert heads in (32, 64) and dim == 128
    assert keys.ndim == 2 and keys.shape[1] == dim
    assert keys.dtype == torch.float8_e4m3fn
    assert weights.shape == (rows, heads) and lens.shape == (rows,)
    width = keys.shape[0]
    if rows == 0 or width == 0:
        return torch.empty((rows, width), device=q.device, dtype=torch.float32)
    import deep_gemm

    if not hasattr(deep_gemm, "fp8_mqa_logits"):
        raise RuntimeError("SM90 prefill needs DeepGEMM's fp8_mqa_logits API")
    if key_scales is None:
        key_scales = torch.ones(width, device=keys.device, dtype=torch.float32)
    assert key_scales.dtype == torch.float32 and key_scales.shape == (width,)
    ends = lens.to(torch.int32).contiguous()
    return deep_gemm.fp8_mqa_logits(
        q.contiguous(),
        (keys.contiguous(), key_scales.contiguous()),
        # Match the head-weight input of the BF16-rounding scorer, but DG's
        # SM90 kernel requires FP32 storage for those values.
        weights.to(torch.bfloat16).float().contiguous(),
        torch.zeros_like(ends),
        ends,
        True,
        0,
    )
