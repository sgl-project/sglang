"""Gather Hopper Indexer operands and score a request with DeepGEMM."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .fp4_indexer import (
    INDEX_HEAD_DIM,
    INDEX_K_PAYLOAD_BYTES,
    INDEX_K_SCALE_BYTES,
    INDEX_K_SLOT_BYTES,
    _e2m1_decode_bits,
)

FP8_E4M3_MAX = 448.0


@triton.jit
def _gather_fp4_index_keys_as_fp8_kernel(
    slots_ptr,
    table_ptr,
    out_ptr,
    num_slots,
    slot_stride,
    row_stride,
    PAGE_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    FP8_MAX: tl.constexpr,
    PAYLOAD_BYTES: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    valid = rows < num_slots
    slots = tl.load(slots_ptr + rows * slot_stride, valid, 0).to(tl.int64)
    page = slots // PAGE_SIZE
    slot = slots % PAGE_SIZE
    base = table_ptr + page * row_stride

    pairs = tl.arange(0, PAYLOAD_BYTES)
    payload = tl.load(
        base[:, None] + slot[:, None] * PAYLOAD_BYTES + pairs[None, :],
        valid[:, None],
        0,
    )
    groups = tl.arange(0, SCALE_BYTES)
    exponents = tl.load(
        base[:, None]
        + PAGE_SIZE * PAYLOAD_BYTES
        + slot[:, None] * SCALE_BYTES
        + groups[None, :],
        valid[:, None],
        127,
    ).to(tl.int32)

    # Only scales in between can change the E4M3 result. At 114 even
    # 6 * 2**-13 rounds to zero; at 137 even 0.5 * 2**10 saturates.
    # This also avoids overflowing FP32 for the largest finite UE8M0 scales.
    scale = (tl.clamp(exponents, 114, 137) << 23).to(tl.float32, bitcast=True)
    codes = tl.interleave(payload & 0x0F, payload >> 4)
    scale = tl.reshape(
        tl.broadcast_to(
            scale[:, :, None],
            (ROWS, SCALE_BYTES, HEAD_DIM // SCALE_BYTES),
        ),
        (ROWS, HEAD_DIM),
    )
    values = _e2m1_decode_bits(codes) * scale
    values = tl.clamp(values, -FP8_MAX, FP8_MAX).to(out_ptr.dtype.element_ty)
    dims = tl.arange(0, HEAD_DIM)
    tl.store(out_ptr + rows[:, None] * HEAD_DIM + dims[None, :], values, valid[:, None])


def gather_fp4_index_keys_as_fp8(
    slots: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Gather paged E2M1 K, apply its UE8M0 scales, and saturate to finite E4M3."""
    assert slots.dim() == 1 and slots.dtype in (torch.int32, torch.int64)
    assert table.dtype == torch.uint8 and table.dim() == 2
    assert table.stride(1) == 1 and table.shape[1] == page_size * INDEX_K_SLOT_BYTES
    num_slots = slots.numel()
    values = torch.empty(
        (num_slots, INDEX_HEAD_DIM), dtype=torch.float8_e4m3fn, device=slots.device
    )
    if num_slots:
        rows = 8
        _gather_fp4_index_keys_as_fp8_kernel[(triton.cdiv(num_slots, rows),)](
            slots,
            table,
            values,
            num_slots,
            slots.stride(0),
            table.stride(0),
            PAGE_SIZE=page_size,
            ROWS=rows,
            FP8_MAX=FP8_E4M3_MAX,
            PAYLOAD_BYTES=INDEX_K_PAYLOAD_BYTES,
            SCALE_BYTES=INDEX_K_SCALE_BYTES,
            HEAD_DIM=INDEX_HEAD_DIM,
            num_warps=4,
        )
    return values


@triton.jit
def _cast_index_queries_to_fp8_kernel(
    q_ptr, out_ptr, numel, TILE: tl.constexpr, FP8_MAX: tl.constexpr
):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    q = tl.load(q_ptr + offsets, offsets < numel, 0).to(tl.float32)
    q = tl.clamp(q, -FP8_MAX, FP8_MAX).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offsets, q, offsets < numel)


def cast_index_queries_to_fp8(q: torch.Tensor) -> torch.Tensor:
    """Saturating cast of FP4 fake-quantized BF16 queries, with no new scales."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    q = q.contiguous()
    out = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    if q.numel():
        tile = 1024
        _cast_index_queries_to_fp8_kernel[(triton.cdiv(q.numel(), tile),)](
            q, out, q.numel(), TILE=tile, FP8_MAX=FP8_E4M3_MAX, num_warps=4
        )
    return out


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
