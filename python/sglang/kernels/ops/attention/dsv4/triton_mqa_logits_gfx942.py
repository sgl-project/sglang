"""gfx942 Triton replacement for aiter_fp4_paged_mqa_logits (DeepSeek V4.1 low-ratio indexer).

The ROCm low-ratio indexer only has a FlyDSL kernel built on the CDNA4-only
`v_mfma_scale_f32_16x16x128_f8f6f4`, which does not compile on gfx942.

Semantics (matching `dsv41_sparse.Indexer.scores`):
    logits[t, n] = sum_h relu(q[t, h, :] . k[n, :]) * weights[t, h]

The index-K pool layout is swizzled for MFMA loads:
    payload [pages, 1, 4, 64, 16] float4_e2m1fn_x2  stride (4096, 4096, 1024, 16, 1)
    scale   [pages, 1, 4, 64]     uint8             one ue8m0 per 32 elements
A row's 64 payload bytes are split into 4 segments x 16 bytes with the slot
dimension in between. Element d lives in byte d // 2, segment d // 32 (also its
scale block), offset (d // 2) % 16 within the segment.

Q is taken in bf16: the FP4 packing exists for CDNA4's native FP4 MFMA, which
gfx942 lacks. FP4 magnitudes need 3 mantissa bits, so the bf16 conversion of K
is exact.

Performance notes:
  1. The grid spans max_seq_len (the page-table width, not the row's length);
     tiles past the row length exit early instead of only masking loads.
  2. The head reduction is a [BLOCK_N, D] x [D, H] tl.dot so it runs on
     v_mfma_f32_16x16x16_bf16, not 32 cross-lane tl.sum reductions.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_DIM = 128


@triton.jit
def _e2m1_decode(nib):
    """4-bit e2m1 -> f32, computed arithmetically: a lookup table would need a
    device tensor whose first allocation can land inside CUDA graph capture.
    Codes 0..7 = 0, .5, 1, 1.5, 2, 3, 4, 6; the high bit is the sign."""
    code = nib & 7
    mag = tl.where(
        code < 2,
        0.5 * code.to(tl.float32),
        (1.0 + 0.5 * (code & 1).to(tl.float32))
        * tl.exp2(((code >> 1) - 1).to(tl.float32)),
    )
    return tl.where((nib >> 3) & 1 == 1, -mag, mag)


@triton.jit
def _mqa_logits_kernel(
    q_ptr,
    w_ptr,
    kpay_ptr,
    ksc_ptr,
    pt_ptr,
    slen_ptr,
    out_ptr,
    s_qt,
    s_qh,
    s_pt,
    s_out,
    kp0,
    kp2,
    kp3,
    ks0,
    ks2,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    t = tl.program_id(0)
    start = tl.program_id(1) * BLOCK_N
    slen = tl.load(slen_ptr + t)
    # the whole tile is past this row's length: exit (the grid spans the page
    # table width, usually far wider than the row)
    if start >= slen:
        return

    offs_n = start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < slen

    page = tl.load(
        pt_ptr + t.to(tl.int64) * s_pt + (offs_n // PAGE).to(tl.int64),
        mask=mask_n,
        other=0,
    ).to(tl.int64)
    slot = (offs_n % PAGE).to(tl.int64)

    offs_d = tl.arange(0, D)
    seg = offs_d // 32  # payload segment, also the scale block
    within = (offs_d // 2) % 16  # byte offset within the segment

    # [BLOCK_N, D]: d is innermost and bytes are contiguous within a segment
    base = page[:, None] * kp0 + slot[:, None] * kp3
    byte = tl.load(
        kpay_ptr + base + seg[None, :] * kp2 + within[None, :],
        mask=mask_n[:, None],
        other=0,
    ).to(tl.int32)
    nib = tl.where((offs_d % 2)[None, :] == 1, (byte >> 4) & 0xF, byte & 0xF)
    kval = _e2m1_decode(nib)

    exp = tl.load(
        ksc_ptr + page[:, None] * ks0 + seg[None, :] * ks2 + slot[:, None],
        mask=mask_n[:, None],
        other=127,
    ).to(tl.int32)
    kval = kval * tl.exp2((exp - 127).to(tl.float32))

    # load q as [D, H] directly (no tl.trans); q is small and cached
    offs_h = tl.arange(0, H)
    qt = tl.load(q_ptr + t * s_qt + offs_d[:, None] + offs_h[None, :] * s_qh)

    # FP4 magnitudes fit bf16 exactly
    scores = tl.dot(kval.to(tl.bfloat16), qt.to(tl.bfloat16))  # [BLOCK_N, H] fp32
    w = tl.load(w_ptr + t * H + offs_h).to(tl.float32)
    acc = tl.sum(tl.maximum(scores, 0.0) * w[None, :], axis=1)

    # prefill rectangles reach ~1000 rows x 500K columns, so the row offset must
    # be int64 (Triton specializes small Python ints to i32)
    tl.store(
        out_ptr + t.to(tl.int64) * s_out.to(tl.int64) + offs_n.to(tl.int64),
        acc,
        mask=mask_n,
    )


def triton_paged_mqa_logits(
    *,
    q: torch.Tensor,  # [T, H, D] bf16
    weights: torch.Tensor,  # [T, H] bf16
    k_payload: torch.Tensor,  # [pages, 1, 4, 64, 16]
    k_scale: torch.Tensor,  # [pages, 1, 4, 64] uint8
    page_table: torch.Tensor,  # [T, P] int32
    c4_seq_lens: torch.Tensor,  # [T] int32
    max_seq_len: int,
    page_size: int = 64,
    out: torch.Tensor = None,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    T, H, D = q.shape
    assert D == HEAD_DIM, D
    kp = k_payload.view(torch.uint8)
    ks = k_scale.view(torch.uint8)
    if c4_seq_lens.dtype != torch.int32:
        c4_seq_lens = c4_seq_lens.to(torch.int32)
    if out is None:
        out = torch.empty((T, max_seq_len), dtype=torch.float32, device=q.device)

    _mqa_logits_kernel[(T, triton.cdiv(max_seq_len, block_n))](
        q,
        weights,
        kp,
        ks,
        page_table,
        c4_seq_lens,
        out,
        q.stride(0),
        q.stride(1),
        page_table.stride(0),
        out.stride(0),
        kp.stride(0),
        kp.stride(2),
        kp.stride(3),
        ks.stride(0),
        ks.stride(2),
        H=H,
        D=D,
        PAGE=page_size,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# ---------------------------------------------------------------------------
# Candidate-only scoring for candidate-consumer layers.
#
# Instead of scoring every visible position with triton_paged_mqa_logits and then
# cutting the compact candidate row with gather_candidate_blocks, the grid spans
# the compact row: column c is position ids[c // CBS] * CBS + c % CBS, and
# unreachable columns (id -1 or position >= seq_len) are -inf, exactly as
# _gather_candidate_blocks_kernel writes them.
#
# Each position is scored with the same BLOCK_N-row tile, the same
# [BLOCK_N, D] x [D, H] tl.dot and the same relu * w head reduction as the full
# kernel; a row's result does not depend on the other rows of its tile, so the
# compact row is bitwise identical to the gathered one.
# ---------------------------------------------------------------------------


@triton.jit
def _cand_mqa_logits_kernel(
    q_ptr,
    w_ptr,
    kpay_ptr,
    ksc_ptr,
    pt_ptr,
    slen_ptr,
    ids_ptr,
    out_ptr,
    s_qt,
    s_qh,
    s_pt,
    s_ids,
    s_out,
    kp0,
    kp2,
    kp3,
    ks0,
    ks2,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CBS: tl.constexpr,
):
    t = tl.program_id(0)
    c0 = tl.program_id(1) * BLOCK_N
    slen = tl.load(slen_ptr + t)

    offs_c = c0 + tl.arange(0, BLOCK_N)
    blk = tl.load(ids_ptr + t.to(tl.int64) * s_ids + offs_c // CBS)
    offs_n = blk * CBS + offs_c % CBS
    mask_n = (blk >= 0) & (offs_n < slen)
    out_row = out_ptr + t.to(tl.int64) * s_out.to(tl.int64) + offs_c.to(tl.int64)
    # the whole tile is unreachable (the -1 tail of ids): write -inf only
    if tl.max(mask_n.to(tl.int32), axis=0) == 0:
        tl.store(out_row, tl.full([BLOCK_N], float("-inf"), tl.float32))
        return
    offs_n = tl.where(mask_n, offs_n, 0)

    page = tl.load(
        pt_ptr + t.to(tl.int64) * s_pt + (offs_n // PAGE).to(tl.int64),
        mask=mask_n,
        other=0,
    ).to(tl.int64)
    slot = (offs_n % PAGE).to(tl.int64)

    offs_d = tl.arange(0, D)
    seg = offs_d // 32
    within = (offs_d // 2) % 16

    base = page[:, None] * kp0 + slot[:, None] * kp3
    byte = tl.load(
        kpay_ptr + base + seg[None, :] * kp2 + within[None, :],
        mask=mask_n[:, None],
        other=0,
    ).to(tl.int32)
    nib = tl.where((offs_d % 2)[None, :] == 1, (byte >> 4) & 0xF, byte & 0xF)
    kval = _e2m1_decode(nib)

    exp = tl.load(
        ksc_ptr + page[:, None] * ks0 + seg[None, :] * ks2 + slot[:, None],
        mask=mask_n[:, None],
        other=127,
    ).to(tl.int32)
    kval = kval * tl.exp2((exp - 127).to(tl.float32))

    offs_h = tl.arange(0, H)
    qt = tl.load(q_ptr + t * s_qt + offs_d[:, None] + offs_h[None, :] * s_qh)

    scores = tl.dot(kval.to(tl.bfloat16), qt.to(tl.bfloat16))  # [BLOCK_N, H] fp32
    w = tl.load(w_ptr + t * H + offs_h).to(tl.float32)
    acc = tl.sum(tl.maximum(scores, 0.0) * w[None, :], axis=1)
    acc = tl.where(mask_n, acc, float("-inf"))
    tl.store(out_row, acc)


def triton_candidate_mqa_logits(
    *,
    q: torch.Tensor,  # [T, H, D] bf16
    weights: torch.Tensor,  # [T, H] bf16
    k_payload: torch.Tensor,  # [pages, 1, 4, 64, 16]
    k_scale: torch.Tensor,  # [pages, 1, 4, 64] uint8
    page_table: torch.Tensor,  # [T, P] int32, the guarded table the full scorer uses
    c4_seq_lens: torch.Tensor,  # [T] int32
    ids: torch.Tensor,  # [T, topk_blocks] int32, -1 padded
    block_size: int,  # candidate block size in positions
    page_size: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """The compact candidate row [T, topk_blocks * block_size] fp32, equal to
    gather_candidate_blocks(triton_paged_mqa_logits(...), seq_lens, ids)."""
    T, H, D = q.shape
    assert D == HEAD_DIM, D
    assert ids.dtype == torch.int32 and ids.stride(1) == 1 and ids.shape[0] == T
    width = ids.shape[1] * block_size
    assert width % block_n == 0 and block_n % block_size == 0, (
        width,
        block_n,
        block_size,
    )
    kp = k_payload.view(torch.uint8)
    ks = k_scale.view(torch.uint8)
    if c4_seq_lens.dtype != torch.int32:
        c4_seq_lens = c4_seq_lens.to(torch.int32)
    out = torch.empty((T, width), dtype=torch.float32, device=q.device)
    _cand_mqa_logits_kernel[(T, width // block_n)](
        q,
        weights,
        kp,
        ks,
        page_table,
        c4_seq_lens,
        ids,
        out,
        q.stride(0),
        q.stride(1),
        page_table.stride(0),
        ids.stride(0),
        out.stride(0),
        kp.stride(0),
        kp.stride(2),
        kp.stride(3),
        ks.stride(0),
        ks.stride(2),
        H=H,
        D=D,
        PAGE=page_size,
        BLOCK_N=block_n,
        CBS=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
