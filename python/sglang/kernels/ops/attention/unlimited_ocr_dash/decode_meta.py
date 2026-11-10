"""Fused one-token decode metadata for page isolation and Page2D RoPE.

One kernel updates page and coordinate state, another builds the candidate mask.
PyTorch remains the reference, prefill, and multi-token-decode path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Triton @jit bodies may only read module globals created as `tl.constexpr(...)`.
_MM_PAD_SHIFT = tl.constexpr(1_000_000)
_INT16_MAX_C = tl.constexpr(32767)


@triton.jit
def _decode_meta_kernel(
    ids_ptr,  # [R] int64/int32  this step's token per request
    bt_ptr,  # [R, W] int64     logical -> physical page
    seq_ptr,  # [R] int32        sequence length INCLUDING this token
    slot_ptr,  # [R] int64        KV slot this token was written to
    row_ptr,  # [R] int64        coordinate-state row of the request
    # page-isolation per-slot state
    s_page_ptr,
    s_mark_ptr,
    s_run_ptr,
    s_pp_ptr,
    # coordinate per-row state
    c_mark_ptr,
    c_pp_ptr,
    c_cnt_ptr,
    # outputs
    q_page_ptr,  # [R] int32
    tok_ptr,  # [R] int64
    page_ptr,  # [R] int64
    stride_bt_r,
    stride_cnt_r,
    image_token_id,
    page_token_id,
    per_page,
    n_slots,
    max_pages,
    bt_w,
    HAS_MARKER: tl.constexpr,
    BS: tl.constexpr,
):
    r = tl.program_id(0)
    tok_id = tl.load(ids_ptr + r)
    is_img = (tok_id >= _MM_PAD_SHIFT) | (tok_id == image_token_id)
    if HAS_MARKER:
        is_mark = tok_id == page_token_id
    else:
        is_mark = False

    seq = tl.load(seq_ptr + r).to(tl.int32)
    # --- carry-in: the slot holding this request's PREVIOUS token -------------
    prev_pos = seq - 2  # seq includes this token
    valid = prev_pos >= 0
    p = tl.maximum(prev_pos, 0)
    blk = tl.minimum(p // BS, bt_w - 1)
    phys = tl.load(bt_ptr + r * stride_bt_r + blk).to(tl.int64)
    prev_slot = tl.minimum(tl.maximum(phys * BS + (p % BS), 0), n_slots - 1)
    prev_slot = tl.where(valid, prev_slot, 0)
    prev_m = tl.where(valid, tl.load(s_mark_ptr + prev_slot), 0).to(tl.int32)
    prev_i = tl.where(valid, tl.load(s_run_ptr + prev_slot), 0).to(tl.int32)
    prev_pp = tl.where(valid, tl.load(s_pp_ptr + prev_slot), 0).to(tl.int32)

    marker_cum = prev_m + is_mark.to(tl.int32)
    img_run = prev_i + is_img.to(tl.int32)

    # --- page attribution (page_state.step, decode branch) -------------------
    img_page = tl.where(marker_cum > 0, marker_cum - 1, (img_run - 1) // per_page)
    img_page = tl.where(is_img, img_page, -1)
    pp = tl.maximum(tl.where(is_img, marker_cum, 0), prev_pp)
    # Decode tokens always take the RESPONSE numbering (raw - prompt_pages).
    resp = marker_cum - 1 - pp
    resp = tl.where(resp >= 0, resp, -1)
    q_page = tl.where(is_img, img_page, resp)
    tl.store(q_page_ptr + r, q_page)

    # --- per-slot writes -----------------------------------------------------
    slot = tl.minimum(tl.maximum(tl.load(slot_ptr + r), 0), n_slots - 1)
    tl.store(s_page_ptr + slot, img_page)
    tl.store(s_mark_ptr + slot, marker_cum)
    tl.store(s_run_ptr + slot, img_run)
    tl.store(s_pp_ptr + slot, pp)

    # Clear unused slots in the final partial KV block; stale image-page state
    # could otherwise misclassify the newest block.
    last_blk = tl.minimum(tl.maximum((seq - 1) // BS, 0), bt_w - 1)
    tail_phys = tl.load(bt_ptr + r * stride_bt_r + last_blk).to(tl.int64)
    rem = seq % BS
    off_b = tl.arange(0, BS)
    stale = (off_b >= rem) & (rem > 0)
    tail = tl.minimum(tl.maximum(tail_phys * BS + off_b, 0), n_slots - 1)
    tl.store(s_page_ptr + tail, tl.full((BS,), -1, tl.int32), mask=stale)

    # --- page2d coordinates (page2d_rope.step, decode branch) ----------------
    row = tl.load(row_ptr + r)
    c_m = tl.load(c_mark_ptr + row).to(tl.int32) + is_mark.to(tl.int32)
    c_p = tl.maximum(tl.where(is_img, c_m, 0), tl.load(c_pp_ptr + row).to(tl.int32))
    # An extra generated <PAGE> stays on the last real page, as in training.
    c_resp = tl.minimum(tl.maximum(c_m - c_p, 0), c_p)
    cpage = tl.minimum(tl.maximum(c_resp, 0), max_pages - 1).to(tl.int64)
    cnt_at = tl.load(c_cnt_ptr + row * stride_cnt_r + cpage).to(tl.int64)
    # This step contributes exactly one token to that page bucket, so the
    # running count IS the token's coordinate (inc == 1 -> cnt + 1 - 1).
    tl.store(tok_ptr + r, cnt_at)
    tl.store(page_ptr + r, cpage)
    tl.store(c_cnt_ptr + row * stride_cnt_r + cpage, (cnt_at + 1).to(tl.int32))
    tl.store(c_mark_ptr + row, c_m)
    tl.store(c_pp_ptr + row, c_p)


@triton.jit
def _decode_allow_kernel(
    bt_ptr,  # [R, W] int64
    s_page_ptr,  # [n_slots] int32   per-slot image page id (-1 == none)
    q_page_ptr,  # [R] int32
    seq_ptr,  # [R] int32         sequence length INCLUDING this token
    out_ptr,  # [R, max_block] int8
    stride_bt_r,
    stride_out_r,
    W,
    num_blocks,
    max_block,
    BS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Build ``allow = same_page | open_row``; needs a full-row reduction, so two scans.

    Scans stop at the live sequence length because fixed-width graph tables may
    contain stale columns from other requests.
    """
    r = tl.program_id(0)
    qp = tl.load(q_page_ptr + r).to(tl.int32)
    seq = tl.load(seq_ptr + r).to(tl.int32)
    off_c = tl.arange(0, CHUNK)
    off_b = tl.arange(0, BS)

    # Count matches explicitly; Python ``not`` does not negate a Triton scalar.
    n_same = tl.zeros((), dtype=tl.int32)
    for c0 in tl.range(0, W, CHUNK):
        w = c0 + off_c
        in_w = (w < W) & (w * BS < seq)
        pages = tl.load(bt_ptr + r * stride_bt_r + w, mask=in_w, other=0).to(tl.int64)
        pages = tl.minimum(tl.maximum(pages, 0), num_blocks - 1)
        slots = pages[:, None] * BS + off_b[None, :]
        ip = tl.load(s_page_ptr + slots).to(tl.int32)  # [CHUNK, BS]
        blk_max = tl.max(tl.where(ip >= 0, ip, -1), axis=1)
        blk_min = tl.min(tl.where(ip >= 0, ip, _INT16_MAX_C), axis=1)
        blk_min = tl.where(blk_max >= 0, blk_min, -1)
        same = in_w & (blk_min >= 0) & (blk_min <= qp) & (qp <= blk_max) & (qp >= 0)
        n_same += tl.sum(same.to(tl.int32), axis=0)

    open_row = (qp < 0) | (n_same == 0)
    for c0 in tl.range(0, max_block, CHUNK):
        w = c0 + off_c
        in_out = w < max_block
        in_w = (w < W) & (w * BS < seq)
        pages = tl.load(bt_ptr + r * stride_bt_r + w, mask=in_w, other=0).to(tl.int64)
        pages = tl.minimum(tl.maximum(pages, 0), num_blocks - 1)
        slots = pages[:, None] * BS + off_b[None, :]
        ip = tl.load(s_page_ptr + slots).to(tl.int32)
        blk_max = tl.max(tl.where(ip >= 0, ip, -1), axis=1)
        blk_min = tl.min(tl.where(ip >= 0, ip, _INT16_MAX_C), axis=1)
        blk_min = tl.where(blk_max >= 0, blk_min, -1)
        same = in_w & (blk_min >= 0) & (blk_min <= qp) & (qp <= blk_max) & (qp >= 0)
        # Columns past the live width are open-row only; the torch path pads the
        # per-request table with -1, so `same` is False there.
        tl.store(
            out_ptr + r * stride_out_r + w,
            (same | open_row).to(tl.int8),
            mask=in_out,
        )


@triton.jit
def _cos_sin_2d_kernel(
    tok_ptr,  # [T] int
    page_ptr,  # [T] int
    inv_tok_ptr,  # [N_TOK] fp32
    inv_page_ptr,  # [PAGE_PAIRS] fp32
    cos_ptr,  # [T, DIM] out
    sin_ptr,  # [T, DIM] out
    stride_c,
    HALF: tl.constexpr,
    N_TOK: tl.constexpr,
    DIM: tl.constexpr,
    PAGE_MOD: tl.constexpr,
    OUT_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Build one row of the Page2D cos/sin table per token.

    Angles are built in fp32 and the frequency axes are indexed at ``j % HALF``
    to reproduce the reference's ``cat(ang, ang)``.
    """
    r = tl.program_id(0)
    t = tl.load(tok_ptr + r).to(tl.float32)
    p = tl.load(page_ptr + r).to(tl.int64)
    if PAGE_MOD > 0:
        # int64 remainder, matching the reference; float modulo drifts on large ids.
        p = p % PAGE_MOD
    pf = p.to(tl.float32)

    off = tl.arange(0, BLOCK)
    j = tl.where(off < HALF, off, off - HALF)
    is_tok = j < N_TOK
    inv_t = tl.load(inv_tok_ptr + tl.minimum(j, N_TOK - 1))
    inv_p = tl.load(inv_page_ptr + tl.maximum(j - N_TOK, 0))
    ang = tl.where(is_tok, t * inv_t, pf * inv_p)

    m = off < DIM
    c = tl.cos(ang)
    s = tl.sin(ang)
    if OUT_BF16:
        c = c.to(tl.bfloat16)
        s = s.to(tl.bfloat16)
    tl.store(cos_ptr + r * stride_c + off, c, mask=m)
    tl.store(sin_ptr + r * stride_c + off, s, mask=m)


@triton.jit
def _local_union_kernel(
    allow_ptr,  # [R, A_W] int8   page-isolation mask
    out_ptr,  # [R, O_W] int8   destination, may have a wider stride
    seq_ptr,  # [R] int32       sequence length INCLUDING this token
    A_W,
    O_W,
    max_block,
    stride_a_r,
    stride_o_r,
    BS: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Union the local window into the candidate mask, one program per request.

    ``clamp(q_blk - j, 0, max_block - 1)`` over contiguous ``j`` is the range
    ``[clamp(q_blk - LOCAL_BLOCKS + 1, 0, max_block - 1), clamp(q_blk, 0, max_block - 1)]``,
    so the window needs no index tensor. It is deliberately not clipped to the
    sequence length: the reference scatter writes the full padded width.
    """
    r = tl.program_id(0)
    seq = tl.load(seq_ptr + r).to(tl.int32)
    q_blk = (seq - 1) // BS
    hi = tl.minimum(tl.maximum(q_blk, 0), max_block - 1)
    lo = tl.minimum(tl.maximum(q_blk - (LOCAL_BLOCKS - 1), 0), max_block - 1)

    off = tl.arange(0, CHUNK)
    for c0 in tl.range(0, max_block, CHUNK):
        w = c0 + off
        m = w < max_block
        a = tl.load(allow_ptr + r * stride_a_r + w, mask=m & (w < A_W), other=0)
        in_win = (w >= lo) & (w <= hi)
        val = ((a != 0) | in_win).to(tl.int8)
        tl.store(out_ptr + r * stride_o_r + w, val, mask=m & (w < O_W))


def _rows_or_new(buf: torch.Tensor | None, rows: int, dtype, device) -> torch.Tensor:
    if buf is None or buf.shape[0] < rows:
        return torch.empty(rows, dtype=dtype, device=device)
    return buf[:rows]


@torch.no_grad()
def dash_decode_meta(
    *,
    input_ids: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_mapping: torch.Tensor,
    coord_rows: torch.Tensor,
    slot_img_page: torch.Tensor,
    slot_marker_cum: torch.Tensor,
    slot_img_run: torch.Tensor,
    slot_prompt_pages: torch.Tensor,
    coord_marker: torch.Tensor,
    coord_prompt_pages: torch.Tensor,
    coord_page_counts: torch.Tensor,
    image_token_id: int,
    page_token_id: int | None,
    per_page_image_tokens: int,
    num_slots: int,
    max_pages: int,
    block_size: int,
    out_q_page: torch.Tensor | None = None,
    out_tok: torch.Tensor | None = None,
    out_page: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Advance one-token decode page state and Page2D coordinates in place.

    ``slot_*`` are per-KV-slot int32 ``[num_slots]`` tables; ``coord_*`` are the
    per-request-row tables indexed by ``coord_rows`` ``[R]`` int64, with
    ``coord_page_counts`` ``[rows, max_pages]``. Returns ``q_page`` ``[R]`` int32
    and ``(tok_pos, page_pos)`` ``[R]`` int64; caller-provided outputs keep their
    addresses for CUDA graph replay.
    """
    R = int(seq_lens.shape[0])
    device = input_ids.device
    q_page = _rows_or_new(out_q_page, R, torch.int32, device)
    tok = _rows_or_new(out_tok, R, torch.int64, device)
    page = _rows_or_new(out_page, R, torch.int64, device)
    # Already int64 on the eager and cuda-graph paths, so avoid a per-step .to()
    # allocation the graph would bake a pointer to.
    slots = slot_mapping[:R]
    if slots.dtype != torch.int64:
        slots = slots.to(torch.int64)
    _decode_meta_kernel[(R,)](
        input_ids[:R],
        block_table,
        seq_lens,
        slots,
        coord_rows,
        slot_img_page,
        slot_marker_cum,
        slot_img_run,
        slot_prompt_pages,
        coord_marker,
        coord_prompt_pages,
        coord_page_counts,
        q_page,
        tok,
        page,
        block_table.stride(0),
        coord_page_counts.stride(0),
        image_token_id,
        page_token_id or 0,
        per_page_image_tokens,
        num_slots,
        max_pages,
        int(block_table.shape[1]),
        HAS_MARKER=page_token_id is not None,
        BS=block_size,
        num_warps=1,
    )
    return q_page, (tok, page)


@torch.no_grad()
def dash_decode_allow(
    *,
    block_table: torch.Tensor,
    slot_img_page: torch.Tensor,
    q_page: torch.Tensor,
    seq_lens: torch.Tensor,
    max_block: int,
    num_blocks: int,
    block_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Page-isolation candidate mask ``[R, max_block]`` int8 (1 = candidate).

    ``slot_img_page`` is the per-KV-slot image page id table (-1 == none).
    """
    R = int(q_page.shape[0])
    W = min(int(max_block), int(block_table.shape[1]))
    if out is None or out.shape[0] < R or out.shape[1] < max_block:
        out = torch.zeros(R, max_block, dtype=torch.int8, device=q_page.device)
    view = out[:R, :max_block]
    _decode_allow_kernel[(R,)](
        block_table,
        slot_img_page,
        q_page,
        seq_lens,
        view,
        block_table.stride(0),
        view.stride(0),
        W,
        num_blocks,
        max_block,
        BS=block_size,
        CHUNK=64,
        num_warps=4,
    )
    return view


@torch.no_grad()
def page2d_cos_sin(
    *,
    inv_tok: torch.Tensor,
    inv_page: torch.Tensor,
    tok_pos: torch.Tensor,
    page_pos: torch.Tensor,
    dim: int,
    page_mod: int,
    dtype: torch.dtype,
    out_cos: torch.Tensor,
    out_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write a Page2D rotary cos/sin table ``[T, dim]`` into caller-owned buffers.

    ``inv_tok`` / ``inv_page`` are the fp32 token- and page-axis frequencies
    (``inv_tok`` pairs first); ``out_cos``/``out_sin`` must be the fixed-address
    buffers the decode graph reads by pointer.
    """
    T = int(tok_pos.shape[0])
    cos, sin = out_cos[:T], out_sin[:T]
    _cos_sin_2d_kernel[(T,)](
        tok_pos,
        page_pos,
        inv_tok.contiguous(),
        inv_page.contiguous(),
        cos,
        sin,
        cos.stride(0),
        HALF=dim // 2,
        N_TOK=int(inv_tok.shape[0]),
        DIM=dim,
        PAGE_MOD=int(page_mod),
        OUT_BF16=dtype == torch.bfloat16,
        BLOCK=triton.next_power_of_2(dim),
        num_warps=1,
    )
    return cos, sin


@torch.no_grad()
def local_union_fused(
    *,
    allow: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    max_block: int,
    block_size: int,
    local_blocks: int,
) -> torch.Tensor:
    """Union the local window into ``allow``, writing int8 into ``out``.

    ``out`` must be the fixed-address buffer the decode graph reads; only its
    first ``max_block`` columns are written.
    """
    R = int(seq_lens.shape[0])
    _local_union_kernel[(R,)](
        allow,
        out,
        seq_lens,
        int(allow.shape[1]),
        int(out.shape[1]),
        max_block,
        allow.stride(0),
        out.stride(0),
        BS=block_size,
        LOCAL_BLOCKS=int(local_blocks),
        CHUNK=256,
        num_warps=4,
    )
    return out[:R, :max_block]
