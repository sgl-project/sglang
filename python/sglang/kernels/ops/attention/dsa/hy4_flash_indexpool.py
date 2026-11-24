"""Triton kernels for the Hunyuan-4-Flash IndexPool indexer and window layers.

Layout contract (shared with ``hunyuan_v4_flash.FlashIndexPool``):

* Each completed group of ``RATIO`` tokens stores one pooled FP8 key (128 bytes)
  plus an fp32 scale in the DSA index-key cache, at the KV slot of the group's
  first token. A cache page holds ``PAGE`` tokens: the keys of the page first
  (``PAGE * 128`` bytes), then their scales (``PAGE * 4`` bytes).
* Logical token ``t`` of a request lives at slot ``pages[b, t // 64] * 64 + t % 64``.
* Rope is GPT-J style on the last 64 dims and emits the rotated half
  de-interleaved ([even pairs, odd pairs]); keys and queries share the layout.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.dsa.kpool_fp8_index import _hadamard128

INDEX_DIM = 128
ROPE_DIM = 64
FP8_MAX = 448.0
TOKENS_PER_PAGE = 64


@triton.jit
def _rope_quant_row(x, pos, inv_freq_ptr):
    """x: [128] fp32 (bf16 values). Returns (fp8 values as fp32 in [-448, 448],
    power-of-two scale)."""
    idx = tl.arange(0, 128)
    is_pe = idx >= 64
    j = tl.where(is_pe, idx - 64, 0)
    # De-interleaved output position j < 32 takes the even element of pair j,
    # j >= 32 the odd element of pair j - 32.
    pair = tl.where(j < 32, j, j - 32)
    inv = tl.load(inv_freq_ptr + pair, mask=is_pe, other=0.0)
    freq = pos.to(tl.float32) * inv
    c = tl.cos(freq)
    s = tl.sin(freq)
    even_idx = 64 + 2 * pair
    odd_idx = even_idx + 1
    xe = tl.gather(x, even_idx, 0)
    xo = tl.gather(x, odd_idx, 0)
    rot = tl.where(j < 32, xe * c - xo * s, xo * c + xe * s)
    x = tl.where(is_pe, rot.to(tl.bfloat16).to(tl.float32), x)
    x = _hadamard128(x)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    q = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    return q, scale


@triton.jit
def _mrope_positions(p0, p1, p2, S1: tl.constexpr, S2: tl.constexpr):
    idx = tl.arange(0, 128)
    j = tl.where(idx >= 64, idx - 64, 0)
    pair = tl.where(j < 32, j, j - 32)
    use_h = (pair % 3 == 1) & (pair < 3 * S1)
    use_w = (pair % 3 == 2) & (pair < 3 * S2)
    return tl.where(use_h, p1, tl.where(use_w, p2, p0))


@triton.jit
def _query_kernel(
    query_ptr,  # [R, H, 128] bf16
    pos_ptr,  # [R] int64, or [R, 3] with MROPE
    hw_ptr,  # [R, H] fp32 head weights (raw projection)
    inv_freq_ptr,  # [32] fp32
    q_out_ptr,  # [R, H, 128] fp8
    w_out_ptr,  # [R, H] fp32
    q_stride_r,
    q_stride_h,
    hw_stride_r,
    pos_stride_r,
    pos_stride_a,
    weight_scale,
    H: tl.constexpr,
    MROPE: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
):
    r = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, 128)
    x = tl.load(query_ptr + r * q_stride_r + h * q_stride_h + d).to(tl.float32)
    if MROPE:
        row = pos_ptr + r * pos_stride_r
        pos = _mrope_positions(
            tl.load(row),
            tl.load(row + pos_stride_a),
            tl.load(row + 2 * pos_stride_a),
            S1,
            S2,
        )
    else:
        pos = tl.load(pos_ptr + r)
    q, scale = _rope_quant_row(x, pos, inv_freq_ptr)
    tl.store(q_out_ptr + (r * H + h) * 128 + d, q.to(q_out_ptr.dtype.element_ty))
    hw = tl.load(hw_ptr + r * hw_stride_r + h).to(tl.float32)
    tl.store(w_out_ptr + r * H + h, hw * weight_scale * scale)


def _mrope_args(positions, mrope_section):
    if mrope_section is None:
        return positions, 0, 0, dict(MROPE=False, S1=0, S2=0)
    if positions.ndim == 1:
        positions = positions[:, None].expand(-1, 3)
    return (
        positions,
        positions.stride(0),
        positions.stride(1),
        dict(MROPE=True, S1=mrope_section[1], S2=mrope_section[2]),
    )


def indexpool_query(
    query, positions, head_weights, inv_freq, num_heads, mrope_section=None
):
    """Rope + Hadamard + FP8 of the indexer queries; head weights folded with the
    query scales. query: [R, H*128] bf16; head_weights: [R, H] (any float dtype,
    unit stride on the head axis)."""
    rows = query.shape[0]
    query = query.view(rows, num_heads, INDEX_DIM)
    q_out = torch.empty(
        (rows, num_heads, INDEX_DIM), dtype=torch.float8_e4m3fn, device=query.device
    )
    w_out = torch.empty((rows, num_heads), dtype=torch.float32, device=query.device)
    if rows == 0:
        return q_out, w_out
    positions, pos_stride_r, pos_stride_a, mrope = _mrope_args(positions, mrope_section)
    _query_kernel[(rows, num_heads)](
        query,
        positions,
        head_weights,
        inv_freq,
        q_out,
        w_out,
        query.stride(0),
        query.stride(1),
        head_weights.stride(0),
        pos_stride_r,
        pos_stride_a,
        num_heads**-0.5 * INDEX_DIM**-0.5,
        H=num_heads,
        num_warps=1,
        **mrope,
    )
    return q_out, w_out


@triton.jit
def _pool_write_kernel(
    raw_ptr,  # [slots, 128] bf16 raw keys
    member_ptr,  # [G, RATIO] int64 member slots
    first_pos_ptr,  # [G] position of the group's first token, [G, 3] with MROPE
    write_ptr,  # [G] int64 destination slot (< 0: skip)
    wv_ptr,  # [128, 128] fp32, transposed: [in, out]
    wg_ptr,  # [128, 128] fp32, transposed: [in, out]
    ape_ptr,  # [RATIO, 128] fp32
    norm_ptr,  # [128] fp32
    inv_freq_ptr,
    buf_ptr,  # uint8 index cache, [pages, PAGE * 132]
    buf_page_stride,
    pos_stride_g,
    pos_stride_a,
    eps,
    RATIO: tl.constexpr,
    PAGE: tl.constexpr,
    MROPE: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
):
    g = tl.program_id(0)
    dst = tl.load(write_ptr + g)
    if dst < 0:
        return
    rows = tl.arange(0, 16)
    live = rows < RATIO
    slots = tl.load(member_ptr + g * RATIO + rows, mask=live, other=0)
    if MROPE:
        row = first_pos_ptr + g * pos_stride_g
        pos = _mrope_positions(
            tl.load(row),
            tl.load(row + pos_stride_a),
            tl.load(row + 2 * pos_stride_a),
            S1,
            S2,
        )
    else:
        pos = tl.load(first_pos_ptr + g)
    _pool_group(
        raw_ptr,
        slots,
        live,
        pos,
        dst,
        wv_ptr,
        wg_ptr,
        ape_ptr,
        norm_ptr,
        inv_freq_ptr,
        buf_ptr,
        buf_page_stride,
        eps,
        PAGE,
    )


@triton.jit
def _pool_group(
    raw_ptr,
    slots,  # [16] int64 member slots (rows >= RATIO masked by `live`)
    live,
    pos,  # position of the group's first token
    dst,  # destination slot
    wv_ptr,
    wg_ptr,
    ape_ptr,
    norm_ptr,
    inv_freq_ptr,
    buf_ptr,
    buf_page_stride,
    eps,
    PAGE: tl.constexpr,
):
    rows = tl.arange(0, 16)
    d = tl.arange(0, 128)
    # y = x @ W^T with W^T pre-transposed ([in, out], row-major); accumulate
    # over 32-wide input chunks to keep the weight tiles out of local memory.
    values = tl.zeros((16, 128), dtype=tl.float32)
    gates = tl.zeros((16, 128), dtype=tl.float32)
    for k0 in tl.static_range(0, 128, 32):
        kk = k0 + tl.arange(0, 32)
        x = tl.load(
            raw_ptr + slots[:, None] * 128 + kk[None, :], mask=live[:, None], other=0.0
        ).to(tl.float32)
        wv = tl.load(wv_ptr + kk[:, None] * 128 + d[None, :])
        wg = tl.load(wg_ptr + kk[:, None] * 128 + d[None, :])
        values += tl.dot(x, wv, input_precision="ieee")
        gates += tl.dot(x, wg, input_precision="ieee")
    ape = tl.load(
        ape_ptr + rows[:, None] * 128 + d[None, :], mask=live[:, None], other=0.0
    )
    logits = tl.where(live[:, None], gates + ape, -float("inf"))
    m = tl.max(logits, axis=0)
    e = tl.exp(logits - m[None, :])
    p = e / tl.sum(e, axis=0)[None, :]
    pooled = tl.sum(tl.where(live[:, None], values * p, 0.0), axis=0)
    w = tl.load(norm_ptr + d)
    pooled = pooled * tl.rsqrt(tl.sum(pooled * pooled, axis=0) / 128 + eps) * w
    pooled = pooled.to(tl.bfloat16).to(tl.float32)
    q, scale = _rope_quant_row(pooled, pos, inv_freq_ptr)
    page = dst // PAGE
    row = dst % PAGE
    base = buf_ptr + page * buf_page_stride
    tl.store(base + row * 128 + d, q.to(tl.float8e4nv).to(tl.uint8, bitcast=True))
    scale_ptr = (base + PAGE * 128 + row * 4).to(tl.pointer_type(tl.float32))
    tl.store(scale_ptr, scale)


@triton.jit
def _decode_update_kernel(
    key_ptr,  # [R, >=128] bf16 raw key projection (before its RMSNorm)
    key_stride,
    key_norm_ptr,  # [128] bf16 RMSNorm weight of the raw key
    key_eps,
    out_loc_ptr,  # [R] this step's KV slot of each row
    seqlen_ptr,  # [R] int32 sequence length including this token
    pages_ptr,  # [R, NP] int32 64-token page table (row r is request r)
    pages_stride,
    num_pages,
    raw_ptr,  # [slots, 128] bf16 raw keys
    raw_slots,
    wv_ptr,
    wg_ptr,
    ape_ptr,
    norm_ptr,
    inv_freq_ptr,
    buf_ptr,
    buf_page_stride,
    xyz_ptr,  # [slots, 3] int32 MRoPE positions per KV slot (MROPE only)
    eps,
    RATIO: tl.constexpr,
    PAGE: tl.constexpr,
    MROPE: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
):
    """Decode: normalize and store each row's raw key, then pool the row's group
    if this token completes it."""
    r = tl.program_id(0)
    d = tl.arange(0, 128)
    x = tl.load(key_ptr + r * key_stride + d).to(tl.float32)
    w = tl.load(key_norm_ptr + d).to(tl.float32)
    key = x * tl.rsqrt(tl.sum(x * x, axis=0) / 128 + key_eps) * w
    slot = tl.load(out_loc_ptr + r).to(tl.int64)
    slot = tl.minimum(tl.maximum(slot, 0), raw_slots - 1)
    tl.store(raw_ptr + slot * 128 + d, key.to(tl.bfloat16))
    pos = tl.load(seqlen_ptr + r).to(tl.int64) - 1
    if (pos >= 0) & (pos % RATIO == RATIO - 1):
        # The other threads re-read this row's key from global memory.
        tl.debug_barrier()
        rows = tl.arange(0, 16)
        live = rows < RATIO
        first = pos - (RATIO - 1)
        tok = first + tl.where(live, rows, 0)
        page = tl.load(
            pages_ptr + r * pages_stride + tl.minimum(tok // 64, num_pages - 1),
            mask=live,
            other=0,
        ).to(tl.int64)
        slots = tl.minimum(tl.maximum(page * 64 + tok % 64, 0), raw_slots - 1)
        dst = tl.sum(tl.where(rows == 0, slots, 0), axis=0)
        if MROPE:
            rope_pos = _mrope_positions(
                tl.load(xyz_ptr + dst * 3),
                tl.load(xyz_ptr + dst * 3 + 1),
                tl.load(xyz_ptr + dst * 3 + 2),
                S1,
                S2,
            )
        else:
            rope_pos = first
        _pool_group(
            raw_ptr,
            slots,
            live,
            rope_pos,
            dst,
            wv_ptr,
            wg_ptr,
            ape_ptr,
            norm_ptr,
            inv_freq_ptr,
            buf_ptr,
            buf_page_stride,
            eps,
            PAGE,
        )


def indexpool_decode_update(
    raw_key_proj,
    key_norm_weight,
    key_eps,
    out_loc,
    seq_lens,
    pages,
    raw_keys,
    wv_t,
    wg_t,
    ape,
    key_norm,
    eps,
    inv_freq,
    index_buf,
    page_size,
    rope_xyz=None,
    mrope_section=None,
):
    """Decode step of the IndexPool cache: RMSNorm + store of each row's raw key,
    and the pooled FP8 key of every group this step completes.

    raw_key_proj: [R, 128] bf16 (row stride free); seq_lens: [R] int32 including
    this token; pages: [R, NP] int32 64-token page table. With MRoPE, rope_xyz
    [slots, 3] int32 holds each KV slot's (t, h, w) position."""
    rows = raw_key_proj.shape[0]
    if rows == 0:
        return
    assert raw_key_proj.stride(1) == 1
    mrope = mrope_section is not None
    _decode_update_kernel[(rows,)](
        raw_key_proj,
        raw_key_proj.stride(0),
        key_norm_weight,
        key_eps,
        out_loc,
        seq_lens,
        pages,
        pages.stride(0),
        pages.shape[1],
        raw_keys,
        raw_keys.shape[0],
        wv_t,
        wg_t,
        ape,
        key_norm,
        inv_freq,
        index_buf,
        index_buf.stride(0),
        rope_xyz if mrope else raw_keys,
        eps,
        RATIO=4,
        PAGE=page_size,
        MROPE=mrope,
        S1=mrope_section[1] if mrope else 0,
        S2=mrope_section[2] if mrope else 0,
        num_warps=4,
    )


def indexpool_pool_write(
    raw_keys,
    member_slots,
    first_positions,
    write_slots,
    wv_t,
    wg_t,
    ape,
    key_norm,
    eps,
    inv_freq,
    index_buf,
    page_size,
    mrope_section=None,
):
    """Pool each group of raw keys and store its FP8 key at ``write_slots``.

    ``wv_t``/``wg_t`` are the value/gate projection weights transposed to
    [in, out] and contiguous. ``first_positions`` is [G], or [G, 3] with MRoPE."""
    groups = member_slots.shape[0]
    if groups == 0:
        return
    first_positions, pos_stride_g, pos_stride_a, mrope = _mrope_args(
        first_positions.contiguous(), mrope_section
    )
    _pool_write_kernel[(groups,)](
        raw_keys,
        member_slots.contiguous(),
        first_positions,
        write_slots.contiguous(),
        wv_t,
        wg_t,
        ape,
        key_norm,
        inv_freq,
        index_buf,
        index_buf.stride(0),
        pos_stride_g,
        pos_stride_a,
        eps,
        RATIO=member_slots.shape[1],
        PAGE=page_size,
        num_warps=4,
        **mrope,
    )


@triton.jit
def _logits_kernel(
    q_ptr,  # [R, H, 128] fp8
    w_ptr,  # [R, H] fp32
    pages_ptr,  # [B, NP] int32
    row_batch_ptr,  # [R] int64
    pos_ptr,  # [R] int64 query positions; groups before the query's are scored
    buf_ptr,
    buf_page_stride,
    out_ptr,  # [R, G] fp32
    pages_stride,
    num_pages,
    G,
    H: tl.constexpr,
    RATIO: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    r = tl.program_id(0)
    gb = tl.program_id(1)
    groups = gb * BLOCK_G + tl.arange(0, BLOCK_G)
    pos = tl.load(pos_ptr + r)
    upper = tl.where(pos >= 0, pos // RATIO, 0)
    in_range = groups < G
    if gb * BLOCK_G >= upper:
        tl.store(out_ptr + r * G + groups, -float("inf"), mask=in_range)
        return
    live = groups < upper
    b = tl.load(row_batch_ptr + r)
    tok = groups * RATIO
    page_col = tl.minimum(tok // 64, num_pages - 1)
    page = tl.load(pages_ptr + b * pages_stride + page_col, mask=live, other=0).to(
        tl.int64
    )
    slot = page * 64 + tok % 64
    kpage = slot // PAGE
    krow = slot % PAGE
    d = tl.arange(0, 128)
    kbase = buf_ptr + kpage * buf_page_stride
    k = (
        tl.load(
            kbase[:, None] + krow[:, None] * 128 + d[None, :],
            mask=live[:, None],
            other=0,
        )
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.bfloat16)
    )
    ks = tl.load(
        (kbase + PAGE * 128 + krow * 4).to(tl.pointer_type(tl.float32)),
        mask=live,
        other=0.0,
    )
    hs = tl.arange(0, H)
    q = tl.load(q_ptr + (r * H + hs[:, None]) * 128 + d[None, :]).to(tl.bfloat16)
    w = tl.load(w_ptr + r * H + hs)
    s = tl.dot(q, tl.trans(k))  # [H, BLOCK_G] fp32
    s = tl.maximum(s, 0.0) * w[:, None]
    logits = tl.sum(s, axis=0) * ks
    logits = tl.where(live, logits, -float("inf"))
    tl.store(out_ptr + r * G + groups, logits, mask=in_range)


@triton.jit
def _logits_rows_kernel(
    q_ptr,  # [R, H, 128] fp8
    w_ptr,  # [R, H] fp32
    pages_ptr,  # [B, NP] int32
    row_batch_ptr,  # [R] int64
    pos_ptr,  # [R] int64
    buf_ptr,
    buf_page_stride,
    out_ptr,  # [R, G] fp32
    pages_stride,
    num_pages,
    R,
    G,
    H: tl.constexpr,
    RATIO: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """BLOCK_R query rows share each tile of pooled keys (one MMA with
    M = BLOCK_R * H). A row block that spans two requests scores row by row."""
    rb = tl.program_id(0)
    gb = tl.program_id(1)
    rows = rb * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = rows < R
    last = tl.minimum(rb * BLOCK_R + BLOCK_R, R) - 1
    groups = gb * BLOCK_G + tl.arange(0, BLOCK_G)
    in_range = groups < G
    pos = tl.load(pos_ptr + rows, mask=row_mask, other=-1)
    upper = tl.where(pos >= 0, pos // RATIO, 0)
    out_offs = rows[:, None].to(tl.int64) * G + groups[None, :]
    out_mask = row_mask[:, None] & in_range[None, :]
    if gb * BLOCK_G >= tl.max(upper, axis=0):
        tl.store(out_ptr + out_offs, -float("inf"), mask=out_mask)
        return
    b_first = tl.load(row_batch_ptr + rb * BLOCK_R)
    b_last = tl.load(row_batch_ptr + last)
    d = tl.arange(0, 128)
    hs = tl.arange(0, H)
    if b_first == b_last:
        live_g = groups < tl.max(upper, axis=0)
        tok = groups * RATIO
        page_col = tl.minimum(tok // 64, num_pages - 1)
        page = tl.load(
            pages_ptr + b_first * pages_stride + page_col, mask=live_g, other=0
        ).to(tl.int64)
        slot = page * 64 + tok % 64
        kpage = slot // PAGE
        krow = slot % PAGE
        kbase = buf_ptr + kpage * buf_page_stride
        k = (
            tl.load(
                kbase[:, None] + krow[:, None] * 128 + d[None, :],
                mask=live_g[:, None],
                other=0,
            )
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.bfloat16)
        )
        ks = tl.load(
            (kbase + PAGE * 128 + krow * 4).to(tl.pointer_type(tl.float32)),
            mask=live_g,
            other=0.0,
        )
        qr = (rows[:, None] * H + hs[None, :]).reshape(BLOCK_R * H)
        qm = tl.broadcast_to(row_mask[:, None], (BLOCK_R, H)).reshape(BLOCK_R * H)
        q = tl.load(
            q_ptr + qr[:, None] * 128 + d[None, :], mask=qm[:, None], other=0.0
        ).to(tl.bfloat16)
        w = tl.load(w_ptr + qr, mask=qm, other=0.0)
        s = tl.dot(q, tl.trans(k))  # [BLOCK_R * H, BLOCK_G]
        s = tl.maximum(s, 0.0) * w[:, None]
        logits = tl.sum(s.reshape(BLOCK_R, H, BLOCK_G), axis=1) * ks[None, :]
        logits = tl.where(groups[None, :] < upper[:, None], logits, -float("inf"))
        tl.store(out_ptr + out_offs, logits, mask=out_mask)
    else:
        for i in tl.static_range(BLOCK_R):
            r = rb * BLOCK_R + i
            if r < R:
                p = tl.load(pos_ptr + r)
                up = tl.where(p >= 0, p // RATIO, 0)
                live = groups < up
                b = tl.load(row_batch_ptr + r)
                tok = groups * RATIO
                page = tl.load(
                    pages_ptr + b * pages_stride + tl.minimum(tok // 64, num_pages - 1),
                    mask=live,
                    other=0,
                ).to(tl.int64)
                slot = page * 64 + tok % 64
                kbase = buf_ptr + (slot // PAGE) * buf_page_stride
                krow = slot % PAGE
                k = (
                    tl.load(
                        kbase[:, None] + krow[:, None] * 128 + d[None, :],
                        mask=live[:, None],
                        other=0,
                    )
                    .to(tl.float8e4nv, bitcast=True)
                    .to(tl.bfloat16)
                )
                ks = tl.load(
                    (kbase + PAGE * 128 + krow * 4).to(tl.pointer_type(tl.float32)),
                    mask=live,
                    other=0.0,
                )
                q = tl.load(q_ptr + (r * H + hs[:, None]) * 128 + d[None, :]).to(
                    tl.bfloat16
                )
                w = tl.load(w_ptr + r * H + hs)
                s = tl.dot(q, tl.trans(k))
                s = tl.maximum(s, 0.0) * w[:, None]
                logits = tl.sum(s, axis=0) * ks
                logits = tl.where(live, logits, -float("inf"))
                tl.store(out_ptr + r.to(tl.int64) * G + groups, logits, mask=in_range)


# Rows from which scoring shares key tiles across query rows.
_LOGITS_ROW_BLOCK_MIN_ROWS = 64
_LOGITS_ROWS_CONFIG = (8, 128, 4)  # (BLOCK_R, BLOCK_G, num_warps)


def indexpool_logits(
    q_fp8, weights, pages, row_to_batch, positions, index_buf, page_size, num_groups
):
    """Logits of every row against the pooled groups before its own
    (``positions // 4`` of them); -inf past those up to ``num_groups`` columns."""
    rows, heads, _ = q_fp8.shape
    out = torch.empty((rows, num_groups), dtype=torch.float32, device=q_fp8.device)
    if rows == 0 or num_groups == 0:
        return out
    block = 64
    if rows >= _LOGITS_ROW_BLOCK_MIN_ROWS:
        block_r, block, warps = _LOGITS_ROWS_CONFIG
        _logits_rows_kernel[
            (triton.cdiv(rows, block_r), triton.cdiv(num_groups, block))
        ](
            q_fp8,
            weights,
            pages,
            row_to_batch,
            positions,
            index_buf,
            index_buf.stride(0),
            out,
            pages.stride(0),
            pages.shape[1],
            rows,
            num_groups,
            H=heads,
            RATIO=4,
            PAGE=page_size,
            BLOCK_R=block_r,
            BLOCK_G=block,
            num_warps=warps,
        )
        return out
    _logits_kernel[(rows, triton.cdiv(num_groups, block))](
        q_fp8,
        weights,
        pages,
        row_to_batch,
        positions,
        index_buf,
        index_buf.stride(0),
        out,
        pages.stride(0),
        pages.shape[1],
        num_groups,
        H=heads,
        RATIO=4,
        PAGE=page_size,
        BLOCK_G=block,
        num_warps=4,
    )
    return out


@triton.jit
def _expand_kernel(
    picked_ptr,  # [R, K] int64/int32 candidate groups (any order)
    value_ptr,  # [R, K] fp32 their logits (-inf: not a candidate)
    pos_ptr,  # [R] int64
    pages_ptr,  # [B, NP] int32
    row_batch_ptr,  # [R] int64
    offset_ptr,  # [R] int64 ragged offsets (RAGGED only)
    out_ptr,  # [R, TOPK] int32
    pages_stride,
    num_pages,
    K,
    RAGGED: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TOPK: tl.constexpr,
):
    r = tl.program_id(0)
    pos = tl.load(pos_ptr + r)
    upper = tl.where(pos >= 0, pos // RATIO, 0)
    b = tl.load(row_batch_ptr + r)
    off = 0
    if RAGGED:
        off = tl.load(offset_ptr + r)
    cols = tl.arange(0, TOPK)
    tl.store(out_ptr + r * TOPK + cols, tl.full((TOPK,), -1, tl.int32))
    tl.debug_barrier()
    # History: valid picked groups, packed in order of appearance.
    ks = tl.arange(0, BLOCK_K)
    in_k = ks < K
    group = tl.load(picked_ptr + r * K + ks, mask=in_k, other=0).to(tl.int64)
    val = tl.load(value_ptr + r * K + ks, mask=in_k, other=-float("inf"))
    valid = in_k & (val > -float("inf")) & (group >= 0) & (group < upper)
    dst = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    n_hist = tl.sum(valid.to(tl.int32), axis=0)
    for o in tl.static_range(RATIO):
        tok = group * RATIO + o
        if RAGGED:
            v = off + tok
        else:
            page = tl.load(
                pages_ptr + b * pages_stride + tl.minimum(tok // 64, num_pages - 1),
                mask=valid,
                other=0,
            ).to(tl.int64)
            v = page * 64 + tok % 64
        tl.store(out_ptr + r * TOPK + dst * RATIO + o, v.to(tl.int32), mask=valid)
    # Tail: the query's own group up to the query.
    t = tl.arange(0, RATIO)
    tok = upper * RATIO + t
    live = (pos >= 0) & (tok <= pos)
    if RAGGED:
        v = off + tok
    else:
        page = tl.load(
            pages_ptr + b * pages_stride + tl.minimum(tok // 64, num_pages - 1),
            mask=live,
            other=0,
        ).to(tl.int64)
        v = page * 64 + tok % 64
    tl.store(out_ptr + r * TOPK + n_hist * RATIO + t, v.to(tl.int32), mask=live)


def indexpool_expand(
    picked, values, positions, pages, row_to_batch, ragged_offsets, topk
):
    """Packed sparse indices: the valid picked groups' tokens, then the query's
    own group up to the query, then -1."""
    rows = positions.shape[0]
    out = torch.empty((rows, topk), dtype=torch.int32, device=positions.device)
    if rows == 0:
        return out
    ragged = ragged_offsets is not None
    k = picked.shape[1]
    assert (k + 1) * 4 <= topk, f"{k} picked groups and the tail exceed topk {topk}"
    _expand_kernel[(rows,)](
        picked,
        values,
        positions,
        pages,
        row_to_batch,
        ragged_offsets if ragged else positions,
        out,
        pages.stride(0),
        pages.shape[1],
        k,
        RAGGED=ragged,
        RATIO=4,
        BLOCK_K=triton.next_power_of_2(max(k, 16)),
        TOPK=topk,
        num_warps=4,
    )
    return out


@triton.jit
def _window_kernel(
    pos_ptr,
    pages_ptr,
    row_batch_ptr,
    offset_ptr,
    out_ptr,
    pages_stride,
    num_pages,
    WINDOW: tl.constexpr,
    RAGGED: tl.constexpr,
    TOPK: tl.constexpr,
):
    r = tl.program_id(0)
    pos = tl.load(pos_ptr + r)
    cols = tl.arange(0, TOPK)
    first = tl.maximum(pos - WINDOW + 1, 0)
    tok = first + cols
    valid = (pos >= 0) & (tok <= pos)
    if RAGGED:
        off = tl.load(offset_ptr + r)
        val = off + tok
    else:
        b = tl.load(row_batch_ptr + r)
        page_col = tl.minimum(tok // 64, num_pages - 1)
        page = tl.load(pages_ptr + b * pages_stride + page_col, mask=valid, other=0).to(
            tl.int64
        )
        val = page * 64 + tok % 64
    tl.store(out_ptr + r * TOPK + cols, tl.where(valid, val, -1).to(tl.int32))


def window_indices(positions, pages, row_to_batch, ragged_offsets, window, topk):
    assert window <= topk, f"window {window} exceeds topk {topk}"
    rows = positions.shape[0]
    out = torch.empty((rows, topk), dtype=torch.int32, device=positions.device)
    if rows == 0:
        return out
    ragged = ragged_offsets is not None
    _window_kernel[(rows,)](
        positions,
        pages,
        row_to_batch,
        ragged_offsets if ragged else positions,
        out,
        pages.stride(0),
        pages.shape[1],
        WINDOW=window,
        RAGGED=ragged,
        TOPK=topk,
        num_warps=4,
    )
    return out
