"""Fused target-verify indexer for gfx950: per-block max index score with a streaming top-k.

A request's draft rows (num_draft x index heads, at most 16) form one MFMA tile. Each
warp of a program streams its own index-K blocks of one context split and keeps every
row's best ``KP`` blocks in registers; the program reduces its warps' lists to the
split's best ``KP``, and a merge kernel reduces the splits' lists, adds the forced
init/local blocks and writes ascending ids. The scores are the 16x16x32 MFMA products
and scaling of ``_verify_score_kernel`` (bf16), or for an NVFP4 cache of
``_verify_score_nvfp4_kernel`` (fp16, even dims then odd dims). Blocks rank by score,
then lower block id, in the radix top-k's order; every list keeps its blocks' top
``KP`` in that order, so the ids equal the score + top-k path's whatever the split of
the blocks.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import buffer_load, mfma

from ..common.nvfp4 import Nvfp4Cache
from ..common.nvfp4_gfx950 import nvfp4_buffer_loadable, nvfp4_even_odd_w

_KP = 16
_WARPS = 4
# Below this many requests the score kernel + radix top-k pair is faster: the
# fused split lists cost a merge the small batch cannot amortize (measured on MI350X;
# the NVFP4 score kernel is slower, so fusing it pays off at fewer requests).
VERIFY_FUSED_MIN_REQS = 16
VERIFY_FUSED_MIN_REQS_NVFP4 = 3
# Splits per request; doubled below 8 requests, which leave CUs idle with 32.
_MAX_SPLITS = 32


_NO_KEY = gl.constexpr(-9223372036854775808)


@gluon.jit
def _ordered(s):
    """fp32 scores as int32 in the radix top-k's order (pack_score_id)."""
    bits = s.to(gl.int32, bitcast=True)
    return gl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@gluon.jit
def _key(ordered, blk):
    """Rank of a block as one int64: score, then the lower block id."""
    return (ordered.to(gl.int64) << 32) | ((~blk).to(gl.int64) & 0xFFFFFFFF)


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _local_range(init_blocks, req_hi, WORLD: gl.constexpr, RANK: gl.constexpr):
    """Local ids [lo, hi) of this rank's candidate blocks l * WORLD + RANK in
    [init_blocks, req_hi); with one rank, the blocks themselves."""
    if WORLD == 1:
        return init_blocks, req_hi
    lo = (gl.maximum(init_blocks - RANK, 0) + WORLD - 1) // WORLD
    hi = (gl.maximum(req_hi - RANK, 0) + WORLD - 1) // WORLD
    return lo, gl.maximum(hi, lo)


@gluon.jit
def _split_plan(
    seq_lens_ptr,
    slot_ids_ptr,
    num_reqs,
    max_blocks,
    init_blocks,
    local_blocks,
    NUM_DRAFT: gl.constexpr,
    BLOCK_N: gl.constexpr,
    R_P2: gl.constexpr,
    ND_P2: gl.constexpr,
    PROGRAMS: gl.constexpr,
    MAX_SPLITS: gl.constexpr,
    MIN_SPLIT: gl.constexpr,
    W: gl.constexpr,
    WORLD: gl.constexpr = 1,
    RANK: gl.constexpr = 0,
):
    """This program's (request, split, request's splits, candidate end, req_to_token row).

    The request is >= num_reqs when idle. Every request's candidate blocks are cut into
    splits of at most S blocks (at least one split, at most MAX_SPLITS), with
    S = max(cdiv(total, PROGRAMS), MIN_SPLIT) when those splits fit in PROGRAMS, else
    the same over PROGRAMS - num_reqs, which always fits; the programs take the splits
    in request order, so a long request gets more programs than a short one. With
    WORLD > 1 only this rank's blocks (RANK, RANK + WORLD, ...) count.
    """
    plan_l: gl.constexpr = gl.BlockedLayout([1, ND_P2], [64, 1], [W, 1], [1, 0])
    r = gl.arange(0, R_P2, layout=gl.SliceLayout(1, plan_l))
    d = gl.arange(0, ND_P2, layout=gl.SliceLayout(0, plan_l))
    valid = r < num_reqs
    ok = valid[:, None] & (d < NUM_DRAFT)[None, :]
    lens = gl.load(seq_lens_ptr + r[:, None] * NUM_DRAFT + d[None, :], mask=ok, other=0)
    n_blk = gl.minimum((lens.to(gl.int32) + BLOCK_N - 1) // BLOCK_N, max_blocks)
    # A row's candidates end at max(n_blk - local_blocks, init_blocks).
    req_hi = gl.max(
        gl.where(ok, gl.maximum(n_blk - local_blocks, init_blocks), init_blocks), axis=1
    )
    slots = gl.load(slot_ids_ptr + r * NUM_DRAFT, mask=valid, other=0)
    lo, hi = _local_range(init_blocks, req_hi, WORLD, RANK)
    blocks = gl.where(valid, hi - lo, 0)
    total = gl.sum(blocks, axis=0)
    size = gl.maximum(gl.cdiv(total, PROGRAMS), MIN_SPLIT)
    n = gl.where(valid, gl.minimum(gl.maximum(gl.cdiv(blocks, size), 1), MAX_SPLITS), 0)
    if gl.sum(n, axis=0) > PROGRAMS:
        size = gl.maximum(gl.cdiv(total, PROGRAMS - num_reqs), MIN_SPLIT)
        n = gl.where(
            valid, gl.minimum(gl.maximum(gl.cdiv(blocks, size), 1), MAX_SPLITS), 0
        )
    scan_l: gl.constexpr = gl.BlockedLayout([1], [64], [W], [0])
    ends = gl.associative_scan(gl.convert_layout(n, scan_l), 0, _add)
    p = gl.program_id(0)
    req = gl.sum((ends <= p).to(gl.int32), axis=0)
    mine = r == req
    n_split = gl.sum(gl.where(mine, n, 0), axis=0)
    split = p - gl.sum(
        gl.where(gl.arange(0, R_P2, layout=scan_l) == req, ends, 0), axis=0
    )
    split = split + n_split
    hi = gl.sum(gl.where(mine, req_hi, 0), axis=0)
    slot = gl.sum(gl.where(mine, slots, 0), axis=0)
    return req, split, n_split, hi, slot


@gluon.jit
def _row_offset(slot, stride_page, stride_row, PAGE_SIZE: gl.constexpr):
    return (slot // PAGE_SIZE) * stride_page + (slot % PAGE_SIZE) * stride_row


@gluon.jit
def _load_k_tile(
    k_cache_ptr,
    k_scale_ptr,
    base,
    k_offsets,
    sc_offsets,
    strides,
    PAGE_SIZE: gl.constexpr,
    K_FP4: gl.constexpr,
):
    """One index-K block per warp from its first slot ``base`` [W], dims x tokens, as stored.

    That is [W, HEAD_DIM, BLOCK_N], or the NVFP4 u16 payload [W, HEAD_DIM // 4, BLOCK_N]
    and each token's 8-byte E4M3 scale row as u32 [W, 2, BLOCK_N].
    """
    stride_k_page, stride_k_s, stride_sc_page = strides
    if K_FP4:
        # 32-bit offsets keep the buffer base uniform across the warps' blocks.
        sc_base = gl.convert_layout(
            base, gl.SliceLayout(1, gl.SliceLayout(2, sc_offsets.type.layout))
        )
        k_row = _row_offset(base, stride_k_page, stride_k_s, PAGE_SIZE)
        # A constexpr row pitch lets the compiler prove each 8-byte row load aligned.
        sc_row = _row_offset(sc_base, stride_sc_page, 2, PAGE_SIZE)
        return (
            buffer_load(ptr=k_cache_ptr, offsets=k_row[:, None, None] + k_offsets),
            buffer_load(ptr=k_scale_ptr, offsets=sc_row[:, None, None] + sc_offsets),
        )
    else:
        slot_offsets = base.to(gl.int64) * stride_k_s
        return gl.load(k_cache_ptr + slot_offsets[:, None, None] + k_offsets)


@gluon.jit
def _scale_bytes(srow, grp, scale_layout: gl.constexpr):
    """u32 scale rows [W, 2, N] -> each token's E4M3 byte of groups ``grp`` [G].

    The bytes are u8 [W, G, N] in ``scale_layout``.
    """
    lo, hi = gl.split(gl.permute(srow, [0, 2, 1]))
    zero = gl.zeros(
        [srow.shape[0], grp.shape[0], srow.shape[2]], gl.int32, scale_layout
    )
    lo = gl.convert_layout(lo, gl.SliceLayout(1, scale_layout))[:, None, :] + zero
    hi = gl.convert_layout(hi, gl.SliceLayout(1, scale_layout))[:, None, :] + zero
    # Byte grp of {hi, lo}, zero above it. Not pure, like the dequant: pure, it is hoisted
    # above the next block's prefetch and waits on the loads it should overlap.
    byte = gl.inline_asm_elementwise(
        "v_perm_b32 $0, $1, $2, $3",
        "=v,v,v,v",
        [hi, lo, (grp[None, :, None] + zero) | 0x0C0C0C00],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    return byte.to(gl.uint8)


@gluon.jit
def _score_round(
    k,
    it,
    q_even,
    q_odd,
    grp,
    b0,
    b1,
    warp,
    cand_hi,
    qk_scale,
    top_k,
    scale_layout: gl.constexpr,
    group_layout: gl.constexpr,
    mfma_layout: gl.constexpr,
    dot_k: gl.constexpr,
    row_layout: gl.constexpr,
    M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    K_FP4: gl.constexpr,
    WORLD: gl.constexpr = 1,
    RANK: gl.constexpr = 0,
):
    """Score round ``it`` (local block b0 + it * W + w for warp w) into each row's list.

    Local block l is physical block l * WORLD + RANK. Without ``K_FP4``, ``q_even`` is the
    whole Q operand and ``q_odd`` is unused.
    """
    W: gl.constexpr = warp.shape[0]
    s = gl.zeros([W, M, BLOCK_N], gl.float32, mfma_layout)
    if K_FP4:
        payload, srow = k
        scales = _scale_bytes(srow, grp, scale_layout)
        k_even, k_odd = nvfp4_even_odd_w(payload, scales, group_layout)
        s = mfma(q_even, gl.convert_layout(k_even, dot_k), s)
        s = mfma(q_odd, gl.convert_layout(k_odd, dot_k), s)
    else:
        s = mfma(q_even, gl.convert_layout(k, dot_k).to(q_even.dtype), s)
    s = gl.convert_layout(gl.max(s, axis=2), row_layout) * qk_scale
    # NaN ranks below every real score yet above an empty entry, as the
    # radix top-k's NaN -> -inf does against its unscored tail.
    s = gl.where(s != s, -3.4028234663852886e38, s)
    local = b0 + it * W + warp
    blk = local * WORLD + RANK
    s = gl.where(
        (local[:, None] < b1) & (blk[:, None] < cand_hi[None, :]),
        s,
        float("-inf"),
    )
    # A warp's blocks come in ascending id order, so a new block outranks the list's
    # worst entry only with a higher score; keys in a list are distinct.
    o = _ordered(s)
    thr = gl.min(top_k, axis=2)
    hit = (o > (thr >> 32).to(gl.int32))[:, :, None] & (top_k == thr[:, :, None])
    return gl.where(hit, _key(o, blk[:, None])[:, :, None], top_k)


@gluon.jit
def _verify_index_topk_partial_kernel(
    q_ptr,  # [rows, num_heads, HEAD_DIM] bf16/fp16, rows = num_reqs * NUM_DRAFT
    k_cache_ptr,  # [max_slots, num_kv_heads, HEAD_DIM] fp8 or q dtype; NVFP4: u16 payload
    k_scale_ptr,  # NVFP4: E4M3 scale rows viewed as u32; unused otherwise
    req_to_token_ptr,  # [max_reqs, max_kv_len]
    seq_lens_ptr,  # [rows] per-row causal lengths
    slot_ids_ptr,  # [rows]
    part_k_ptr,  # [num_heads, rows, MAX_SPLITS * KP] int64 keys (_key), by split
    part_n_ptr,  # [num_reqs] int32 splits of each request
    num_reqs,
    sm_scale,
    k_scale,
    max_blocks,
    init_blocks,
    local_blocks,
    gqa_group_size,
    stride_q_r,
    stride_q_h,
    stride_k_page,  # NVFP4 only; fp8 slots address the cache directly
    stride_k_h,
    stride_k_s,
    stride_sc_page,
    stride_sc_h,
    stride_r2t_b,
    stride_p_h,
    stride_p_r,
    NUM_DRAFT: gl.constexpr,
    MAX_SPLITS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_N: gl.constexpr,
    KP: gl.constexpr,
    W: gl.constexpr,
    MIN_BLOCKS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    K_FP4: gl.constexpr,
    PAIRED: gl.constexpr,
    PROGRAMS: gl.constexpr,
    R_P2: gl.constexpr,
    ND_P2: gl.constexpr,
    num_warps: gl.constexpr,
    # Rows of a request's tile (NUM_DRAFT * gqa_group_size, padded); 32 fits the
    # indexer CP's four gathered heads, as two 16-row MFMA tiles of the same K.
    M: gl.constexpr = 16,
    # Indexer CP: this rank scores local blocks l as physical blocks l * WORLD + RANK.
    WORLD: gl.constexpr = 1,
    RANK: gl.constexpr = 0,
):
    req, split, n_split, req_hi, slot = _split_plan(
        seq_lens_ptr, slot_ids_ptr, num_reqs, max_blocks, init_blocks, local_blocks,
        NUM_DRAFT, BLOCK_N, R_P2, ND_P2, PROGRAMS, MAX_SPLITS, W * MIN_BLOCKS, W,
        WORLD, RANK,
    )  # fmt: skip
    if req >= num_reqs:
        return
    pid_kh = gl.program_id(1)
    # Same fp32 evaluation order as _verify_score_kernel, so scaled scores match bitwise.
    qk_scale = sm_scale * 1.4426950409 * k_scale

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[W, 1, 1]
    )
    dot_q: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    dot_k: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )
    q_load: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 4, 16], [W, 1, 1], [2, 1, 0])
    # fp8: 16 contiguous bytes per lane, 8 lanes per token row (coalesced rows).
    # NVFP4: dot_k's order, lane l holding the 4 u16 (one 16-dim group of even and odd
    # dims) its k_width covers for token l % 16, so dequantized tiles need no shuffle.
    k_load: gl.constexpr = (
        gl.BlockedLayout([1, 4, 1], [1, 4, 16], [W, 1, 1], [2, 1, 0])
        if K_FP4
        else gl.BlockedLayout([1, 16, 1], [1, 8, 8], [W, 1, 1], [1, 2, 0])
    )
    # NVFP4 payload as [W, groups of 16 dims, 4 u16, tokens]; scales slice out the u16.
    group_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4, 1], [1, 4, 1, 16], [W, 1, 1, 1], [3, 2, 1, 0]
    )
    scale_layout: gl.constexpr = gl.SliceLayout(2, group_layout)
    # A token's scale row as its two u32 words, whole in each of the four lanes holding
    # one of its groups: one dwordx2 per token instead of a byte load per group.
    srow_layout: gl.constexpr = gl.SliceLayout(
        2, gl.BlockedLayout([1, 2, 1, 1], [1, 1, 4, 16], [W, 1, 1, 1], [1, 3, 2, 0])
    )
    # Rows sit on lanes like the transposed MFMA result, so row maxima need no shuffle.
    top_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, KP // 4], [1, 16, 4], [W, 1, 1], [1, 2, 0]
    )
    row_layout: gl.constexpr = gl.SliceLayout(2, top_layout)  # [W, M]

    # Row m holds draft m // gqa of the request, index head pid_kh * gqa + m % gqa.
    m = gl.arange(0, M, layout=gl.SliceLayout(0, row_layout))
    m_ok = m < NUM_DRAFT * gqa_group_size
    m_row = req * NUM_DRAFT + gl.minimum(m // gqa_group_size, NUM_DRAFT - 1)
    n_blk = gl.minimum(
        (gl.load(seq_lens_ptr + m_row).to(gl.int32) + BLOCK_N - 1) // BLOCK_N,
        max_blocks,
    )
    # Candidates of a row are [init_blocks, cand_hi); the merge adds the forced blocks.
    cand_hi = gl.where(m_ok, gl.maximum(n_blk - local_blocks, init_blocks), 0)
    if (split == 0) & (pid_kh == 0):
        gl.store(part_n_ptr + req, n_split)
    # b0, b1 and the round indices are local ids (the blocks themselves for one rank)
    lo, hi = _local_range(init_blocks, req_hi, WORLD, RANK)
    per_split = gl.cdiv(hi - lo, n_split)
    b0 = lo + split * per_split
    b1 = gl.minimum(hi, b0 + per_split)

    qw = gl.arange(0, W, layout=gl.SliceLayout(1, gl.SliceLayout(2, q_load)))
    qm = gl.arange(0, M, layout=gl.SliceLayout(0, gl.SliceLayout(2, q_load)))
    qd = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, gl.SliceLayout(1, q_load)))
    q_row = req * NUM_DRAFT + gl.minimum(qm // gqa_group_size, NUM_DRAFT - 1)
    q_head = pid_kh * gqa_group_size + qm % gqa_group_size
    q = gl.load(
        q_ptr
        + qw[:, None, None] * 0
        + (q_row.to(gl.int64) * stride_q_r + q_head * stride_q_h)[None, :, None]
        + qd[None, None, :],
        mask=(qm < NUM_DRAFT * gqa_group_size)[None, :, None],
        other=0.0,
    )
    if K_FP4:
        # fp16 halves like the NVFP4 scorer's; bf16 widens to fp16 exactly in range.
        q_even, q_odd = gl.split(gl.reshape(q, [W, M, HEAD_DIM // 2, 2]))
        q_even = gl.convert_layout(q_even.to(gl.float16), dot_q)
        q_odd = gl.convert_layout(q_odd.to(gl.float16), dot_q)
    else:
        q_even = gl.convert_layout(q, dot_q)
        q_odd = q_even

    LOAD_ROWS: gl.constexpr = HEAD_DIM // 4 if K_FP4 else HEAD_DIM
    kw = gl.arange(0, W, layout=gl.SliceLayout(1, gl.SliceLayout(2, k_load)))
    kd = gl.arange(0, LOAD_ROWS, layout=gl.SliceLayout(0, gl.SliceLayout(2, k_load)))
    kt = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, gl.SliceLayout(1, k_load)))
    k_offsets = pid_kh * stride_k_h + kd[None, :, None] + kt[None, None, :] * stride_k_s
    grp = gl.arange(
        0, HEAD_DIM // 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, scale_layout))
    )
    sw = gl.arange(0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(2, srow_layout)))
    st = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, gl.SliceLayout(1, srow_layout)))
    sc_offsets = pid_kh * stride_sc_h + sw[None, :, None] + st[None, None, :] * 2
    strides = (stride_k_page, stride_k_s, stride_sc_page)
    # Draft rows of a request share one req_to_token row.
    r2t_row = req_to_token_ptr + slot.to(gl.int64) * stride_r2t_b

    cols = (
        gl.full([W, M, KP], 0, gl.int32, top_layout)
        + gl.arange(0, KP, layout=gl.SliceLayout(0, gl.SliceLayout(1, top_layout)))[
            None, None, :
        ]
    )
    # Empty entries carry distinct negative ids, stored as -1.
    top_k = _key(
        _ordered(gl.full([W, M, KP], float("-inf"), gl.float32, top_layout)), -1 - cols
    )
    warp = gl.arange(0, W, layout=gl.SliceLayout(1, row_layout))

    # Warp w takes blocks b0 + w, b0 + w + W, ...; the next round's K is in flight while
    # the current round is scored.
    n_rounds = gl.cdiv(b1 - b0, W)
    if n_rounds > 0:
        base = gl.load(r2t_row + (gl.minimum(b0 + kw, b1 - 1) * WORLD + RANK) * BLOCK_N)
        k_next = _load_k_tile(
            k_cache_ptr,
            k_scale_ptr,
            base,
            k_offsets,
            sc_offsets,
            strides,
            PAGE_SIZE,
            K_FP4,
        )
        if PAIRED:
            # Unrolled by two with two register sets, so no prefetched tile is
            # register-copied (the copy would wait on its load). An odd last round
            # rereads block b1 - 1 and scores nothing.
            base_b = gl.load(
                r2t_row + (gl.minimum(b0 + W + kw, b1 - 1) * WORLD + RANK) * BLOCK_N
            )
            base = gl.load(
                r2t_row + (gl.minimum(b0 + 2 * W + kw, b1 - 1) * WORLD + RANK) * BLOCK_N
            )
            for it in range(0, n_rounds, 2):
                k_b = _load_k_tile(
                    k_cache_ptr,
                    k_scale_ptr,
                    base_b,
                    k_offsets,
                    sc_offsets,
                    strides,
                    PAGE_SIZE,
                    K_FP4,
                )
                base_b = gl.load(
                    r2t_row
                    + (gl.minimum(b0 + (it + 3) * W + kw, b1 - 1) * WORLD + RANK)
                    * BLOCK_N
                )
                top_k = _score_round(
                    k_next, it, q_even, q_odd, grp, b0, b1, warp, cand_hi, qk_scale,
                    top_k, scale_layout, group_layout, mfma_layout, dot_k,
                    row_layout, M, BLOCK_N, K_FP4, WORLD, RANK,
                )  # fmt: skip
                k_next = _load_k_tile(
                    k_cache_ptr,
                    k_scale_ptr,
                    base,
                    k_offsets,
                    sc_offsets,
                    strides,
                    PAGE_SIZE,
                    K_FP4,
                )
                base = gl.load(
                    r2t_row
                    + (gl.minimum(b0 + (it + 4) * W + kw, b1 - 1) * WORLD + RANK)
                    * BLOCK_N
                )
                top_k = _score_round(
                    k_b, it + 1, q_even, q_odd, grp, b0, b1, warp, cand_hi, qk_scale,
                    top_k, scale_layout, group_layout, mfma_layout, dot_k,
                    row_layout, M, BLOCK_N, K_FP4, WORLD, RANK,
                )  # fmt: skip
        else:
            base = gl.load(
                r2t_row + (gl.minimum(b0 + W + kw, b1 - 1) * WORLD + RANK) * BLOCK_N
            )
            for it in range(n_rounds):
                k = k_next
                k_next = _load_k_tile(
                    k_cache_ptr,
                    k_scale_ptr,
                    base,
                    k_offsets,
                    sc_offsets,
                    strides,
                    PAGE_SIZE,
                    K_FP4,
                )
                base = gl.load(
                    r2t_row
                    + (gl.minimum(b0 + (it + 2) * W + kw, b1 - 1) * WORLD + RANK)
                    * BLOCK_N
                )
                top_k = _score_round(
                    k, it, q_even, q_odd, grp, b0, b1, warp, cand_hi, qk_scale,
                    top_k, scale_layout, group_layout, mfma_layout, dot_k,
                    row_layout, M, BLOCK_N, K_FP4, WORLD, RANK,
                )  # fmt: skip

    # The split's best KP per row from its W warp lists, best first.
    red_l: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [W, 1], [1, 0])
    ck = gl.reshape(gl.permute(top_k, [1, 0, 2]), [M, W * KP])
    ck = gl.convert_layout(ck, red_l)
    kk = gl.arange(0, KP, layout=gl.SliceLayout(0, red_l))[None, :]
    best = gl.full([M, KP], _NO_KEY, gl.int64, red_l)
    for j in gl.static_range(KP):
        bk = gl.max(ck, axis=1)
        best = gl.where(kk == j, bk[:, None], best)
        ck = gl.where(ck == bk[:, None], _NO_KEY, ck)
    m2 = gl.arange(0, M, layout=gl.SliceLayout(1, red_l))
    row2 = req * NUM_DRAFT + gl.minimum(m2 // gqa_group_size, NUM_DRAFT - 1)
    head2 = pid_kh * gqa_group_size + m2 % gqa_group_size
    out = (head2 * stride_p_h + row2.to(gl.int64) * stride_p_r)[:, None] + (
        split * KP + kk
    )
    gl.store(part_k_ptr + out, best, mask=(m2 < NUM_DRAFT * gqa_group_size)[:, None])


@triton.jit
def _verify_index_topk_merge_kernel(
    part_k_ptr,  # [num_heads * rows, NC] int64 keys
    part_n_ptr,  # [num_reqs] splits of each request
    topk_ptr,  # [num_heads * rows, topk] int32
    seq_lens_ptr,
    rows,
    max_blocks,
    topk,
    init_blocks,
    local_blocks,
    stride_p,
    stride_t,
    NUM_DRAFT: tl.constexpr,
    KP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NC: tl.constexpr,  # next_pow2(MAX_SPLITS * KP)
    KT: tl.constexpr,  # next_pow2(topk)
):
    r = tl.program_id(0)
    req = (r % rows) // NUM_DRAFT
    n_blocks = tl.minimum(
        (tl.load(seq_lens_ptr + r % rows).to(tl.int32) + BLOCK_N - 1) // BLOCK_N,
        max_blocks,
    )
    c = tl.arange(0, NC)
    NO_KEY: tl.constexpr = -9223372036854775808
    key = tl.load(
        part_k_ptr + r.to(tl.int64) * stride_p + c,
        mask=c < tl.load(part_n_ptr + req) * KP,
        other=NO_KEY,
    )
    # Empty entries have negative ids.
    key = tl.where((~key).to(tl.int32) >= 0, key, NO_KEY)
    BIG: tl.constexpr = 0x7FFFFFFF

    # Forced blocks: init [0, init_blocks) and local [n_blocks - local_blocks, n_blocks).
    n_init = tl.minimum(init_blocks, n_blocks)
    local_start = tl.maximum(tl.maximum(n_blocks - local_blocks, 0), n_init)
    n_forced = n_init + n_blocks - local_start
    t = tl.arange(0, KT)
    kept = tl.full([KT], BIG, tl.int32)
    # Highest score first, lower block id on ties.
    for j in tl.static_range(KT):
        if j < topk - n_forced:
            best = tl.max(key, axis=0)
            pick = tl.where(best != NO_KEY, (~best).to(tl.int32), BIG)
            kept = tl.where(t == j, pick, kept)
            key = tl.where(key == best, NO_KEY, key)
    forced = tl.where(
        t < n_forced, tl.where(t < n_init, t, local_start + t - n_init), BIG
    )
    # Ascending ids with the -1 tail the sparse decode kernels expect.
    ids = -tl.topk(-tl.reshape(tl.join(kept, forced), [2 * KT]), KT)
    ids = tl.where(ids == BIG, -1, ids)
    tl.store(topk_ptr + r.to(tl.int64) * stride_t + t, ids, mask=t < topk)


def _scale_rows_loadable(cache: Nvfp4Cache) -> bool:
    """Each token's scale row is 8 bytes the kernel reads as two aligned u32."""
    sc = cache.scales
    return (
        sc.shape[-1] == 8
        and cache.scale_strides()[2] == 8
        and all(st % 4 == 0 for st in (*sc.stride()[:-1], sc.storage_offset()))
    )


def can_fuse_verify_topk(
    q: torch.Tensor,
    k_cache: torch.Tensor | Nvfp4Cache,
    num_draft: int,
    topk: int,
    local_blocks: int,
    page_size: int,
    block_size: int,
) -> bool:
    if isinstance(k_cache, Nvfp4Cache):
        num_kv_heads, min_reqs = k_cache.num_heads, VERIFY_FUSED_MIN_REQS_NVFP4
        # A block within one page has contiguous slots.
        k_ok = (
            nvfp4_buffer_loadable(k_cache)
            and _scale_rows_loadable(k_cache)
            and k_cache.page_size % block_size == 0
        )
    else:
        num_kv_heads, min_reqs = k_cache.shape[1], VERIFY_FUSED_MIN_REQS
        k_ok = (
            k_cache.dtype in (q.dtype, torch.float8_e4m3fn)
            and page_size % block_size == 0
            and k_cache.stride(-1) == 1
        )
    return (
        q.shape[0] // num_draft >= min_reqs
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.shape[-1] == 128
        and num_draft * (q.shape[1] // num_kv_heads) <= 16
        and block_size == 128
        and topk <= _KP
        # Candidates are whole causal blocks; a row's partial last block is forced.
        and local_blocks >= 1
        and k_ok
    )


def _split_config(
    num_reqs: int, num_kv_heads: int, k_fp4: bool
) -> tuple[int, int, int, bool]:
    """(programs, max splits per request, min candidate blocks per warp, paired rounds)."""
    groups = num_reqs * num_kv_heads
    # Programs per KV head: one full wave of two per CU (a partial wave doubles some CUs'
    # work), measured on MI350X at 8-128 requests; the plan needs more than one per request.
    programs = max((256 if groups <= 32 else 512) // num_kv_heads, 2 * num_reqs)
    # Up to twice the mean splits per request, for requests up to twice the mean length.
    cap = 2 * _MAX_SPLITS if num_reqs < 8 else _MAX_SPLITS
    max_splits = max(1, min(cap, 2 * programs // num_reqs))
    min_blocks = 4 if num_reqs < 64 else 8
    # Paired rounds pay off on NVFP4 only (measured on MI350X at 8-64 requests).
    paired = k_fp4
    return programs, max_splits, min_blocks, paired


def fused_verify_topk_idx(
    q: torch.Tensor,  # [rows, num_heads, 128], rows = num_reqs * num_draft
    k_cache: torch.Tensor | Nvfp4Cache,  # [max_slots, num_kv_heads, 128] or NVFP4
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,  # [rows]
    max_seqlen: int,
    slot_ids: torch.Tensor,  # [rows]
    num_draft: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: float,
    k_scale: float,
    out: torch.Tensor | None = None,  # [num_heads, rows, topk] int32
) -> torch.Tensor:
    rows, num_heads, head_dim = q.shape
    k_fp4 = isinstance(k_cache, Nvfp4Cache)
    if k_fp4:
        k_ptr, sc_ptr = (
            k_cache.payload.view(torch.int16),
            k_cache.scales.view(torch.int32),
        )
        k_strides = tuple(s // 2 for s in k_cache.payload_strides())
        # The scale row pitch is the kernel's constexpr 2 u32 (see _scale_rows_loadable).
        sc_strides = tuple(s // 4 for s in k_cache.scale_strides()[:2])
        page_size, num_kv_heads = k_cache.page_size, k_cache.num_heads
    else:
        k_ptr = sc_ptr = k_cache
        k_strides = (0, k_cache.stride(1), k_cache.stride(0))
        sc_strides = (0, 0)
        page_size, num_kv_heads = 1, k_cache.shape[1]
    num_reqs = rows // num_draft
    max_blocks = triton.cdiv(min(max_seqlen, req_to_token.shape[1]), block_size)
    programs, max_splits, min_blocks, paired = _split_config(
        num_reqs, num_kv_heads, k_fp4
    )
    # The merge reads a row's split lists at once.
    nc = triton.next_power_of_2(max_splits * _KP)
    part_k = torch.empty((num_heads, rows, nc), dtype=torch.int64, device=q.device)
    part_n = torch.empty(num_reqs, dtype=torch.int32, device=q.device)
    _verify_index_topk_partial_kernel[(programs, num_kv_heads)](
        q,
        k_ptr,
        sc_ptr,
        req_to_token,
        seq_lens,
        slot_ids,
        part_k,
        part_n,
        num_reqs,
        sm_scale,
        k_scale,
        max_blocks,
        init_blocks,
        local_blocks,
        num_heads // num_kv_heads,
        q.stride(0),
        q.stride(1),
        *k_strides,
        *sc_strides,
        req_to_token.stride(0),
        part_k.stride(0),
        part_k.stride(1),
        NUM_DRAFT=num_draft,
        MAX_SPLITS=max_splits,
        HEAD_DIM=head_dim,
        BLOCK_N=block_size,
        KP=_KP,
        W=_WARPS,
        MIN_BLOCKS=min_blocks,
        PAGE_SIZE=page_size,
        K_FP4=k_fp4,
        PAIRED=paired,
        PROGRAMS=programs,
        R_P2=triton.next_power_of_2(num_reqs),
        ND_P2=triton.next_power_of_2(num_draft),
        num_warps=_WARPS,
    )
    if out is None:
        out = torch.empty((num_heads, rows, topk), dtype=torch.int32, device=q.device)
    else:
        # The merge writes num_heads * rows flat rows through stride(1).
        assert out.shape == (num_heads, rows, topk) and out.dtype == torch.int32
        assert out.stride(2) == 1
        assert num_heads == 1 or out.stride(0) == rows * out.stride(1)
    _verify_index_topk_merge_kernel[(num_heads * rows,)](
        part_k,
        part_n,
        out,
        seq_lens,
        rows,
        max_blocks,
        topk,
        init_blocks,
        local_blocks,
        part_k.stride(1),
        out.stride(1),
        NUM_DRAFT=num_draft,
        KP=_KP,
        BLOCK_N=block_size,
        NC=nc,
        KT=triton.next_power_of_2(topk),
        num_warps=1,
    )
    return out


@triton.jit
def _cp_local_keys_kernel(
    part_k_ptr,  # [num_heads * rows, NC] int64 keys of a request's splits
    part_n_ptr,  # [num_reqs] splits of each request
    keys_ptr,  # [num_heads * rows, KP] int64: the best KP, best first, _NO_KEY empty
    rows,
    stride_p,
    NUM_DRAFT: tl.constexpr,
    KP: tl.constexpr,
    NC: tl.constexpr,
):
    """One rank's split lists reduced to its best KP candidates, without the forced
    blocks: what the native merge would pick from were these all the blocks."""
    r = tl.program_id(0)
    req = (r % rows) // NUM_DRAFT
    c = tl.arange(0, NC)
    NO_KEY: tl.constexpr = -9223372036854775808
    key = tl.load(
        part_k_ptr + r.to(tl.int64) * stride_p + c,
        mask=c < tl.load(part_n_ptr + req) * KP,
        other=NO_KEY,
    )
    # Empty entries have negative ids.
    key = tl.where((~key).to(tl.int32) >= 0, key, NO_KEY)
    tl.store(keys_ptr + r.to(tl.int64) * KP + tl.arange(0, KP), tl.topk(key, KP))


@triton.jit
def _cp_merge_kernel(
    keys_ptr,  # [WORLD sources, num_heads, rows, KP] int64, gathered from every rank
    topk_ptr,  # [rows, topk] int32
    seq_lens_ptr,
    rows,
    max_blocks,
    topk,
    init_blocks,
    local_blocks,
    HEAD: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    WORLD: tl.constexpr,
    KP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KT: tl.constexpr,
):
    """_verify_index_topk_merge_kernel over every rank's best KP of head HEAD: rank r
    scored blocks r, r + WORLD, ..., so together they hold the whole row's best."""
    r = tl.program_id(0)
    n_blocks = tl.minimum(
        (tl.load(seq_lens_ptr + r).to(tl.int32) + BLOCK_N - 1) // BLOCK_N, max_blocks
    )
    c = tl.arange(0, WORLD * KP)
    src, j = c // KP, c % KP
    key = tl.load(
        keys_ptr + ((src * NUM_HEADS + HEAD) * rows + r).to(tl.int64) * KP + j
    )
    NO_KEY: tl.constexpr = -9223372036854775808
    # A key with a negative block id is empty. That includes the zeros the runtime's
    # custom all-gather writes instead of gathering during CUDA-graph warmup.
    key = tl.where((~key).to(tl.int32) >= 0, key, NO_KEY)
    BIG: tl.constexpr = 0x7FFFFFFF
    # Forced blocks: init [0, init_blocks) and local [n_blocks - local_blocks, n_blocks).
    n_init = tl.minimum(init_blocks, n_blocks)
    local_start = tl.maximum(tl.maximum(n_blocks - local_blocks, 0), n_init)
    n_forced = n_init + n_blocks - local_start
    t = tl.arange(0, KT)
    kept = tl.full([KT], BIG, tl.int32)
    # Highest score first, lower block id on ties.
    for i in tl.static_range(KT):
        if i < topk - n_forced:
            best = tl.max(key, axis=0)
            pick = tl.where(best != NO_KEY, (~best).to(tl.int32), BIG)
            kept = tl.where(t == i, pick, kept)
            key = tl.where(key == best, NO_KEY, key)
    forced = tl.where(
        t < n_forced, tl.where(t < n_init, t, local_start + t - n_init), BIG
    )
    # Ascending ids with the -1 tail the sparse decode kernels expect.
    ids = -tl.topk(-tl.reshape(tl.join(kept, forced), [2 * KT]), KT)
    ids = tl.where(ids == BIG, -1, ids)
    tl.store(topk_ptr + r.to(tl.int64) * topk + t, ids, mask=t < topk)


def _cp_split_config(num_reqs: int) -> tuple[int, int, int, bool]:
    """`_split_config` for a rank's quarter of the blocks with every head's rows in
    one tile: one wave of short splits up to 96 requests, then two waves of longer
    splits, which leave fewer CUs idle (measured on MI350X)."""
    if num_reqs <= 96:
        return 256, 16, 4, num_reqs <= 64
    return max(512, 2 * num_reqs), 8, 16, False


def _cp_partial(
    q_heads: torch.Tensor,  # [heads, rows, 128]: every rank's index heads, gathered
    k_cache: Nvfp4Cache,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,  # [rows]
    max_seqlen: int,
    slot_ids: torch.Tensor,  # [rows]
    num_draft: int,
    block_size: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: float,  # q and k scales folded in, as verify_topk_idx does for NVFP4
    world: int,
    rank: int,
):
    """The fused verify indexer's partial pass over this rank's blocks, every head's
    rows in one tile: (split lists [heads, rows, nc], splits per request, nc)."""
    heads, rows, head_dim = q_heads.shape
    num_reqs = rows // num_draft
    max_blocks = triton.cdiv(min(max_seqlen, req_to_token.shape[1]), block_size)
    programs, max_splits, min_blocks, paired = _cp_split_config(num_reqs)
    nc = triton.next_power_of_2(max_splits * _KP)
    part_k = torch.empty((heads, rows, nc), dtype=torch.int64, device=q_heads.device)
    part_n = torch.empty(num_reqs, dtype=torch.int32, device=q_heads.device)
    _verify_index_topk_partial_kernel[(programs, 1)](
        q_heads,
        k_cache.payload.view(torch.int16),
        k_cache.scales.view(torch.int32),
        req_to_token,
        seq_lens,
        slot_ids,
        part_k,
        part_n,
        num_reqs,
        sm_scale,
        1.0,
        max_blocks,
        init_blocks,
        local_blocks,
        heads,
        q_heads.stride(1),
        q_heads.stride(0),
        *(s // 2 for s in k_cache.payload_strides()),
        *(s // 4 for s in k_cache.scale_strides()[:2]),
        req_to_token.stride(0),
        part_k.stride(0),
        part_k.stride(1),
        NUM_DRAFT=num_draft,
        MAX_SPLITS=max_splits,
        HEAD_DIM=head_dim,
        BLOCK_N=block_size,
        KP=_KP,
        W=_WARPS,
        MIN_BLOCKS=min_blocks,
        PAGE_SIZE=k_cache.page_size,
        K_FP4=True,
        PAIRED=paired,
        PROGRAMS=programs,
        R_P2=triton.next_power_of_2(num_reqs),
        ND_P2=triton.next_power_of_2(num_draft),
        num_warps=_WARPS,
        M=max(16, triton.next_power_of_2(num_draft * heads)),
        WORLD=world,
        RANK=rank,
    )
    return part_k, part_n, nc


def cp_verify_local_keys(
    q_heads: torch.Tensor,  # [heads, rows, 128]: every rank's index heads, gathered
    k_cache: Nvfp4Cache,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,  # [rows]
    max_seqlen: int,
    slot_ids: torch.Tensor,  # [rows]
    num_draft: int,
    block_size: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: float,  # q and k scales folded in, as verify_topk_idx does for NVFP4
    world: int,
    rank: int,
) -> torch.Tensor:
    """Indexer CP on an NVFP4 index-K pool: the best _KP candidate keys per (head, row)
    among this rank's blocks rank, rank + world, ..., as [heads, rows, _KP] int64.
    The partial kernel is the fused verify indexer's, every head's rows in one tile, so
    each K block is read once per request and scores match the unsharded ones."""
    heads, rows, _ = q_heads.shape
    part_k, part_n, nc = _cp_partial(
        q_heads,
        k_cache,
        req_to_token,
        seq_lens,
        max_seqlen,
        slot_ids,
        num_draft,
        block_size,
        init_blocks,
        local_blocks,
        sm_scale,
        world,
        rank,
    )
    keys = torch.empty((heads, rows, _KP), dtype=torch.int64, device=q_heads.device)
    _cp_local_keys_kernel[(heads * rows,)](
        part_k,
        part_n,
        keys,
        rows,
        part_k.stride(1),
        NUM_DRAFT=num_draft,
        KP=_KP,
        NC=nc,
        num_warps=1,
    )
    return keys


def cp_verify_merge(
    gathered_keys: torch.Tensor,  # [world, heads, rows, _KP] int64
    head: int,
    seq_lens: torch.Tensor,
    max_seqlen: int,
    req_to_token: torch.Tensor,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Head ``head``'s top-k ids [1, rows, topk] from every rank's candidates, in the
    layout fused_verify_topk_idx writes."""
    world, heads, rows, kp = gathered_keys.shape
    out = torch.empty((1, rows, topk), dtype=torch.int32, device=gathered_keys.device)
    if rows:
        _cp_merge_kernel[(rows,)](
            gathered_keys,
            out,
            seq_lens,
            rows,
            triton.cdiv(min(max_seqlen, req_to_token.shape[1]), block_size),
            topk,
            init_blocks,
            local_blocks,
            HEAD=head,
            NUM_HEADS=heads,
            WORLD=world,
            KP=kp,
            BLOCK_N=block_size,
            KT=triton.next_power_of_2(topk),
            num_warps=1,
        )
    return out


# ---- Indexer CP candidate exchange over IPC (no collective) ----
# A rank writes its best KP candidates of head h straight into rank h's receive
# buffer, [CP_PARITIES, world, CP_MAX_ROWS, KP] int64 there, and rank h's merge
# reads them where they land. Each word carries the writing call's tag, so it is
# its own arrival flag; the tag of a row's n-th call is n % 255 + 1 (0 is never
# written, so the zeroed buffer reads as "not arrived"), counted per row by the
# kernels themselves so graph replay keeps every rank's count in step. Slots
# alternate by call parity: a rank's writes for call n + 2 can only start after
# its own merge of call n + 1, which needed every peer's call n + 1 writes, so
# no peer can still be reading call n's slots.
CP_PARITIES = 2
CP_MAX_ROWS = 4096


@triton.jit
def _wire(key, tag):
    """A native key (ordered score << 32 | ~block) as an exchange word: ordered
    score, tag, then 0xFFFFFF - block, so that within one call the words order as
    the keys do. An empty key becomes the lowest score with block field 0."""
    NO_KEY: tl.constexpr = -9223372036854775808
    blk = (~key).to(tl.int32)
    score = (key >> 32).to(tl.int32)
    empty = key == NO_KEY
    score = tl.where(empty, -2147483648, score)
    blk_field = tl.where(empty, 0, 0xFFFFFF - blk)
    return (
        (score.to(tl.int64) << 32) | (tag.to(tl.int64) << 24) | blk_field.to(tl.int64)
    )


@triton.jit
def _cp_local_keys_ipc_kernel(
    part_k_ptr,  # [num_heads * rows, NC] int64 keys of a request's splits
    part_n_ptr,  # [num_reqs] splits of each request
    peer_ptrs,  # [world] int64 device addresses of every rank's receive buffer
    counters_ptr,  # [num_heads, CP_MAX_ROWS] int32, this rank's per-(head, row) calls
    rows,
    stride_p,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    NUM_DRAFT: tl.constexpr,
    KP: tl.constexpr,
    NC: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    """_cp_local_keys_kernel, delivering the best KP of (head, row) into the
    receive buffer of the rank that owns head ``head`` instead of writing them out."""
    pid = tl.program_id(0)
    head, r = pid // rows, pid % rows
    req = r // NUM_DRAFT
    c = tl.arange(0, NC)
    NO_KEY: tl.constexpr = -9223372036854775808
    key = tl.load(
        part_k_ptr + pid.to(tl.int64) * stride_p + c,
        mask=c < tl.load(part_n_ptr + req) * KP,
        other=NO_KEY,
    )
    key = tl.where((~key).to(tl.int32) >= 0, key, NO_KEY)
    best = tl.topk(key, KP)
    n = tl.load(counters_ptr + head * MAX_ROWS + r)
    tl.store(counters_ptr + head * MAX_ROWS + r, n + 1)
    tag = n % 255 + 1
    parity = n % 2
    dst = tl.load(peer_ptrs + head).to(tl.pointer_type(tl.int64))
    off = ((parity * WORLD + RANK) * MAX_ROWS + r).to(tl.int64) * KP
    tl.store(dst + off + tl.arange(0, KP), _wire(best, tag), cache_modifier=".wt")


@triton.jit
def _cp_merge_ipc_kernel(
    recv_ptr,  # this rank's receive buffer [CP_PARITIES, WORLD, MAX_ROWS, KP] int64
    counters_ptr,  # [MAX_ROWS] int32, this rank's per-row merges
    topk_ptr,  # [rows, topk] int32
    seq_lens_ptr,
    max_blocks,
    topk,
    init_blocks,
    local_blocks,
    WORLD: tl.constexpr,
    KP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KT: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    """_cp_merge_kernel over the words every rank delivered for this row: waits until
    all WORLD * KP carry this call's tag, then selects exactly as the native merge."""
    r = tl.program_id(0)
    n = tl.load(counters_ptr + r)
    tl.store(counters_ptr + r, n + 1)
    tag = (n % 255 + 1).to(tl.int64)
    parity = n % 2
    c = tl.arange(0, WORLD * KP)
    src, j = c // KP, c % KP
    ptrs = recv_ptr + ((parity * WORLD + src) * MAX_ROWS + r).to(tl.int64) * KP + j
    word = tl.load(ptrs, cache_modifier=".cv", volatile=True)
    while tl.min((((word >> 24) & 0xFF) == tag).to(tl.int32), axis=0) == 0:
        word = tl.load(ptrs, cache_modifier=".cv", volatile=True)
    # back to the native key order: ordered score, then the lower block id
    blk_field = word & 0xFFFFFF
    blk = 0xFFFFFF - blk_field
    NO_KEY: tl.constexpr = -9223372036854775808
    key = tl.where(
        blk_field == 0,
        NO_KEY,
        ((word >> 32) << 32) | ((~blk.to(tl.int32)).to(tl.int64) & 0xFFFFFFFF),
    )
    n_blocks = tl.minimum(
        (tl.load(seq_lens_ptr + r).to(tl.int32) + BLOCK_N - 1) // BLOCK_N, max_blocks
    )
    BIG: tl.constexpr = 0x7FFFFFFF
    n_init = tl.minimum(init_blocks, n_blocks)
    local_start = tl.maximum(tl.maximum(n_blocks - local_blocks, 0), n_init)
    n_forced = n_init + n_blocks - local_start
    t = tl.arange(0, KT)
    kept = tl.full([KT], BIG, tl.int32)
    for i in tl.static_range(KT):
        if i < topk - n_forced:
            best = tl.max(key, axis=0)
            pick = tl.where(best != NO_KEY, (~best).to(tl.int32), BIG)
            kept = tl.where(t == i, pick, kept)
            key = tl.where(key == best, NO_KEY, key)
    forced = tl.where(
        t < n_forced, tl.where(t < n_init, t, local_start + t - n_init), BIG
    )
    ids = -tl.topk(-tl.reshape(tl.join(kept, forced), [2 * KT]), KT)
    ids = tl.where(ids == BIG, -1, ids)
    tl.store(topk_ptr + r.to(tl.int64) * topk + t, ids, mask=t < topk)


class CpExchange:
    """One rank's IPC receive buffer, the peers' addresses and the call counters of
    the indexer-CP candidate exchange. Allocate once, before graph capture, on every
    rank of ``group`` (a collective); every rank must then make the same calls."""

    def __init__(self, group, device):
        import torch.distributed as dist

        self.world = group.world_size
        self.rank = group.rank_in_group
        nbytes = CP_PARITIES * self.world * CP_MAX_ROWS * _KP * 8
        addrs = _uncached_ipc_buffers(nbytes, group.cpu_group, self.rank, self.world)
        self.recv = _RawPointer(addrs[self.rank], torch.int64)
        self.peer_ptrs = torch.tensor(addrs, dtype=torch.int64, device=device)
        self.write_calls = torch.zeros(
            (self.world, CP_MAX_ROWS), dtype=torch.int32, device=device
        )
        self.merge_calls = torch.zeros(CP_MAX_ROWS, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)
        dist.barrier(group=group.cpu_group)


def _hip_runtime():
    """The HIP runtime this process already loaded (ROCm wheels ship only the
    versioned soname), through ctypes."""
    import ctypes

    with open("/proc/self/maps") as maps:
        for line in maps:
            if "libamdhip64.so" in line:
                return ctypes.CDLL(line.split()[-1])
    raise RuntimeError("the HIP runtime is not loaded in this process")


def _uncached_ipc_buffers(nbytes, cpu_group, rank, world):
    """Every rank's zeroed ``nbytes`` of uncached device memory, as this rank's
    addresses of them. Uncached, because a peer's writes land in this GPU's memory
    behind its L2: a cached buffer would keep serving the waiting merge a stale line."""
    import ctypes

    import torch.distributed as dist

    class _IpcHandle(ctypes.Structure):
        _fields_ = [("reserved", ctypes.c_byte * 64)]

    hip = _hip_runtime()

    def check(err, what):
        if err != 0:
            raise RuntimeError(f"{what} failed with hipError {err}")

    ptr = ctypes.c_void_p()
    hip_uncached = 3  # hipDeviceMallocUncached
    check(
        hip.hipExtMallocWithFlags(
            ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(hip_uncached)
        ),
        "hipExtMallocWithFlags",
    )
    check(hip.hipMemset(ptr, 0, ctypes.c_size_t(nbytes)), "hipMemset")
    check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
    handle = _IpcHandle()
    check(hip.hipIpcGetMemHandle(ctypes.byref(handle), ptr), "hipIpcGetMemHandle")
    handles = [None] * world
    dist.all_gather_object(handles, bytes(handle.reserved), group=cpu_group)
    hip.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _IpcHandle,
        ctypes.c_uint,
    ]
    addrs = []
    for r, raw in enumerate(handles):
        if r == rank:
            addrs.append(ptr.value)
            continue
        peer = _IpcHandle()
        ctypes.memmove(ctypes.byref(peer), raw, 64)
        opened = ctypes.c_void_p()
        check(
            hip.hipIpcOpenMemHandle(ctypes.byref(opened), peer, ctypes.c_uint(1)),
            "hipIpcOpenMemHandle",
        )
        addrs.append(opened.value)
    return addrs


class _RawPointer:
    """A device address Triton takes as a pointer argument."""

    def __init__(self, addr: int, dtype: torch.dtype):
        self.addr, self.dtype = addr, dtype

    def data_ptr(self) -> int:
        return self.addr


def cp_verify_exchange(
    q_heads,
    k_cache,
    req_to_token,
    seq_lens,
    max_seqlen,
    slot_ids,
    num_draft,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    sm_scale,
    exchange: CpExchange,
):
    """Indexer CP with the candidate exchange done in place over IPC: score this
    rank's blocks, deliver each head's best _KP to its owner, and merge what the
    peers delivered for this rank's head. [1, rows, topk] int32, as cp_verify_merge."""
    heads, rows, head_dim = q_heads.shape
    assert rows <= CP_MAX_ROWS and heads == exchange.world
    num_reqs = rows // num_draft
    max_blocks = triton.cdiv(min(max_seqlen, req_to_token.shape[1]), block_size)
    part_k, part_n, nc = _cp_partial(
        q_heads,
        k_cache,
        req_to_token,
        seq_lens,
        max_seqlen,
        slot_ids,
        num_draft,
        block_size,
        init_blocks,
        local_blocks,
        sm_scale,
        exchange.world,
        exchange.rank,
    )
    _cp_local_keys_ipc_kernel[(heads * rows,)](
        part_k,
        part_n,
        exchange.peer_ptrs,
        exchange.write_calls,
        rows,
        part_k.stride(1),
        RANK=exchange.rank,
        WORLD=exchange.world,
        NUM_DRAFT=num_draft,
        KP=_KP,
        NC=nc,
        MAX_ROWS=CP_MAX_ROWS,
        num_warps=1,
    )
    out = torch.empty((1, rows, topk), dtype=torch.int32, device=q_heads.device)
    _cp_merge_ipc_kernel[(rows,)](
        exchange.recv,
        exchange.merge_calls,
        out,
        seq_lens,
        max_blocks,
        topk,
        init_blocks,
        local_blocks,
        WORLD=exchange.world,
        KP=_KP,
        BLOCK_N=block_size,
        KT=triton.next_power_of_2(topk),
        MAX_ROWS=CP_MAX_ROWS,
        num_warps=1,
    )
    return out
