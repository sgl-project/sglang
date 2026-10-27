"""Fused prefill indexer for gfx950: per-block max index score with a streaming top-k.

Each program scores one tile of queries against one split of the causal context and keeps
the best ``KP`` blocks per query in registers; a merge kernel then reduces the per-split
lists, adds the forced init/local blocks and writes ascending ids. The block scores are the
same MFMA dot products as the score-only kernels (bf16 for an fp8 cache, fp16 even dims
then odd dims for NVFP4), so the selection only differs from ``_topk_index_kernel`` on
exactly tied scores.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import (
    buffer_load,
    compute_efficient_padded_shared_layout,
    mfma,
)

from ..common.nvfp4 import Nvfp4Cache
from ..common.nvfp4_gfx950 import nvfp4_buffer_loadable, nvfp4_even_odd

# Per-split list length; it must cover topk minus the forced blocks.
_KP = 16
_BLOCK_M = 128
# A long context per query tile is scored at 64 rows per warp, halving the LDS reads of K
# per MFMA; measured on MI350X from 4k-query chunks with at least as much prefix.
_BLOCK_M_LONG = 256
_SUB_N = 32
_NUM_WARPS = 4
# Measured on gfx950 (256 CUs) over 2k-32k query chunks with 0-150k prefixes.
_TARGET_PROGRAMS = 2048
_MIN_BLOCKS_PER_SPLIT = 4
_MAX_SPLITS = 32
# One wave of programs (256 CUs, two each by the fp16 K staging's LDS): chunks of fewer
# query tiles split further to fill it, measured from 16 to 2k queries.
_WAVE_PROGRAMS = 512
# Candidates one merge program reduces per query; larger lists go through a first pass.
_MERGE_GROUP = 128
# Candidates a merge program holds across its query rows.
_MERGE_PROGRAM_CANDIDATES = 2048


@gluon.jit
def _min_with_idx(a_v, a_i, b_v, b_i):
    take_b = (b_v < a_v) | ((b_v == a_v) & (b_i < a_i))
    return gl.where(take_b, b_v, a_v), gl.where(take_b, b_i, a_i)


@gluon.jit
def _row_offset(slot, stride_page, stride_row, PAGE_SIZE: gl.constexpr):
    return (slot // PAGE_SIZE) * stride_page + (slot % PAGE_SIZE) * stride_row


@gluon.jit
def _load_k_block(
    k_cache_ptr,
    k_scale_ptr,
    base,
    k_offsets,
    sc_offsets,
    strides,
    PAGE_SIZE: gl.constexpr,
    MMA_DTYPE: gl.constexpr,
    K_FP4: gl.constexpr,
):
    """The index-K block whose first slot is ``base``, dims x tokens: [HEAD_DIM,
    BLOCK_N] in the MMA dtype, or NVFP4 as stored, u16 payload [HEAD_DIM // 4, BLOCK_N]
    and E4M3 scales [HEAD_DIM // 16, BLOCK_N]."""
    stride_k_page, stride_k_s, stride_sc_page, stride_sc_s = strides
    # The slot offset rides in the 32-bit offsets so the buffer base stays uniform.
    k = buffer_load(
        ptr=k_cache_ptr,
        offsets=k_offsets + _row_offset(base, stride_k_page, stride_k_s, PAGE_SIZE),
    )
    if K_FP4:
        sc_base = _row_offset(base, stride_sc_page, stride_sc_s, PAGE_SIZE)
        return k, buffer_load(ptr=k_scale_ptr, offsets=sc_offsets + sc_base)
    else:
        return k.to(MMA_DTYPE)


@gluon.jit
def _stage_k_block(
    smem_k, buf, k_block, GROUP_LAYOUT: gl.constexpr, K_FP4: gl.constexpr
):
    """Write a loaded block to LDS buffer ``buf``; NVFP4 dequantizes into its even dims
    followed by its odd dims."""
    if K_FP4:
        payload, scales = k_block
        even, odd = nvfp4_even_odd(payload, scales, GROUP_LAYOUT)
        HALF: gl.constexpr = even.shape[0]
        N: gl.constexpr = even.shape[1]
        k_block = gl.reshape(gl.permute(gl.join(even, odd), [2, 0, 1]), [2 * HALF, N])
    smem_k.index(buf).store(k_block)


@gluon.jit
def _index_topk_partial_kernel(
    q_ptr,  # [total_q, num_heads, HEAD_DIM] bf16/fp16
    k_cache_ptr,  # [max_slots, num_kv_heads, HEAD_DIM] fp8 or q dtype; NVFP4: u16 payload
    k_scale_ptr,  # NVFP4: uint8 E4M3 scales; unused otherwise
    req_to_token_ptr,  # [max_reqs, max_kv_len]
    cu_seqlens,
    prefix_lens,
    slot_ids,
    part_s_ptr,  # [num_heads, total_q, num_splits * KP] fp32 unscaled q.k block maxima
    part_i_ptr,  # [num_heads, total_q, num_splits * KP] int32 block ids, -1 = empty
    num_heads,
    gqa_group_size,
    init_blocks,
    local_blocks,
    stride_q_n,
    stride_q_h,
    stride_k_page,  # fp8: a page is one slot (PAGE_SIZE 1)
    stride_k_h,
    stride_k_s,
    stride_sc_page,
    stride_sc_h,
    stride_sc_s,
    stride_r2t_b,
    stride_p_h,
    stride_p_n,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    SUB_N: gl.constexpr,
    KP: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    K_FP4: gl.constexpr,
    num_warps: gl.constexpr,
):
    pid_q = gl.program_id(0)
    pid_b = gl.program_id(1) // num_heads
    pid_h = gl.program_id(1) % num_heads
    pid_s = gl.program_id(2)
    num_splits = gl.num_programs(2)
    seq_start = gl.load(cu_seqlens + pid_b)
    q_len = gl.load(cu_seqlens + pid_b + 1) - seq_start
    if pid_q * BLOCK_M >= q_len:
        return
    prefix_len = gl.load(prefix_lens + pid_b)
    r2t_row = req_to_token_ptr + gl.load(slot_ids + pid_b).to(gl.int64) * stride_r2t_b
    kv_head = pid_h // gqa_group_size
    strides = (stride_k_page, stride_k_s, stride_sc_page, stride_sc_s)

    # At 64 rows per warp a sub-tile reduces to its row maxima at once, to fit two waves'
    # registers; at 32 rows the elementwise running max is cheaper.
    ROW_MAX: gl.constexpr = BLOCK_M // num_warps > 32
    MMA_DTYPE: gl.constexpr = gl.float16 if K_FP4 else q_ptr.dtype.element_ty
    # A lane loads 16 contiguous bytes of a token row: fp8 dims or NVFP4 u16.
    LOAD_ROWS: gl.constexpr = HEAD_DIM // 4 if K_FP4 else HEAD_DIM
    LOAD_VEC: gl.constexpr = 8 if K_FP4 else 16
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    dot_q: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    dot_k: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )
    q_load: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [num_warps, 1], [1, 0])
    k_load: gl.constexpr = gl.BlockedLayout(
        [LOAD_VEC, 1],
        [LOAD_ROWS // LOAD_VEC, 64 // (LOAD_ROWS // LOAD_VEC)],
        [1, num_warps],
        [0, 1],
    )
    # NVFP4 payload as [groups of 16 dims, 4 u16, tokens]; scales slice out the u16 dim.
    group_layout: gl.constexpr = gl.BlockedLayout(
        [2, 4, 1],
        [HEAD_DIM // 32, 1, 64 // (HEAD_DIM // 32)],
        [1, 1, num_warps],
        [1, 0, 2],
    )
    scale_layout: gl.constexpr = gl.SliceLayout(1, group_layout)
    k_shared: gl.constexpr = compute_efficient_padded_shared_layout(
        dot_k, [HEAD_DIM, BLOCK_N], MMA_DTYPE, True
    )
    # Rows sit on lanes like the transposed MFMA result, so the row maxima need no shuffle.
    top_layout: gl.constexpr = gl.BlockedLayout(
        [1, KP // 2], [32, 2], [num_warps, 1], [0, 1]
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, top_layout)

    # Candidates of a row are [init_blocks, cand_hi); the merge adds the forced blocks.
    rows = pid_q * BLOCK_M + gl.arange(0, BLOCK_M, layout=row_layout)
    cand_hi = gl.maximum((prefix_len + rows) // BLOCK_N + 1 - local_blocks, init_blocks)
    last_row = gl.minimum(q_len, (pid_q + 1) * BLOCK_M) - 1
    tile_hi = gl.maximum(
        (prefix_len + last_row) // BLOCK_N + 1 - local_blocks, init_blocks
    )
    per_split = gl.cdiv(tile_hi - init_blocks, num_splits)
    b0 = init_blocks + pid_s * per_split
    b1 = gl.minimum(tile_hi, b0 + per_split)

    offs_qm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, q_load))
    offs_qd = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_load))
    q = buffer_load(
        ptr=q_ptr
        + (seq_start + pid_q * BLOCK_M).to(gl.int64) * stride_q_n
        + pid_h * stride_q_h,
        offsets=offs_qm[:, None] * stride_q_n + offs_qd[None, :],
        mask=(pid_q * BLOCK_M + offs_qm)[:, None] < q_len,
        other=0.0,
    )
    if K_FP4:
        # fp16 like the NVFP4 scorer's (bf16 widens to fp16 exactly in range), its even
        # dims then its odd dims as K is staged: the K = 128 MFMA chain runs the even
        # dims' MFMAs before the odd dims', as that scorer accumulates.
        q = gl.permute(gl.reshape(q, [BLOCK_M, HEAD_DIM // 2, 2]), [0, 2, 1])
        q = gl.convert_layout(gl.reshape(q, [BLOCK_M, HEAD_DIM]).to(gl.float16), dot_q)
    else:
        q = gl.convert_layout(q, dot_q)

    offs_kd = gl.arange(0, LOAD_ROWS, layout=gl.SliceLayout(1, k_load))
    offs_kn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, k_load))
    k_offsets = kv_head * stride_k_h + offs_kd[:, None] + offs_kn[None, :] * stride_k_s
    offs_sg = gl.arange(0, HEAD_DIM // 16, layout=gl.SliceLayout(1, scale_layout))
    offs_sn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, scale_layout))
    sc_offsets = (
        kv_head * stride_sc_h + offs_sg[:, None] + offs_sn[None, :] * stride_sc_s
    )
    smem_k = gl.allocate_shared_memory(MMA_DTYPE, [2, HEAD_DIM, BLOCK_N], k_shared)

    cols = (
        gl.full([BLOCK_M, KP], 0, gl.int32, top_layout)
        + gl.arange(0, KP, layout=gl.SliceLayout(0, top_layout))[None, :]
    )
    top_s = gl.full([BLOCK_M, KP], float("-inf"), gl.float32, top_layout)
    top_i = gl.full([BLOCK_M, KP], -1, gl.int32, top_layout)

    # Block b0 + i is read from LDS at iteration i, staged there at i - 1 and loaded at
    # i - 2; its page base slot is read one iteration before the load, ahead of that
    # iteration's K loads, so the in-order wait for the slot leaves them in flight.
    n_blk = b1 - b0
    if n_blk > 0:
        base = gl.load(r2t_row + b0 * BLOCK_N).to(gl.int32)
        _stage_k_block(
            smem_k,
            0,
            _load_k_block(
                k_cache_ptr,
                k_scale_ptr,
                base,
                k_offsets,
                sc_offsets,
                strides,
                PAGE_SIZE,
                MMA_DTYPE,
                K_FP4,
            ),
            group_layout,
            K_FP4,
        )
        base = gl.load(r2t_row + gl.minimum(b0 + 1, b1 - 1) * BLOCK_N).to(gl.int32)
        k_next = _load_k_block(
            k_cache_ptr,
            k_scale_ptr,
            base,
            k_offsets,
            sc_offsets,
            strides,
            PAGE_SIZE,
            MMA_DTYPE,
            K_FP4,
        )
        base_next = gl.load(r2t_row + gl.minimum(b0 + 2, b1 - 1) * BLOCK_N).to(gl.int32)
        for it in range(n_blk):
            blk = b0 + it
            buf = it % 2
            _stage_k_block(smem_k, 1 - buf, k_next, group_layout, K_FP4)
            base_after = gl.load(r2t_row + gl.minimum(blk + 3, b1 - 1) * BLOCK_N).to(
                gl.int32
            )
            k_next = _load_k_block(
                k_cache_ptr,
                k_scale_ptr,
                base_next,
                k_offsets,
                sc_offsets,
                strides,
                PAGE_SIZE,
                MMA_DTYPE,
                K_FP4,
            )
            base_next = base_after
            if ROW_MAX:
                row_max = gl.full(
                    [BLOCK_M], float("-inf"), gl.float32, gl.SliceLayout(1, mfma_layout)
                )
            else:
                run_max = gl.full(
                    [BLOCK_M, SUB_N], float("-inf"), gl.float32, mfma_layout
                )
            for j in gl.static_range(BLOCK_N // SUB_N):
                acc = gl.zeros([BLOCK_M, SUB_N], gl.float32, mfma_layout)
                k_sub = smem_k.index(buf).slice(j * SUB_N, SUB_N, dim=1).load(dot_k)
                acc = mfma(q, k_sub, acc)
                if ROW_MAX:
                    row_max = gl.maximum(row_max, gl.max(acc, axis=1))
                else:
                    run_max = gl.maximum(run_max, acc)
            if ROW_MAX:
                s = gl.convert_layout(row_max, row_layout)
            else:
                s = gl.convert_layout(gl.max(run_max, axis=1), row_layout)
            # NaN ranks like the -1e30 the reference top-k substitutes for it.
            s = gl.where(s != s, -1e30, s)
            s = gl.where(blk < cand_hi, s, float("-inf"))
            thr, at = gl.reduce((top_s, cols), 1, _min_with_idx)
            hit = (s > thr)[:, None] & (cols == at[:, None])
            top_s = gl.where(hit, s[:, None], top_s)
            top_i = gl.where(hit, blk, top_i)

    out = (
        pid_h * stride_p_h + (seq_start + rows).to(gl.int64) * stride_p_n + pid_s * KP
    )[:, None] + cols
    row_ok = (rows < q_len)[:, None]
    gl.store(part_s_ptr + out, top_s, mask=row_ok)
    gl.store(part_i_ptr + out, top_i, mask=row_ok)


@triton.jit
def _pack_key(score, blk):
    # High word: fp32 order as signed int32; low word: ~blk, so ties prefer lower blocks.
    bits = score.to(tl.int32, bitcast=True)
    ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    key = (ordered.to(tl.int64) << 32) | ((~blk).to(tl.int64) & 0xFFFFFFFF)
    return tl.where(blk >= 0, key, -9223372036854775808)


@triton.jit
def _reduce_lists_kernel(
    in_s_ptr,  # [rows, nc]
    in_i_ptr,
    out_s_ptr,  # [rows, num_groups * KT]
    out_i_ptr,
    n_rows,
    nc,
    stride_in,
    stride_out,
    BQ: tl.constexpr,
    GROUP: tl.constexpr,
    KT: tl.constexpr,
):
    r = tl.program_id(0) * BQ + tl.arange(0, BQ)
    g = tl.program_id(1)
    c = g * GROUP + tl.arange(0, GROUP)
    r_ok = (r < n_rows)[:, None]
    off = r.to(tl.int64)[:, None] * stride_in + c[None, :]
    s = tl.load(in_s_ptr + off, mask=r_ok & (c[None, :] < nc), other=0.0)
    i = tl.load(in_i_ptr + off, mask=r_ok & (c[None, :] < nc), other=-1)
    best = tl.topk(_pack_key(s, i), KT, dim=1)
    best_i = tl.where(
        best != -9223372036854775808, ~(best & 0xFFFFFFFF).to(tl.int32), -1
    )
    hi = (best >> 32).to(tl.int32)
    best_s = tl.where(hi < 0, hi ^ 0x7FFFFFFF, hi).to(tl.float32, bitcast=True)
    t = tl.arange(0, KT)
    o = r.to(tl.int64)[:, None] * stride_out + g * KT + t[None, :]
    tl.store(out_s_ptr + o, best_s, mask=r_ok)
    tl.store(out_i_ptr + o, best_i, mask=r_ok)


@triton.jit
def _index_topk_merge_kernel(
    part_s_ptr,  # [num_heads * total_q, nc]
    part_i_ptr,
    topk_ptr,  # [num_heads * total_q, topk] int32
    cu_seqlens,
    prefix_lens,
    total_q,
    n_rows,
    batch_size,
    nc,
    topk,
    init_blocks,
    local_blocks,
    stride_p,
    stride_t,
    BLOCK_N: tl.constexpr,
    BQ: tl.constexpr,
    NC: tl.constexpr,
    KT: tl.constexpr,
):
    r = tl.program_id(0) * BQ + tl.arange(0, BQ)
    r_ok = r < n_rows
    n = r % total_q
    req = tl.zeros([BQ], tl.int32)
    for b in range(1, batch_size):
        req += (tl.load(cu_seqlens + b) <= n).to(tl.int32)
    pos = tl.load(prefix_lens + req) + n - tl.load(cu_seqlens + req)
    n_blocks = pos // BLOCK_N + 1

    c = tl.arange(0, NC)
    m = r_ok[:, None] & (c[None, :] < nc)
    off = r.to(tl.int64)[:, None] * stride_p + c[None, :]
    s = tl.load(part_s_ptr + off, mask=m, other=0.0)
    i = tl.load(part_i_ptr + off, mask=m, other=-1)
    best = tl.topk(_pack_key(s, i), KT, dim=1)
    best_id = tl.where(
        best != -9223372036854775808, ~(best & 0xFFFFFFFF).to(tl.int32), -1
    )

    # Forced blocks: init [0, init_blocks) and local [n_blocks - local_blocks, n_blocks).
    lo_local = tl.maximum(n_blocks - local_blocks, 0)
    n_init = tl.minimum(init_blocks, n_blocks)
    local_start = tl.maximum(lo_local, n_init)
    n_forced = n_init + n_blocks - local_start
    t = tl.arange(0, KT)[None, :]
    kept = tl.where(
        (t < (topk - n_forced)[:, None]) & (best_id >= 0), best_id, 0x7FFFFFFF
    )
    forced = tl.where(
        t < n_forced[:, None],
        tl.where(t < n_init[:, None], t, local_start[:, None] + t - n_init[:, None]),
        0x7FFFFFFF,
    )
    # Ascending ids with the -1 tail the sparse attention kernels expect.
    ids = -tl.topk(-tl.reshape(tl.join(kept, forced), [BQ, 2 * KT]), KT, dim=1)
    ids = tl.where(ids == 0x7FFFFFFF, -1, ids)
    tl.store(
        topk_ptr + r.to(tl.int64)[:, None] * stride_t + t,
        ids,
        mask=r_ok[:, None] & (t < topk),
    )


def can_fuse_index_topk(
    q: torch.Tensor,
    k_cache: torch.Tensor | Nvfp4Cache,
    topk: int,
    local_blocks: int,
    page_size: int,
    block_size_k: int,
) -> bool:
    if isinstance(k_cache, Nvfp4Cache):
        k_ok = nvfp4_buffer_loadable(k_cache) and k_cache.page_size % block_size_k == 0
    else:
        k_ok = (
            k_cache.dtype in (q.dtype, torch.float8_e4m3fn)
            and page_size % block_size_k == 0
            and k_cache.stride(-1) == 1
            # buffer_load takes 32-bit byte offsets
            and k_cache.numel() * k_cache.element_size() < 2**31
        )
    return (
        q.dtype in (torch.bfloat16, torch.float16)
        # Candidates are whole causal blocks; the partial self block is always forced.
        and local_blocks >= 1
        and q.shape[-1] == 128
        and block_size_k == 128
        and topk <= _KP
        and k_ok
    )


def _num_splits(q_tiles: int, max_seqlen_k: int, block_size_k: int) -> int:
    max_blocks = triton.cdiv(max_seqlen_k, block_size_k)
    splits = max(
        1,
        min(
            triton.cdiv(_TARGET_PROGRAMS, q_tiles),
            triton.cdiv(max_blocks, _MIN_BLOCKS_PER_SPLIT),
            max(_MAX_SPLITS, _WAVE_PROGRAMS // q_tiles),
        ),
    )
    # Whole waves: the programs of a last partial wave take as long as a full one.
    waves = q_tiles * splits // _WAVE_PROGRAMS
    return max(1, waves * _WAVE_PROGRAMS // q_tiles) if waves else splits


def _merge_partial_topk(
    part_s: torch.Tensor,
    part_i: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefix_lens: torch.Tensor,
    total_q: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    block_size_k: int,
) -> torch.Tensor:
    n_rows, nc = part_s.shape
    kt = triton.next_power_of_2(topk)
    if nc > 2 * _MERGE_GROUP:
        groups = triton.cdiv(nc, _MERGE_GROUP)
        red_s = torch.empty(
            (n_rows, groups * kt), dtype=torch.float32, device=part_s.device
        )
        red_i = torch.empty(
            (n_rows, groups * kt), dtype=torch.int32, device=part_s.device
        )
        bq = _MERGE_PROGRAM_CANDIDATES // _MERGE_GROUP
        _reduce_lists_kernel[(triton.cdiv(n_rows, bq), groups)](
            part_s,
            part_i,
            red_s,
            red_i,
            n_rows,
            nc,
            part_s.stride(0),
            red_s.stride(0),
            BQ=bq,
            GROUP=_MERGE_GROUP,
            KT=kt,
            num_warps=4,
        )
        part_s, part_i, nc = red_s, red_i, groups * kt
    topk_idx = torch.empty((n_rows, topk), dtype=torch.int32, device=part_s.device)
    nc_pow2 = triton.next_power_of_2(nc)
    bq = max(1, min(64, _MERGE_PROGRAM_CANDIDATES // nc_pow2))
    _index_topk_merge_kernel[(triton.cdiv(n_rows, bq),)](
        part_s,
        part_i,
        topk_idx,
        cu_seqlens,
        prefix_lens,
        total_q,
        n_rows,
        cu_seqlens.shape[0] - 1,
        nc,
        topk,
        init_blocks,
        local_blocks,
        part_s.stride(0),
        topk_idx.stride(0),
        BLOCK_N=block_size_k,
        BQ=bq,
        NC=nc_pow2,
        KT=kt,
        num_warps=4,
    )
    return topk_idx


def fused_index_topk(
    q: torch.Tensor,  # [total_q, num_heads, 128]
    k_cache: torch.Tensor | Nvfp4Cache,  # [max_slots, num_kv_heads, 128] or NVFP4
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    block_size_k: int,
) -> torch.Tensor:
    """Top-k block ids per query and index head: [num_heads, total_q, topk] int32,
    ascending with a -1 tail, like ``_topk_index_kernel`` with ``block_size_q == 1``."""
    total_q, num_heads, head_dim = q.shape
    batch = cu_seqlens.shape[0] - 1
    k_fp4 = isinstance(k_cache, Nvfp4Cache)
    if k_fp4:
        k_ptr, sc_ptr = k_cache.payload.view(torch.int16), k_cache.scales
        k_strides = tuple(s // 2 for s in k_cache.payload_strides())
        sc_strides = k_cache.scale_strides()
        page_size, num_kv_heads = k_cache.page_size, k_cache.num_heads
    else:
        k_ptr = sc_ptr = k_cache
        k_strides = (k_cache.stride(0), k_cache.stride(1), k_cache.stride(0))
        sc_strides = (0, 0, 0)
        page_size, num_kv_heads = 1, k_cache.shape[1]
    block_m = (
        _BLOCK_M_LONG
        if max_seqlen_q >= 4096 and max_seqlen_k >= 2 * max_seqlen_q
        else _BLOCK_M
    )
    q_tiles = triton.cdiv(max_seqlen_q, block_m)
    num_splits = _num_splits(q_tiles * batch * num_heads, max_seqlen_k, block_size_k)
    part_s = torch.empty(
        (num_heads, total_q, num_splits * _KP), dtype=torch.float32, device=q.device
    )
    part_i = torch.empty_like(part_s, dtype=torch.int32)
    _index_topk_partial_kernel[(q_tiles, batch * num_heads, num_splits)](
        q,
        k_ptr,
        sc_ptr,
        req_to_token,
        cu_seqlens,
        prefix_lens,
        slot_ids,
        part_s,
        part_i,
        num_heads,
        num_heads // num_kv_heads,
        init_blocks,
        local_blocks,
        q.stride(0),
        q.stride(1),
        *k_strides,
        *sc_strides,
        req_to_token.stride(0),
        part_s.stride(0),
        part_s.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_size_k,
        HEAD_DIM=head_dim,
        SUB_N=_SUB_N,
        KP=_KP,
        PAGE_SIZE=page_size,
        K_FP4=k_fp4,
        num_warps=_NUM_WARPS,
        # The LDS staging allows two programs per CU anyway; given the registers of two
        # waves per SIMD the NVFP4 dequant schedules with fewer stalls.
        waves_per_eu=2 if k_fp4 else 0,
    )
    topk_idx = _merge_partial_topk(
        part_s.view(num_heads * total_q, -1),
        part_i.view(num_heads * total_q, -1),
        cu_seqlens,
        prefix_lens,
        total_q,
        topk,
        init_blocks,
        local_blocks,
        block_size_k,
    )
    return topk_idx.view(num_heads, total_q, topk)
