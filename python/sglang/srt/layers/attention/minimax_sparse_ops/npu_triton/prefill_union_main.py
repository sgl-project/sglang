"""NPU varlen multi-token sparse-attention PREFILL **main** kernel.

This is the "finish prefill sparse" step (round 9 item #1, done correctly for
``block_size_q=1``). The prefill main attention previously reused the **decode**
main kernel ``flash_decode_bnsd_with_gqa_share_sparse`` by flattening every
extend query token into its own "batch row" (``q/block_table/seq_lens/topk_idx``
all ``[total_q, ...]``). With ``total_q ~= 512`` the decode kernel's
``_choose_num_topk_chunks`` collapses to 1, the grid explodes 16x, and there is
**zero cross-query KV reuse** -- correct, but slower than PyTorch's masked-dense
path below the ~12K adaptive crossover.

MiniMax-M3 selects top-k blocks at **per-token** granularity (``block_size_q=1``,
``minimax_sparse_backend.py:106``), so the GPU ``flash_prefill_with_gqa_share_sparse``
trick of "tokens in a query-block share one topk set" does not apply directly.
Instead this kernel tiles ``BSQ_KERNEL`` consecutive query tokens of one request
into a single program and gathers the **UNION** of their per-token topk blocks
once, dots each union block against the whole ``[BSQ_KERNEL * H_TILE, D]`` query
tile (shared KV gather/dot = the win), and restores per-token correctness with a
**per-token-in-union mask** (block ``uj`` only contributes to token ``qi`` if it
is in ``qi``'s own topk). The mask keeps selected-block parity with the per-token
topk, so output matches the per-query decode main at bf16-noise level.

Ascend-safe (reuses the validated patterns + avoids the documented TBE traps):
  * Q stays ``[BSQ*H_TILE, D]`` 1D-row-indexed via divmod -- NO ``[BSQ, small_H]``
    2D int32 (stride-alignment trap, ``prefill_block_score.py:111-118``) and NO
    3D reshape of the dot result (miscompile / ~1500x slow,
    ``flash_block_score_decode.py:525-532``).
  * ``CHUNK_SIZE_U >= 2`` static minimum (``topk_sparse_decode.py:509-515`` LLVM
    crash at width 1).
  * tile sized to fit the 192KB UB: ``BSQ_KERNEL * H_TILE <= 32`` for the default
    (gqa=16, head_dim=128, page=128) -> primary tile (BSQ=4, H_TILE=8).
  * split-K over the union + ``_merge_topk_attn_out_bnsd_kernel`` (reused verbatim
    from ``topk_sparse_decode.py``) restores parallelism when ``all_qtile`` is
    small; single-chunk fast path aliases ``o_partial`` to the output buffer.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.topk_sparse_decode import (
    _choose_num_topk_chunks,  # noqa: F401  (re-exported for parity/tests)
    _floor_power_of_2,
    _get_vectorcore_num_safe,
    _merge_topk_attn_out_bnsd_kernel,
    _MERGE_NS,
    _MERGE_NW,
    _normalize_topk_idx_for_gqa,
    _SPARSE_DECODE_NS,
    _SPARSE_DECODE_NW,
)

# Default kernel-side query tile. BSQ_KERNEL * H_TILE must fit the UB; (4, 8) is
# the validated primary config (see module docstring + plan UB table). Override
# via env MINIMAX_NPU_TRITON_PREFILL_MAIN_BSQ for the sweep.
_DEFAULT_BSQ_KERNEL = 4
_DEFAULT_H_TILE = 8
# Cap on the per-tile union size (power-of-2 constexpr per launch). Adjacent tokens
# share ~12-14 of 16 topk blocks -> a 4-token tile's union is ~18-24, so 32 covers
# p99. If the measured union exceeds this, the wrapper falls back to the per-query
# decode main for that batch (see ``flash_prefill_union_main_bnsd``).
_MAX_U_MAX = 64


def _choose_num_union_chunks(
    all_qtile: int,
    num_kv_heads: int,
    num_h_tiles: int,
    u_max: int,
    max_num_union_chunks: int = 8,
) -> int:
    """Split-K over the union blocks -- Ascend-conservative analogue of
    ``_choose_num_topk_chunks`` (topk_sparse_decode.py:55)."""
    if u_max <= 1:
        return 1
    num_vectorcore = _get_vectorcore_num_safe()
    target_grid = num_vectorcore * 4
    target = max(1, target_grid // max(1, all_qtile * num_kv_heads * num_h_tiles))
    target = min(u_max, max_num_union_chunks, target)
    return _floor_power_of_2(target)


# =============================================================================
# Kernel
# =============================================================================


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_N": lambda args: triton.next_power_of_2(args["block_size"]),
    }
)
@triton.jit
def _gqa_share_sparse_prefill_union_kernel(
    q_ptr,  # [total_q, num_q_heads, head_dim]
    sink_ptr,  # optional [num_q_heads, head_dim] (q when HAS_SINK=False)
    k_cache_ptr,  # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache_ptr,  # same
    block_table_ptr,  # [all_qtile, max_blocks]
    union_idx_ptr,  # [num_kv_heads, all_qtile, u_max]
    union_mask_ptr,  # [num_kv_heads, all_qtile, BSQ_KERNEL, u_max] int8
    o_ptr,  # [C, total_q, num_q_heads, head_dim]
    lse_ptr,  # [C, total_q, num_q_heads]
    qt_to_qstart_ptr,  # [all_qtile] int32
    qt_prefix_ptr,  # [all_qtile] int32
    qt_within_base_ptr,  # [all_qtile] int32
    qt_last_qcount_ptr,  # [all_qtile] int32
    qt_seq_lens_ptr,  # [all_qtile] int32  (last token's KV len; for pos_mask)
    # scalars
    total_q,
    all_qtile,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    u_max,
    max_kv_len,
    # block/scaling
    block_size: tl.constexpr,
    sm_scale,
    # strides -- q
    stride_q_n,
    stride_q_h,
    stride_q_d,
    # sink
    stride_sink_h,
    stride_sink_d,
    # k / v
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_v_block,
    stride_v_offset,
    stride_v_h,
    stride_v_d,
    # block_table
    stride_bt_qt,
    stride_bt_n,
    # union_idx [kv, qt, u]
    stride_ui_h,
    stride_ui_qt,
    stride_ui_u,
    # union_mask [kv, qt, qi, u]
    stride_um_h,
    stride_um_qt,
    stride_um_qi,
    stride_um_u,
    # o / lse partials [C, total_q, h, d] / [C, total_q, h]
    stride_o_c,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_n,
    stride_l_h,
    # meta
    BSQ_KERNEL: tl.constexpr,
    H_TILE: tl.constexpr,
    NUM_H_TILES: tl.constexpr,
    NUM_UNION_CHUNKS: tl.constexpr,
    CHUNK_SIZE_U: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    """One program = (query-tile, kv-head, head-tile) x one union chunk.

    Grid: ``(all_qtile * NUM_UNION_CHUNKS, num_kv_heads * NUM_H_TILES)``.
    """
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_uc_qt = tl.program_id(0)
    pid_kht = tl.program_id(1)

    pid_qt = pid_uc_qt % all_qtile
    pid_uc = pid_uc_qt // all_qtile
    pid_kh = pid_kht // NUM_H_TILES
    pid_ht = pid_kht % NUM_H_TILES

    # Per-tile varlen mapping (host-precomputed, cheap PyTorch -- no in-kernel
    # reverse search, same trick as _prefill_bnsd_score_kernel).
    q_start = tl.load(qt_to_qstart_ptr + pid_qt).to(tl.int32)
    prefix = tl.load(qt_prefix_ptr + pid_qt).to(tl.int32)
    within_base = tl.load(qt_within_base_ptr + pid_qt).to(tl.int32)
    last_qcount = tl.load(qt_last_qcount_ptr + pid_qt).to(tl.int32)
    seq_len = tl.minimum(tl.load(qt_seq_lens_ptr + pid_qt).to(tl.int32), max_kv_len)

    pid_h_base = pid_kh * gqa_group_size + pid_ht * H_TILE

    # Flattened (token-in-tile, head-in-tile) -> [BSQ*H_TILE], kept 1D: a
    # [BSQ, H] int32 2D tensor with a small H fails the Ascend TBE stride-
    # alignment check ("cannot align 0 axis").
    off_qh = tl.arange(0, BSQ_KERNEL * H_TILE)  # [BSQ*H_TILE]
    qi = off_qh // H_TILE  # token-in-tile [0, BSQ_KERNEL)
    hhi = off_qh % H_TILE  # head-in-tile [0, H_TILE)

    q_token = q_start + qi  # global q-token index [BSQ*H_TILE]
    head_idx = pid_h_base + hhi  # global q-head index [BSQ*H_TILE]
    # Absolute KV position of each query token (causal boundary).
    q_pos = prefix + within_base + qi  # [BSQ*H_TILE]
    row_valid = (qi < last_qcount) & (hhi < H_TILE) & (q_token < total_q)

    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    dim_mask = off_d < head_dim

    # Q: [BSQ*H_TILE, D]
    q_offsets = (
        q_token[:, None] * stride_q_n
        + head_idx[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=row_valid[:, None] & (off_d[None, :] < head_dim),
        other=0.0,
    )

    # Sink belongs only to chunk 0 so it is counted once across split-union chunks.
    if HAS_SINK and pid_uc == 0:
        sink_offsets = (
            head_idx[:, None] * stride_sink_h + off_d[None, :] * stride_sink_d
        )
        sink = tl.load(
            sink_ptr + sink_offsets,
            mask=row_valid[:, None] & (off_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        qsink = tl.sum(q.to(tl.float32) * sink, axis=1) * sm_scale
        m_i = qsink
        lse_i = qsink
    else:
        m_i = tl.full((BSQ_KERNEL * H_TILE,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BSQ_KERNEL * H_TILE,), float("-inf"), dtype=tl.float32)

    acc_o = tl.zeros((BSQ_KERNEL * H_TILE, BLOCK_SIZE_D), dtype=tl.float32)

    ui_base = union_idx_ptr + pid_kh * stride_ui_h + pid_qt * stride_ui_qt
    um_base = union_mask_ptr + pid_kh * stride_um_h + pid_qt * stride_um_qt
    chunk_start_u = pid_uc * CHUNK_SIZE_U

    # Iterate the union blocks assigned to this chunk. Invalid slots are -1
    # padded (sentinel) and skipped via ``valid_block`` -- mirror the decode
    # kernel's topk loop (topk_sparse_decode.py:234-302).
    for step in tl.range(CHUNK_SIZE_U):
        u_pos = chunk_start_u + step
        in_range = u_pos < u_max

        logical_block = tl.load(
            ui_base + u_pos * stride_ui_u, mask=in_range, other=-1
        ).to(tl.int32)
        valid_block = logical_block >= 0

        physical_block = tl.load(
            block_table_ptr + pid_qt * stride_bt_qt + logical_block * stride_bt_n,
            mask=valid_block,
            other=0,
        ).to(tl.int64)

        key_pos = logical_block * block_size + off_n  # [N]
        pos_mask = valid_block & (key_pos < seq_len)

        # K: [D, N] -- gathered ONCE, shared across the whole BSQ*H_TILE query tile.
        k_offsets = (
            physical_block * stride_k_block
            + off_n[None, :] * stride_k_offset
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_offsets,
            mask=(off_d[:, None] < head_dim) & pos_mask[None, :],
            other=0.0,
        )

        # [BSQ*H_TILE, D] @ [D, N] -> [BSQ*H_TILE, N] -- 2D dot, NO 3D reshape.
        qk = tl.dot(q, k) * sm_scale

        if BSQ_KERNEL == 1:
            # BSQ=1: the union IS this token's own topk (all blocks are causal-
            # valid by indexer construction), so neither the causal vector nor the
            # per-token-in-union mask is needed -- structurally identical to the
            # decode kernel. (These two masks are the dominant overhead at BSQ>1;
            # constexpr-folded out here.)
            qk = tl.where(pos_mask[None, :], qk, float("-inf"))
        else:
            # (a) causal: query token must be >= key position (broadcast heads).
            causal = q_pos[:, None] >= key_pos[None, :]
            # (b) per-token-in-union (load-bearing): this union block must be in
            # token qi's own per-token topk. Gathered by qi (token-in-tile).
            umask = tl.load(
                um_base + qi * stride_um_qi + u_pos * stride_um_u,
                mask=row_valid & in_range,
                other=0,
            )
            qk = tl.where(
                causal & pos_mask[None, :] & (umask != 0)[:, None], qk, float("-inf")
            )

        # online softmax recurrence (verbatim from decode main, on [BSQ*H_TILE]).
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.where(
            valid_block,
            tl.exp(qk - m_ij[:, None]),
            tl.zeros((BSQ_KERNEL * H_TILE, BLOCK_SIZE_N), dtype=tl.float32),
        )
        l_ij = tl.sum(p, axis=1)

        acc_o_scale = tl.where(
            valid_block,
            tl.exp(m_i - m_ij),
            tl.full((BSQ_KERNEL * H_TILE,), 1.0, dtype=tl.float32),
        )

        # V: [N, D]
        v_offsets = (
            physical_block * stride_v_block
            + off_n[:, None] * stride_v_offset
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d
        )
        v = tl.load(
            v_cache_ptr + v_offsets,
            mask=pos_mask[:, None] & (off_d[None, :] < head_dim),
            other=0.0,
        )

        acc_o_new = acc_o * acc_o_scale[:, None] + tl.dot(p.to(v.dtype), v)
        lse_i_new = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)

        acc_o = tl.where(valid_block, acc_o_new, acc_o)
        m_i = tl.where(valid_block, m_ij, m_i)
        lse_i = tl.where(valid_block, lse_i_new, lse_i)

    # Final scale (empty tiles keep lse_i=-inf -> clean zeros).
    scale = tl.where(
        lse_i > float("-inf"),
        tl.exp(m_i - lse_i),
        tl.zeros_like(lse_i),
    )
    acc_o = acc_o * scale[:, None]

    o_offsets = (
        pid_uc * stride_o_c
        + q_token[:, None] * stride_o_n
        + head_idx[:, None] * stride_o_h
        + off_d[None, :] * stride_o_d
    )
    tl.store(
        o_ptr + o_offsets,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=row_valid[:, None] & (off_d[None, :] < head_dim),
    )

    l_offsets = (
        pid_uc * stride_l_c + q_token * stride_l_n + head_idx * stride_l_h
    )
    tl.store(
        lse_ptr + l_offsets,
        lse_i.to(lse_ptr.dtype.element_ty),
        mask=row_valid,
    )


# =============================================================================
# Host prep
# =============================================================================


def _build_qtile_mappings(
    bsq_kernel: int,
    cu_seqlens: torch.Tensor,  # [bs+1] (extend q-token cumulative)
    seq_lens: torch.Tensor,  # [bs] total KV len
    prefix_lens: torch.Tensor,  # [bs]
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,  # [bs]
    page_size: int,
    max_blocks: int,
    device,
) -> Tuple[
    torch.Tensor,  # qt_to_qstart [all_qtile]
    torch.Tensor,  # qt_prefix   [all_qtile]
    torch.Tensor,  # qt_within_base [all_qtile]
    torch.Tensor,  # qt_last_qcount [all_qtile]
    torch.Tensor,  # qt_seq_lens [all_qtile]
    torch.Tensor,  # block_table [all_qtile, max_blocks]
    int,  # all_qtile
    torch.Tensor,  # global_token_index [all_qtile, BSQ_KERNEL] (for the topk gather)
]:
    """Precompute per-query-tile varlen mappings (cheap PyTorch).

    Tiles NEVER cross request boundaries (``qt_per_req = ceil(q_len / BSQ)``),
    so every mapping is uniform-within-tile except the per-token causal position
    (derived from ``prefix + within_base + qi``) and the tail valid count.
    """
    cu_q = cu_seqlens.to(device=device, dtype=torch.long)
    seq_lens_l = seq_lens.to(device=device, dtype=torch.long)
    prefix_lens_l = prefix_lens.to(device=device, dtype=torch.long)
    reqs = req_pool_indices.to(device=device, dtype=torch.long)

    q_lens = cu_q[1:] - cu_q[:-1]  # [bs]
    qt_per_req = (q_lens + bsq_kernel - 1) // bsq_kernel  # [bs]
    all_qtile = int(qt_per_req.sum().item())

    if all_qtile == 0:
        empty = torch.empty(0, device=device, dtype=torch.int32)
        empty_bt = torch.empty((0, max_blocks), device=device, dtype=torch.int32)
        gt = torch.empty((0, bsq_kernel), device=device, dtype=torch.long)
        return empty, empty, empty, empty, empty, empty_bt, 0, gt

    qt_to_req = reqs.repeat_interleave(qt_per_req)  # [all_qtile]
    qt_prefix = prefix_lens_l.repeat_interleave(qt_per_req)  # [all_qtile]
    cu_q_extend = cu_q[:-1].repeat_interleave(qt_per_req)  # [all_qtile]

    # Local tile index within its request -> within-extend offset of first token.
    cu_qt = torch.zeros_like(qt_per_req)
    cu_qt[1:] = qt_per_req[:-1].cumsum(0)
    arange_all = torch.arange(all_qtile, device=device, dtype=torch.long)
    tile_idx_in_req = arange_all - cu_qt.repeat_interleave(qt_per_req)  # [all_qtile]
    qt_within_base = tile_idx_in_req * bsq_kernel  # [all_qtile]
    qt_to_qstart = cu_q_extend + qt_within_base  # [all_qtile] global q-token index

    q_len_per_tile = q_lens.repeat_interleave(qt_per_req)  # [all_qtile]
    qt_last_qcount = torch.clamp(
        q_len_per_tile - qt_within_base, min=0, max=bsq_kernel
    )  # [all_qtile]
    # Last token's KV len (for pos_mask / physical slot validity).
    qt_seq_lens = qt_prefix + qt_within_base + qt_last_qcount  # [all_qtile]

    # block_table[qt, blk] = physical page of logical block blk of qt's request.
    max_cols = req_to_token.shape[1]
    blk_cols = torch.arange(max_blocks, device=device, dtype=torch.long) * page_size
    blk_cols = blk_cols.clamp(max=max_cols - 1)
    token_slots = req_to_token[qt_to_req][:, blk_cols]  # [all_qtile, max_blocks]
    block_table = (token_slots // page_size).to(torch.int32)

    # Per-(tile, qi) global q-token index, clamped to a valid token for the tail
    # (invalid qi rows are masked at the q-load/store; their gathered topk is
    # harmless because those rows never contribute).
    qi_idx = torch.arange(bsq_kernel, device=device, dtype=torch.long)  # [BSQ]
    global_token = (qt_to_qstart[:, None] + qi_idx[None, :]).clamp(
        min=0, max=max(int(cu_q[-1].item()) - 1, 0)
    )  # [all_qtile, BSQ]

    return (
        qt_to_qstart.to(torch.int32),
        qt_prefix.to(torch.int32),
        qt_within_base.to(torch.int32),
        qt_last_qcount.to(torch.int32),
        qt_seq_lens.to(torch.int32),
        block_table,
        all_qtile,
        global_token.to(torch.int64),
    )


def _build_union_topk_and_mask(
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk] int32, -1 padded
    global_token: torch.Tensor,  # [all_qtile, BSQ_KERNEL] long
    bsq_kernel: int,
    num_total_blocks: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Vectorized union + per-token-in-union mask (no per-tile Python loop).

    Returns:
        union_idx   [num_kv_heads, all_qtile, u_max] int32 (-1 padded)
        union_mask  [num_kv_heads, all_qtile, BSQ_KERNEL, u_max] int8 (1 if the
                    union block is in that token's own topk)
        u_max       int (power-of-2 constexpr)
    """
    num_kv_heads, total_q, topk = topk_idx.shape
    all_qtile = global_token.shape[0]

    # Per-(tile, qi) topk via gather along the token dim.
    # T[:, gt, :] -> [num_kv_heads, all_qtile, BSQ_KERNEL, topk]
    t_tile = topk_idx[:, global_token, :].to(torch.int32)

    # Flatten the tile's tokens -> candidate list [num_kv_heads, all_qtile, BSQ*topk].
    t_flat = t_tile.reshape(num_kv_heads, all_qtile, bsq_kernel * topk)

    big = 1 << 30  # sentinel larger than any valid block id; -1 sentinels -> big
    cand = t_flat.long()
    cand = torch.where(cand >= 0, cand, torch.full_like(cand, big))
    cand_sorted, _ = cand.sort(dim=-1)  # ascending; big at the tail

    # First-occurrence mask in the sorted list (dedup).
    first = torch.ones_like(cand_sorted[..., :1], dtype=torch.bool)
    is_first = torch.cat(
        [first, cand_sorted[..., 1:] != cand_sorted[..., :-1]], dim=-1
    ) & (cand_sorted < big)  # [num_kv_heads, all_qtile, BSQ*topk]

    # Compact: drop non-first occurrences, re-sort so valid uniques lead.
    masked = torch.where(is_first, cand_sorted, big)
    masked_sorted, _ = masked.sort(dim=-1)
    union_count = (masked_sorted < big).sum(dim=-1)  # [num_kv_heads, all_qtile]
    u_max_actual = int(union_count.max().item()) if union_count.numel() else 1
    u_max = max(1, min(triton.next_power_of_2(max(u_max_actual, 1)), _MAX_U_MAX))

    union_idx = masked_sorted[..., :u_max].to(torch.int32)  # [kv, qt, u_max]
    union_idx = torch.where(
        union_idx >= big, torch.full_like(union_idx, -1), union_idx
    )

    # Per-token-in-union mask: is union_idx[kv,qt,u] in t_tile[kv,qt,qi,:]?
    # [kv, qt, BSQ, topk, 1] vs [kv, qt, 1, 1, u_max] -> any over topk.
    mask = (
        t_tile[..., None] == union_idx[:, :, None, None, :]
    ).any(dim=-2)  # [kv, qt, BSQ, u_max]
    # Exclude matches against -1 padding in union_idx (shouldn't matter -- the
    # kernel skips -1 blocks via valid_block -- but be safe/exact).
    mask = mask & (union_idx[:, :, None, :] >= 0)
    union_mask = mask.to(torch.int8)
    return union_idx.contiguous(), union_mask.contiguous(), u_max


# =============================================================================
# Wrapper
# =============================================================================


@torch.no_grad()
def flash_prefill_union_main_bnsd(
    q: torch.Tensor,  # [total_q, num_q_heads, head_dim]
    sink: Optional[torch.Tensor],  # optional [num_q_heads, head_dim]
    k_cache_bnsd: torch.Tensor,  # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache_bnsd: torch.Tensor,  # same
    topk_idx: torch.Tensor,  # [num_kv_heads (or num_q_heads), total_q, topk]
    cu_seqlens: torch.Tensor,  # [bs+1]
    seq_lens: torch.Tensor,  # [bs]
    prefix_lens: torch.Tensor,  # [bs]
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,  # [bs]
    bsq_kernel: int,
    page_size: int,
    sm_scale: Optional[float] = None,
    h_tile: Optional[int] = None,
    max_num_union_chunks: int = 8,
    fallback_cb=None,  # optional () -> Tensor, invoked when union too large
) -> torch.Tensor:
    """Prefill main sparse attention via the per-query-tile union kernel.

    Returns ``o`` ``[total_q, num_q_heads, head_dim]``. Consumes the indexer's
    per-token ``topk_idx`` (post GQA-reduce + forced-block merge) and tiles
    ``bsq_kernel`` consecutive query tokens per program, sharing the union gather.
    """
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache_bnsd.dtype == q.dtype
    assert v_cache_bnsd.dtype == q.dtype
    assert k_cache_bnsd.shape == v_cache_bnsd.shape

    total_q, num_q_heads, head_dim = q.shape
    _, block_size_from_cache, num_kv_heads, cache_head_dim = k_cache_bnsd.shape
    block_size = page_size
    assert block_size_from_cache == block_size
    assert cache_head_dim == head_dim
    assert num_q_heads % num_kv_heads == 0
    gqa_group_size = num_q_heads // num_kv_heads

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    topk_idx = _normalize_topk_idx_for_gqa(
        topk_idx, num_q_heads, num_kv_heads, gqa_group_size
    )  # [num_kv_heads, total_q, topk]
    assert topk_idx.shape[1] == total_q

    if h_tile is None:
        # Pick the largest power-of-2 <= gqa_group_size that keeps the tile in UB.
        # Constraint (default shapes): BSQ*h_tile <= 32.
        h_tile = _DEFAULT_H_TILE
    assert gqa_group_size % h_tile == 0, (
        f"h_tile={h_tile} must divide gqa_group_size={gqa_group_size}"
    )
    num_h_tiles = gqa_group_size // h_tile

    device = q.device
    max_kv_len = req_to_token.shape[1] * block_size
    max_blocks = (int(max_kv_len) + block_size - 1) // block_size
    # Cap to the actual max seq len (cheaper tensors) -- prefix+extend.
    if seq_lens.numel():
        max_blocks = min(max_blocks, (int(seq_lens.max().item()) + block_size - 1) // block_size)
    num_total_blocks = max_blocks

    (
        qt_to_qstart,
        qt_prefix,
        qt_within_base,
        qt_last_qcount,
        qt_seq_lens,
        block_table,
        all_qtile,
        global_token,
    ) = _build_qtile_mappings(
        bsq_kernel,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        req_to_token,
        req_pool_indices,
        page_size,
        max_blocks,
        device,
    )

    if all_qtile == 0:
        return torch.zeros_like(q)

    union_idx, union_mask, u_max = _build_union_topk_and_mask(
        topk_idx, global_token, bsq_kernel, num_total_blocks, device
    )

    # Pathological union blowup guard -> fall back to the per-query decode main.
    if u_max >= _MAX_U_MAX and (
        int(
            (union_idx >= 0).sum(dim=-1).max().item()
        )
        >= _MAX_U_MAX
    ):
        if fallback_cb is not None:
            return fallback_cb()
        # No fallback supplied: clamp (drop trailing blocks) -- last-resort.
        u_max = _MAX_U_MAX // 2
        union_idx = union_idx[..., :u_max]
        union_mask = union_mask[..., :u_max]

    num_union_chunks = _choose_num_union_chunks(
        all_qtile, num_kv_heads, num_h_tiles, u_max, max_num_union_chunks
    )
    chunk_size_u = (u_max + num_union_chunks - 1) // num_union_chunks
    # Ascend BiSheng ConvertLinalgRToBinary crash at static loop width 1
    # (topk_sparse_decode.py:509-515) -- keep a minimum width of 2.
    chunk_size_u = max(2, chunk_size_u)

    out = torch.empty_like(q)
    single_chunk = num_union_chunks == 1
    # o_partial / lse_partial are indexed by the GLOBAL query token (q is
    # [total_q, ...]); tiles partition the tokens, so [C, total_q, num_q_heads, *].
    if single_chunk:
        o_partial = out.view(1, total_q, num_q_heads, head_dim)
    else:
        o_partial = torch.empty(
            (num_union_chunks, total_q, num_q_heads, head_dim),
            dtype=q.dtype,
            device=device,
        )
    lse_partial = torch.empty(
        (num_union_chunks, total_q, num_q_heads),
        dtype=torch.float32,
        device=device,
    )

    sink_arg = sink if sink is not None else q  # typed placeholder for dead branch

    grid = (all_qtile * num_union_chunks, num_kv_heads * num_h_tiles)
    _gqa_share_sparse_prefill_union_kernel[grid](
        q,
        sink_arg,
        k_cache_bnsd,
        v_cache_bnsd,
        block_table,
        union_idx,
        union_mask,
        o_partial,
        lse_partial,
        qt_to_qstart,
        qt_prefix,
        qt_within_base,
        qt_last_qcount,
        qt_seq_lens,
        total_q,
        all_qtile,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        u_max,
        max_kv_len,
        block_size,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sink_arg.stride(0) if sink is not None else 0,
        sink_arg.stride(1) if sink is not None else 0,
        k_cache_bnsd.stride(0),
        k_cache_bnsd.stride(1),
        k_cache_bnsd.stride(2),
        k_cache_bnsd.stride(3),
        v_cache_bnsd.stride(0),
        v_cache_bnsd.stride(1),
        v_cache_bnsd.stride(2),
        v_cache_bnsd.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        union_idx.stride(0),
        union_idx.stride(1),
        union_idx.stride(2),
        union_mask.stride(0),
        union_mask.stride(1),
        union_mask.stride(2),
        union_mask.stride(3),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        BSQ_KERNEL=bsq_kernel,
        H_TILE=h_tile,
        NUM_H_TILES=num_h_tiles,
        NUM_UNION_CHUNKS=num_union_chunks,
        CHUNK_SIZE_U=chunk_size_u,
        HAS_SINK=sink is not None,
        num_warps=_SPARSE_DECODE_NW,
        num_stages=_SPARSE_DECODE_NS,
    )

    if not single_chunk:
        merge_grid = (total_q, num_q_heads)
        _merge_topk_attn_out_bnsd_kernel[merge_grid](
            o_partial,
            lse_partial,
            out,
            head_dim,
            o_partial.stride(0),
            o_partial.stride(1),
            o_partial.stride(2),
            o_partial.stride(3),
            lse_partial.stride(0),
            lse_partial.stride(1),
            lse_partial.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            NUM_TOPK_CHUNKS=num_union_chunks,
            num_warps=_MERGE_NW,
            num_stages=_MERGE_NS,
        )

    return out
