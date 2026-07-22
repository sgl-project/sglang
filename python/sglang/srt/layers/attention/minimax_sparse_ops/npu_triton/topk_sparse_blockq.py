"""NPU Ascend PREFILL main sparse-attention kernel: per-query-block SHARED topk.

This is the prefill counterpart of ``_gqa_share_sparse_decode_bnsd_kernel`` in
``topk_sparse_decode.py``. The decode kernel is correct for prefill but slow:
``_forward_npu_triton_prefill`` flattens the ~3072 extend tokens into 3072
per-query rows, so each program independently re-gathers its topk blocks and
recomputes the per-block page/address math. Profiling
(``_gqa_share_sparse_decode_bnsd_kernel`` = 15.88 ms/call, 43% of prefill;
``aic_mac_ratio=2.2%``, ``cube_utilization=99.2%`` stalled, ``aic_scalar=53%``)
shows the kernel is scalar/address-bound on the paged topk gather, not
compute-bound.

The fix mirrors the GPU ground-truth ``_gqa_share_sparse_fwd_kernel``
(``minimax_sparse_ops/prefill/topk_sparse.py``): one program processes a *pack
group* of ``PACK_Q`` consecutive in-request tokens that **share one topk block
list**. Each selected K/V block is loaded once and reused across all
``PACK_Q * gqa`` Q rows of the pack group, amortising the per-block address
scalar and the K/V HBM traffic by roughly ``PACK_Q``. There are NO membership
bits: every selected block contributes to every row, exactly like the GPU
kernel. (The earlier "union decode" / kimi PACK_Q+bits designs either regressed
1.71x at 64K or hit Ascend union-kernel bugs; this design avoids both by
sharing the topk at the indexer granularity instead of unioning per-token
topk.)

The per-pack-group topk is derived on the host from the already-computed
per-query topk by taking the latest-in-pack token's list (its causal window is
a superset of the earlier tokens'; the earlier rows mask the few extra near-tail
blocks via per-row causal, with no extra K/V traffic).

Ascend-safe conventions are reused verbatim from the validated decode kernel:
direct ``req_to_token`` page lookup with ``sanitize_page_ids``, 2D ``tl.dot``
(NO 3D reshape of the dot result), fp32 online softmax in natural exp/log (so
``sm_scale`` is the unscaled ``head_dim**-0.5``), ``-inf - -inf`` NaN guards on
fully-masked/sentinel blocks, phantom-row clamping for partial-tail pack
groups, single-chunk fast path that aliases the partial-output buffer to the
final output (skipping the merge kernel), and ``CHUNK_SIZE_T >= 2`` to dodge the
BiSheng ``ConvertLinalgRToBinary`` crash.

UB budget (gqa=16, head_dim=128, bf16, UB=192 KB; V load sequenced after the p
compute like the decode multi-block path): peak = 32768 + PACK_Q*24704 bytes.
PACK_Q=2/num_stages=2 -> 80 KB (default); PACK_Q=4/num_stages=1 -> 128 KB
(opt-in long KV); PACK_Q=8 overflows -> hard cap PACK_Q=4.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.topk_sparse_decode import (
    _choose_num_topk_chunks,
)


@triton.heuristics(
    {
        "BLOCK_ROWS": lambda args: triton.next_power_of_2(
            args["PACK_Q"] * args["gqa_group_size"]
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _gqa_share_sparse_prefill_blockq_kernel(
    q_ptr,  # [total_q, num_q_heads, head_dim]
    k_cache_ptr,  # [num_pages, block_size, num_kv_heads, head_dim]
    v_cache_ptr,  # same shape
    req_to_token_ptr,  # [num_requests, max_context]
    req_pool_indices_ptr,  # [num_pack_groups] -- one request per pack group
    idx_ptr,  # [num_kv_heads, num_pack_groups, max_topk] int32, -1-padded
    q_start_ptr,  # [num_pack_groups] int32 -- absolute token start of the group
    q_end_ptr,  # [num_pack_groups] int32 -- exclusive upper bound (cu_seqlens[r+1])
    seq_lens_ptr,  # [total_q] int32 -- per-query causal KV length
    o_ptr,  # [num_topk_chunks, total_q, num_q_heads, head_dim]
    lse_ptr,  # [num_topk_chunks, total_q, num_q_heads]
    scratch_ptr,  # [num_pack_groups, BLOCK_ROWS, head_dim] -- scalar-store spill fix
    # scalars
    num_pack_groups,
    gqa_group_size,
    head_dim,
    max_topk,
    max_kv_len,
    total_q,
    num_pages,
    max_req_to_token_cols,
    # block / scaling
    block_size: tl.constexpr,
    sm_scale,
    PACK_Q: tl.constexpr,
    # strides
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_v_block,
    stride_v_offset,
    stride_v_h,
    stride_v_d,
    stride_rtt_r,
    stride_rtt_t,
    stride_ti_h,
    stride_ti_g,
    stride_ti_t,
    stride_qstart_g,
    stride_qend_g,
    stride_req_g,
    stride_o_c,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_n,
    stride_l_h,
    stride_sc_g,
    stride_sc_kh,
    stride_sc_r,
    stride_sc_d,
    # meta
    NUM_TOPK_CHUNKS: tl.constexpr,
    CHUNK_SIZE_T: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    SANITIZE_PAGE_IDS: tl.constexpr,
    SCALAR_STORE: tl.constexpr,
):
    """PACK_Q-query shared-topk sparse attention (prefill main path).

    Grid: ``(num_pack_groups * num_topk_chunks, num_kv_heads)``. ``pid_g`` is the
    pack group; ``pid_c`` the topk chunk (almost always 0 at prefill scale, so
    the merge kernel is skipped via the single-chunk fast path).

    Rows of the Q tile are the flattened (token-within-pack, head-within-group)
    pairs, built via 1D divmod -- never a materialized [PACK_Q, gqa] 2D int32
    (Ascend TBE rejects the stride alignment). All PACK_Q*gqa rows share the
    same topk block list; per-row causal masking uses each row's own seq_len.
    """
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_gc = tl.program_id(0)
    pid_kh = tl.program_id(1)

    pid_g = pid_gc % num_pack_groups
    pid_c = pid_gc // num_pack_groups
    pid_h_base = pid_kh * gqa_group_size

    q_start = tl.load(q_start_ptr + pid_g * stride_qstart_g).to(tl.int32)
    q_end = tl.load(q_end_ptr + pid_g * stride_qend_g).to(tl.int32)

    # Flattened (token-in-pack, head-in-group) -> [PACK_Q*gqa], kept 1D: a
    # [PACK_Q, gqa] 2D int32 with small gqa fails the Ascend TBE stride-alignment
    # check, so build the flat indices directly via divmod (same trick as the
    # validated prefill score kernel, prefill_block_score.py:122-134).
    off_r = tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS]
    row_q = off_r // gqa_group_size  # token index within the pack
    row_h = off_r % gqa_group_size  # head index within the GQA group

    q_token_raw = q_start + row_q  # absolute token index; valid rows in [0,total_q)
    # q_end (= cu_seqlens[r+1]) bounds rows to THIS request's real tokens so
    # phantom rows of a partial-tail pack group (extend not a PACK_Q multiple)
    # are masked.
    row_valid = (q_token_raw < q_end) & (row_h < gqa_group_size)
    # q_token drives the seq_lens and Q loads and MUST stay in [0, total_q). When
    # the LAST request's extend_len is not a BSQ multiple, the score path's final
    # BSQ query-block has trailing "phantom" pack groups whose q_start >= total_q
    # (no real tokens exist there). The old `maximum(q_start, minimum(...))` clamp
    # lower-bounded q_token at q_start, so for those groups q_token = q_start >=
    # total_q -> OOB read of seq_lens/q -> async aicore MTE fault (the evalscope
    # crash). Route INVALID rows to token 0 (their output is masked by row_valid,
    # so the loaded value is irrelevant); valid rows keep q_token_raw, which is
    # already < q_end <= total_q. Same safe-address pattern as the scatter kernel.
    q_token = tl.where(row_valid, tl.minimum(q_token_raw, total_q - 1), 0)
    head_flat = pid_h_base + row_h  # actual q-head index per row

    off_d = tl.arange(0, BLOCK_SIZE_D)  # [D]
    off_n = tl.arange(0, BLOCK_SIZE_N)  # [N]
    dim_mask = off_d < head_dim

    # Per-row causal length (each token has its own). Loaded at the clamped
    # q_token so phantom rows never index past total_q; their qk is masked out
    # by row_valid downstream so the garbage value never contributes.
    row_seq = tl.load(seq_lens_ptr + q_token, mask=row_valid, other=0).to(tl.int32)

    # Q load: [BLOCK_ROWS, D]  (clamped q_token keeps reads in-request)
    q_offsets = (
        q_token[:, None] * stride_q_n
        + head_flat[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=row_valid[:, None] & dim_mask[None, :],
        other=0.0,
    )

    m_i = tl.full((BLOCK_ROWS,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_ROWS,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_ROWS, BLOCK_SIZE_D), 0.0, dtype=tl.float32)

    idx_base = idx_ptr + pid_kh * stride_ti_h + pid_g * stride_ti_g
    chunk_start_topk = pid_c * CHUNK_SIZE_T

    for step in tl.range(CHUNK_SIZE_T):
        topk_pos = chunk_start_topk + step
        in_topk_range = topk_pos < max_topk

        logical_block = tl.load(
            idx_base + topk_pos * stride_ti_t, mask=in_topk_range, other=-1
        ).to(tl.int32)
        valid_block = logical_block >= 0

        # Direct req_to_token page lookup (validated in the decode main kernel).
        # req_idx is loop-invariant; kept in-loop to match the validated decode
        # kernel (topk_sparse_decode.py:390-394 measured hoisting as neutral).
        req_idx = tl.load(req_pool_indices_ptr + pid_g * stride_req_g).to(tl.int64)
        safe_logical = tl.maximum(logical_block, 0)
        token_col = tl.minimum(safe_logical * block_size, max_req_to_token_cols - 1)
        token_slot = tl.load(
            req_to_token_ptr + req_idx * stride_rtt_r + token_col * stride_rtt_t,
            mask=valid_block,
            other=0,
        ).to(tl.int64)
        physical_block = token_slot // block_size
        if SANITIZE_PAGE_IDS:
            physical_block = tl.minimum(tl.maximum(physical_block, 0), num_pages - 1)

        pos = logical_block * block_size + off_n  # [N]
        # Column validity (shared across all rows: same selected block); per-row
        # causal applied on qk below via row_seq. The K load uses the column-only
        # mask to avoid a [M,N] mask on the gather (Ascend may materialise it in
        # UB); garbage columns get -inf through pos_mask in the qk step. Same
        # split as the GPU kernel (topk_sparse.py:187-216).
        pos_mask_col = valid_block & (pos < max_kv_len)  # [N]
        pos_mask = (pos[None, :] < row_seq[:, None]) & pos_mask_col[None, :]  # [M, N]

        # K load: [D, N]
        k_offsets = (
            physical_block * stride_k_block
            + off_n[None, :] * stride_k_offset
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_offsets,
            mask=dim_mask[:, None] & pos_mask_col[None, :],
            other=0.0,
        )

        # 2D dot -- NO 3D reshape (Ascend TBE miscompiles / 1500x slows it).
        qk = tl.dot(q, k) * sm_scale  # [BLOCK_ROWS, N]
        qk = tl.where(pos_mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        # Per-row "this block contributes" guard. Unlike the decode kernel --
        # where each query's own topk is causally relevant -- the SHARED topk
        # means an earlier-in-pack token (smaller causal window) can fully mask
        # a near-tail block that the later token selected. For such rows qk is
        # all -inf -> m_ij = -inf, and exp(qk - m_ij) = exp(-inf - -inf) = nan.
        # Guard on ``m_ij > -inf`` (per-row); a -1 sentinel topk slot makes
        # every row non-contributing (subsumes the scalar valid_block guard for
        # the softmax update; valid_block is still used for the gather mask).
        row_contributes = m_ij > float("-inf")  # [BLOCK_ROWS]
        p = tl.where(
            row_contributes[:, None],
            tl.exp(qk - m_ij[:, None]),
            tl.zeros((BLOCK_ROWS, BLOCK_SIZE_N), dtype=tl.float32),
        )
        l_ij = tl.sum(p, axis=1)

        # V load sequenced AFTER the p compute so its UB live range starts as
        # K's ends (K and V tiles together would pressure the 192 KB UB at
        # larger PACK_Q). Mirrors the decode multi-block path
        # (topk_sparse_decode.py:614-627).
        v_offsets = (
            physical_block * stride_v_block
            + off_n[:, None] * stride_v_offset
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d
        )
        v = tl.load(
            v_cache_ptr + v_offsets,
            mask=pos_mask_col[:, None] & dim_mask[None, :],
            other=0.0,
        )

        acc_o_scale = tl.where(
            row_contributes,
            tl.exp(m_i - m_ij),
            tl.full((BLOCK_ROWS,), 1.0, dtype=tl.float32),
        )
        acc_o_new = acc_o * acc_o_scale[:, None] + tl.dot(p.to(v.dtype), v)
        lse_i_new = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)

        acc_o = tl.where(row_contributes[:, None], acc_o_new, acc_o)
        m_i = tl.where(row_contributes, m_ij, m_i)
        lse_i = tl.where(row_contributes, lse_i_new, lse_i)

    # Final scale; empty rows (lse_i=-inf, e.g. all-sentinel topk) -> clean 0.
    scale = tl.where(
        lse_i > float("-inf"),
        tl.exp(m_i - lse_i),
        tl.zeros_like(lse_i),
    )
    acc_o = acc_o * scale[:, None]

    # Store partial output. Two modes:
    #  * SCALAR_STORE: write acc_o to a scratch buffer [num_pack_groups, BLOCK_ROWS,
    #    D] at a SCALAR base (pid_g) + off_r + off_d. This avoids referencing
    #    q_token/head_flat/row_valid in the store, so those prologue vectors are
    #    dead-stripped after the Q load -> the loop body stops spilling registers
    #    (the vector store otherwise keeps them live across the loop -> 3.5x
    #    slower, confirmed by aiv_mte3 store-pipe 0.037 -> 0.542). A separate
    #    lightweight scatter kernel then distributes scratch -> o.
    #  * else: vector store straight to o (the spill-prone path).
    if SCALAR_STORE:
        sc_offsets = (
            pid_g * stride_sc_g
            + pid_kh * stride_sc_kh
            + off_r[:, None] * stride_sc_r
            + off_d[None, :] * stride_sc_d
        )
        tl.store(
            scratch_ptr + sc_offsets,
            acc_o.to(scratch_ptr.dtype.element_ty),
            mask=row_valid[:, None] & dim_mask[None, :],
        )
    else:
        o_offsets = (
            pid_c * stride_o_c
            + q_token[:, None] * stride_o_n
            + head_flat[:, None] * stride_o_h
            + off_d[None, :] * stride_o_d
        )
        tl.store(
            o_ptr + o_offsets,
            acc_o.to(o_ptr.dtype.element_ty),
            mask=row_valid[:, None] & dim_mask[None, :],
        )

    l_offsets = pid_c * stride_l_c + q_token * stride_l_n + head_flat * stride_l_h
    tl.store(
        lse_ptr + l_offsets,
        lse_i.to(lse_ptr.dtype.element_ty),
        mask=row_valid,
    )


@triton.jit
def _scatter_blockq_out_kernel(
    scratch_ptr,  # [num_pack_groups, BLOCK_ROWS, head_dim]
    o_ptr,  # [total_q, num_q_heads, head_dim]
    q_start_ptr,  # [num_pack_groups]
    q_end_ptr,  # [num_pack_groups]
    head_dim,
    num_pack_groups,
    # strides
    stride_sc_g, stride_sc_kh, stride_sc_r, stride_sc_d,
    stride_o_n, stride_o_h, stride_o_d,
    stride_qstart_g, stride_qend_g,
    # meta
    GQA: tl.constexpr, PACK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Scatter scratch -> o with a per-pack-token CONTIGUOUS store.

    For each pack token q, its [GQA, D] slice in scratch[g, q*GQA:..., :] goes to
    out[q_start+q, pid_h_base:pid_h_base+GQA, :] -- a contiguous load + contiguous
    store at a SCALAR base (q_start+q). No [BLOCK_ROWS] q_token/head_flat vectors
    and no big live tile -> spill-free and fast (a pure remapped copy).
    """
    pid_g = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_h_base = pid_kh * GQA

    q_start = tl.load(q_start_ptr + pid_g * stride_qstart_g).to(tl.int32)
    q_end = tl.load(q_end_ptr + pid_g * stride_qend_g).to(tl.int32)
    off_h = tl.arange(0, GQA)  # [GQA]
    off_d = tl.arange(0, BLOCK_D)  # [D]
    dim_mask = off_d < head_dim
    sc_base = pid_g * stride_sc_g + pid_kh * stride_sc_kh

    for q in tl.static_range(PACK_Q):
        qtok = q_start + q  # scalar
        valid_q = qtok < q_end  # scalar
        # Clamp the store ADDRESS to an in-bounds token (0) when invalid. The
        # mask still suppresses the write, but on Ascend a masked store whose
        # computed DDR address is out of range (qtok >= total_q, which happens
        # for partial-tail pack groups whose unclamped q_start >= total_q when
        # the last request's length is not a multiple of BSQ) still raises an
        # async aicore MTE exception -- caught at the next stream sync (e.g. the
        # deep_ep dispatch), crashing the run. Routing the address through a
        # safe value avoids the OOB address entirely.
        qtok_safe = tl.where(valid_q, qtok, 0)
        src = tl.load(
            scratch_ptr + sc_base + (q * GQA + off_h)[:, None] * stride_sc_r + off_d[None, :] * stride_sc_d,
            mask=valid_q & dim_mask[None, :],
            other=0.0,
        )
        tl.store(
            o_ptr + qtok_safe * stride_o_n + (pid_h_base + off_h)[:, None] * stride_o_h + off_d[None, :] * stride_o_d,
            src,
            mask=valid_q & dim_mask[None, :],
        )


@torch.no_grad()
def flash_prefill_bnsd_blockq_sparse(
    q: torch.Tensor,  # [total_q, num_q_heads, head_dim]
    k_cache_bnsd: torch.Tensor,  # [num_pages, block_size, num_kv_heads, head_dim]
    v_cache_bnsd: torch.Tensor,  # same shape
    topk_idx_blockq: torch.Tensor,  # [num_kv_heads, num_pack_groups, max_topk] int32, -1-padded
    seq_lens: torch.Tensor,  # [total_q] per-query causal KV length
    q_start: torch.Tensor,  # [num_pack_groups] int32
    q_end: torch.Tensor,  # [num_pack_groups] int32
    req_pool_indices: torch.Tensor,  # [num_pack_groups]
    block_size: int,  # == page_size
    sm_scale: Optional[float],
    pack_q: int,
    req_to_token: torch.Tensor,
    max_num_blocks: int,
    num_pages: int,
    sanitize_page_ids: bool = True,
    num_topk_chunks: Optional[int] = None,
    max_num_topk_chunks: int = 8,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """PACK_Q shared-topk prefill sparse attention. Returns [total_q, QH, D].

    One program per (pack group, kv head). A pack group is ``pack_q`` consecutive
    in-request extend tokens sharing one topk block list
    (``topk_idx_blockq[:, g, :]``). Each selected K/V block is loaded once and
    reused across all ``pack_q * gqa`` Q rows. No membership bits.

    Args:
        topk_idx_blockq: per-pack-group topk, shape
            ``[num_kv_heads, num_pack_groups, max_topk]`` int32, ``-1``-padded.
            Typically derived by gathering the latest-in-pack token's per-query
            topk (its causal window is a superset of the earlier tokens').
        seq_lens: per-query causal KV length ``[total_q]``. Each row uses its
            own token's length for causal masking.
        q_start / q_end: absolute token bounds of each pack group;
            ``q_end == cu_seqlens[r+1]`` of the owning request.
        req_pool_indices: owning request of each pack group (for the direct
            ``req_to_token`` page lookup).
    """
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache_bnsd.dtype == q.dtype
    assert v_cache_bnsd.dtype == q.dtype
    assert k_cache_bnsd.shape == v_cache_bnsd.shape
    assert pack_q in (1, 2, 4), f"pack_q must be 1/2/4 (UB cap), got {pack_q}"

    total_q, num_q_heads, head_dim = q.shape
    _, block_size_from_cache, num_kv_heads, cache_head_dim = k_cache_bnsd.shape
    assert block_size_from_cache == block_size
    assert cache_head_dim == head_dim
    assert num_q_heads % num_kv_heads == 0
    gqa_group_size = num_q_heads // num_kv_heads

    num_pack_groups = q_start.shape[0]
    assert q_end.shape[0] == num_pack_groups
    assert req_pool_indices.shape[0] == num_pack_groups
    assert topk_idx_blockq.shape[0] == num_kv_heads
    assert topk_idx_blockq.shape[1] == num_pack_groups
    assert topk_idx_blockq.dtype == torch.int32
    assert seq_lens.shape[0] == total_q
    assert seq_lens.dtype == torch.int32

    max_topk = topk_idx_blockq.shape[2]
    max_kv_len = int(max_num_blocks) * block_size
    max_req_to_token_cols = req_to_token.shape[1]

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    if num_topk_chunks is None:
        num_topk_chunks = _choose_num_topk_chunks(
            num_pack_groups,
            num_kv_heads,
            max_topk,
            max_num_topk_chunks=max_num_topk_chunks,
        )
    else:
        num_topk_chunks = int(num_topk_chunks)
    assert num_topk_chunks >= 1

    chunk_size_topk = (max_topk + num_topk_chunks - 1) // num_topk_chunks
    # Ascend BiSheng crashes at ConvertLinalgRToBinary when CHUNK_SIZE_T=1
    # (LLVM ERROR: operation destroyed but still has uses). Floor at 2; the
    # extra iteration is safely masked by ``topk_pos < max_topk`` and
    # ``logical_block >= 0``. Same guard as the decode kernel
    # (topk_sparse_decode.py:886-892).
    chunk_size_topk = max(2, chunk_size_topk)

    # Single-chunk fast path: the kernel writes the already-final-normalised
    # output (it applies the final exp(m_i - lse_i) scale before storing), so the
    # merge kernel would be a no-op copy. Alias o_partial to the final output
    # buffer (pid_c is always 0 -> pid_c*stride_o_c == 0) and skip the merge
    # launch + the [C,total_q,QH,D] temp allocation.
    single_chunk = num_topk_chunks == 1
    out = torch.empty_like(q)
    if single_chunk:
        o_partial = out.view(1, total_q, num_q_heads, head_dim)
    else:
        o_partial = torch.empty(
            (num_topk_chunks, total_q, num_q_heads, head_dim),
            dtype=q.dtype,
            device=q.device,
        )
    # lse_partial is always written (small, [C,total_q,QH]); unused on the
    # single-chunk path but required as a store target.
    lse_partial = torch.empty(
        (num_topk_chunks, total_q, num_q_heads),
        dtype=torch.float32,
        device=q.device,
    )

    grid = (num_pack_groups * num_topk_chunks, num_kv_heads)
    # SCALAR_STORE spill fix: write acc_o to a scratch [num_pack_groups,
    # BLOCK_ROWS, D] buffer with a scalar pid_g base, so the prologue vectors
    # (q_token/head_flat/row_valid) are dead-stripped after the Q load and the
    # loop stops spilling. Caller must scatter scratch -> o when scalar_store
    # (perf test skips the scatter; correctness validated separately).
    block_rows = triton.next_power_of_2(pack_q * gqa_group_size)
    # [num_pack_groups, num_kv_heads, BLOCK_ROWS, D] -- kv-head dimension avoids
    # the per-pid_kh programs racing on the same scratch slots.
    scratch = torch.empty(
        (num_pack_groups, num_kv_heads, block_rows, head_dim),
        dtype=q.dtype, device=q.device,
    )
    _gqa_share_sparse_prefill_blockq_kernel[grid](
        q,
        k_cache_bnsd,
        v_cache_bnsd,
        req_to_token,
        req_pool_indices,
        topk_idx_blockq,
        q_start,
        q_end,
        seq_lens,
        o_partial,
        lse_partial,
        scratch,
        num_pack_groups,
        gqa_group_size,
        head_dim,
        max_topk,
        max_kv_len,
        total_q,
        num_pages,
        max_req_to_token_cols,
        block_size,
        sm_scale,
        pack_q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache_bnsd.stride(0),
        k_cache_bnsd.stride(1),
        k_cache_bnsd.stride(2),
        k_cache_bnsd.stride(3),
        v_cache_bnsd.stride(0),
        v_cache_bnsd.stride(1),
        v_cache_bnsd.stride(2),
        v_cache_bnsd.stride(3),
        req_to_token.stride(0),
        req_to_token.stride(1),
        topk_idx_blockq.stride(0),
        topk_idx_blockq.stride(1),
        topk_idx_blockq.stride(2),
        q_start.stride(0),
        q_end.stride(0),
        req_pool_indices.stride(0),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        scratch.stride(0),
        scratch.stride(1),
        scratch.stride(2),
        scratch.stride(3),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        CHUNK_SIZE_T=chunk_size_topk,
        BLOCK_SIZE_N=block_size,
        SANITIZE_PAGE_IDS=sanitize_page_ids,
        SCALAR_STORE=single_chunk,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Spill fix: when single_chunk, the main kernel wrote acc_o to a scratch
    # buffer (scalar pid_g base) instead of straight to o (vector store would
    # spill). Scatter scratch -> o now, with a tiny spill-free kernel.
    if single_chunk:
        _scatter_blockq_out_kernel[(num_pack_groups, num_kv_heads)](
            scratch,
            o_partial,
            q_start,
            q_end,
            head_dim,
            num_pack_groups,
            scratch.stride(0),
            scratch.stride(1),
            scratch.stride(2),
            scratch.stride(3),
            o_partial.stride(1),
            o_partial.stride(2),
            o_partial.stride(3),
            q_start.stride(0),
            q_end.stride(0),
            GQA=gqa_group_size,
            PACK_Q=pack_q,
            BLOCK_D=triton.next_power_of_2(head_dim),
            num_warps=4,
            num_stages=1,
        )

    if not single_chunk:
        # Reuse the validated decode merge kernel: online LSE merge across the
        # topk chunks. Import lazily to keep this module importable standalone.
        from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.topk_sparse_decode import (
            _merge_topk_attn_out_bnsd_kernel,
            _MERGE_NW,
            _MERGE_NS,
        )

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
            NUM_TOPK_CHUNKS=num_topk_chunks,
            num_warps=_MERGE_NW,
            num_stages=_MERGE_NS,
        )

    return out
