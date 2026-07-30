from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# --- Native Ascend block-sparse attention (npu_sparse_attention_score) ---
# Auto-detection: prefer the sgl-kernel-npu attentions plugin op
# torch.ops.attentions.npu_sparse_attention_score. It is engaged when the op is
# already registered, or when a lazy `import attentions` (which loads
# libPTAExtensionOPS.so and registers the op) succeeds. If unavailable the validated
# Triton split-K kernel is used.
_NATIVE_SPARSE_LOADED = False
# Resolved op handle: torch.ops.attentions.npu_sparse_attention_score (sgl-kernel-npu
# attentions plugin). None -> fall back to the Triton path.
_npu_sparse_attention_score_op = None


def _ensure_native_sparse_loaded():
    """Lazily resolve the native op handle once.

    Order: already-registered torch.ops.attentions op -> attempt `import attentions`
    (loads libPTAExtensionOPS.so and registers torch.ops.attentions.*) -> leave
    unresolved so the Triton path is used. Any failure is swallowed; the Triton path
    takes over with no side effects.
    """
    global _NATIVE_SPARSE_LOADED, _npu_sparse_attention_score_op
    if _NATIVE_SPARSE_LOADED:
        return
    # 1. Already registered by some other import (e.g. a prior plugin load).
    try:
        _npu_sparse_attention_score_op = (
            torch.ops.attentions.npu_sparse_attention_score
        )
        _NATIVE_SPARSE_LOADED = True
        return
    except Exception:
        pass
    # 2. Lazy: import the sgl-kernel-npu attentions plugin, which loads
    #    libPTAExtensionOPS.so and registers torch.ops.attentions.*.
    try:
        import attentions  # noqa: F401

        _npu_sparse_attention_score_op = (
            torch.ops.attentions.npu_sparse_attention_score
        )
    except Exception:
        # Plugin absent or failed to load -> fall back to the Triton path.
        _npu_sparse_attention_score_op = None
    _NATIVE_SPARSE_LOADED = True


def _is_native_sparse_available() -> bool:
    """Auto-detect whether the native Ascend block-sparse attention op is usable."""
    _ensure_native_sparse_loaded()
    return _npu_sparse_attention_score_op is not None


@triton.jit
def _native_sanitize_topk_kernel(
    sel_ptr,  # [num_kv_heads, batch, SLOTS] int32 (in-place OUT)
    select_num_idx_ptr,  # [num_kv_heads, batch] int32 (OUT)
    seq_lens_ptr,  # [batch] int32
    stride_sel_h,
    stride_sel_b,
    stride_sel_s,
    stride_sn_h,
    stride_sn_b,
    block_size: tl.constexpr,
    SLOTS: tl.constexpr,
):
    """One-launch sanitize for the native op's select_idx/select_num_idx.

    Per (kv_head, batch) program:
      1. sanitize: sel >= cdiv(seq_len, block_size) -> -1 (skip OOB; the verify
         cuda-graph score phase can emit sel >= max_blocks. Skipping matches the
         Triton main kernel's pos<seq_len mask).
      2. select_num_idx = count(sel >= 0).
    No fold/cap: the native op's kernel array (validPhysicalIds) is now [32] (was
    [16], which OOB'd at attend=17 -- fixed in the recompiled .o), so the op
    safely handles up to 32 attended blocks.
    """
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + pid_b)
    nblocks = tl.cdiv(seq_len, block_size)
    off = tl.arange(0, SLOTS)
    base = pid_h * stride_sel_h + pid_b * stride_sel_b
    sel = tl.load(sel_ptr + base + off * stride_sel_s)  # [SLOTS]
    sel = tl.where(sel >= nblocks, -1, sel)  # sanitize OOB
    count = tl.sum((sel >= 0).to(tl.int32), axis=0)  # scalar
    tl.store(sel_ptr + base + off * stride_sel_s, sel)
    tl.store(select_num_idx_ptr + pid_h * stride_sn_h + pid_b * stride_sn_b, count)


def _native_decode_main(
    q,
    k,
    v,
    topk_idx,
    seq_lens,
    block_size,
    sm_scale,
    block_table,
    req_to_token,
    req_pool_indices,
    max_num_blocks,
    num_kv_heads,
    head_dim,
):
    device = q.device
    batch_size = q.shape[0]
    if block_table is not None:
        bt = block_table.to(torch.int32)
    else:
        # Gather only the B x maxBlocks block-start slots (NOT the full B x max_ctx),
        # then // block_size -> physical page id per (request, logical block). The full
        # gather would dominate the native op's ~0.14ms cost.
        bidx = torch.arange(max_num_blocks, device=device, dtype=torch.int32)
        slots = req_to_token[
            req_pool_indices[:, None].to(torch.int64),
            (bidx * block_size)[None, :].to(torch.int64),
        ]
        bt = (slots // block_size).to(torch.int32)
    num_pages = k.shape[0]
    sel = topk_idx.to(torch.int32).clone()
    select_num_idx = torch.empty(
        (num_kv_heads, batch_size), dtype=torch.int32, device=q.device
    )
    _native_sanitize_topk_kernel[(num_kv_heads, batch_size)](
        sel,
        select_num_idx,
        seq_lens.to(torch.int32),
        sel.stride(0),
        sel.stride(1),
        sel.stride(2),
        select_num_idx.stride(0),
        select_num_idx.stride(1),
        block_size=block_size,
        SLOTS=sel.shape[-1],
        num_warps=1,
        num_stages=1,
    )
    # bt is already valid for attended slots (sel < per-query nblocks after sanitize
    # -> bt from the req_to_token gather is a valid page id in [0, num_pages));
    # unattended tail slots are not dereferenced by the native op (it attends only
    # [0, select_num_idx)). No clamp needed (the old safety net cost ~140us +
    # ~97MB graph pool per forward).
    # actual_seq_lengths_kv carries the per-request KV length. In the cuda-graph
    # path `seq_lens` is the layer-invariant int32 STATIC buffer
    # (_decode_seq_lens_i32_cg / _verify_meta_cg) that sglang refreshes OUTSIDE
    # graph replay each forward, so the captured kernel reads the live value on
    # replay -- it is graph-safe by construction (int32 .to() is a no-op alias).
    #
    # Do NOT MAX-pad this. The op's host tiling is shape-derived
    # (totalTaskNum = totalQTokens * kvHeads in sparse_attention_score_tiling.cpp
    # CalculateTaskSplit) and never reads this value, so padding gains nothing.
    # Worse, a constant MAX value makes the device kernel's KV-boundary check walk
    # past the real block_table on replay and OOB -> CCU instruction address check
    # error (507011). The workspace itself comes through the standard EXEC_NPU_CMD
    # NPU caching allocator (graph memory pool), which is what makes replay safe.
    actual_kv = seq_lens.to(torch.int32)
    _ensure_native_sparse_loaded()
    out = _npu_sparse_attention_score_op(
        q,
        k,
        v,
        sel,
        bt,
        select_num_idx=select_num_idx,
        actual_seq_lengths=torch.ones(batch_size, dtype=torch.int32, device=device),
        actual_seq_lengths_kv=actual_kv,
        num_key_value_heads=num_kv_heads,
        scale_value=sm_scale if sm_scale is not None else head_dim**-0.5,
        block_size=block_size,
        top_k=topk_idx.shape[-1],
        inner_precise=0,
    )
    return out


# =============================================================================
# Utilities
# =============================================================================


def _floor_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (int(x).bit_length() - 1)


# ---------------------------------------------------------------------------
# Tunable launch configs for the served decode kernels.
#
# Defaults are the validated (num_warps=4, num_stages=2). The bench/tuning
# scripts override these module globals before the first launch to sweep configs
# and land the winners here. They are NOT triton.autotune so that each shape
# bucket compiles to a single deterministic artifact (cuda-graph capture safe).
# ---------------------------------------------------------------------------
# NOTE: swept num_warps∈{2,4,8} × num_stages∈{2,3} via sweep_cfg.py on the real
# captured decode workload (mostly short seqs -> launch-bound). No config beat the
# validated (4,2) outside run-to-run noise, so 4/2 is kept. The hooks remain for
# re-tuning if the served batch/context-length profile shifts to compute-bound.
_SPARSE_DECODE_NW = 4
_SPARSE_DECODE_NS = 2
_MERGE_NW = 4
_MERGE_NS = 2

# MiniMax-M3 selects 16 scored blocks and appends one forced local block.  On
# Ascend, the resulting short TopK list is launch/merge bound for C1 decode and
# target verification.  Keep small batches in the single-chunk fast path, which
# avoids the partial-output allocation and merge-kernel launch entirely.
_MINIMAX_SINGLE_CHUNK_MAX_TOPK = 17
_MINIMAX_SINGLE_CHUNK_MAX_BATCH = 4


def _get_vectorcore_num_safe() -> int:
    """Return the Ascend NPU vector-core count (sglang-native).

    Read ``num_vectorcore`` from triton's active-driver device properties for the
    current NPU. Falls back to 32 off-NPU or if the property is unavailable.
    """
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        n = int(props.get("num_vectorcore", -1))
    except Exception:
        # Conservative fallback.
        return 32
    return max(1, n) if n > 0 else 32


def _choose_num_topk_chunks(
    batch_size: int,
    num_kv_heads: int,
    max_topk: int,
    max_num_topk_chunks: int = 8,
) -> int:
    """Choose split-topk chunks in an SGLang-like but Ascend-conservative way."""
    if max_topk <= 1:
        return 1

    if (
        num_kv_heads == 1
        and batch_size <= _MINIMAX_SINGLE_CHUNK_MAX_BATCH
        and max_topk <= _MINIMAX_SINGLE_CHUNK_MAX_TOPK
    ):
        return 1

    num_vectorcore = _get_vectorcore_num_safe()
    # SGLang CUDA uses TARGET_GRID=256 for this sparse decode kernel.
    # Ascend: aim to SATURATE the vector cores (1 program/core, no 4x oversubscribe)
    # -- bench (B=8 x nkvh=4 on 32-vc) showed num_topk_chunks=1 is ~18% faster than
    # the prior *4 over-split (256us -> 211us): once B*nkvh >= vc, extra chunks only
    # add wave + merge overhead. Small batches still split (nchunks = vc/(B*nkvh)).
    target_grid = num_vectorcore
    target = max(1, target_grid // max(1, batch_size * num_kv_heads))
    target = min(max_topk, max_num_topk_chunks, target)
    return _floor_power_of_2(target)


def _normalize_topk_idx_for_gqa(
    topk_idx: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    gqa_group_size: int,
) -> torch.Tensor:
    """Ensure topk_idx has shape [num_kv_heads, batch_size, topk].

    The sparse GQA-share decode kernel uses one topk list per KV head, shared by
    all query heads in the corresponding GQA group.

    If a per-query-head topk tensor [num_q_heads, batch_size, topk] is provided,
    we take the first q-head from each GQA group.
    """
    if topk_idx.shape[0] == num_kv_heads:
        return topk_idx.contiguous()

    if topk_idx.shape[0] == num_q_heads:
        batch_size = topk_idx.shape[1]
        max_topk = topk_idx.shape[2]
        return topk_idx.view(num_kv_heads, gqa_group_size, batch_size, max_topk)[
            :, 0, :, :
        ].contiguous()

    raise AssertionError(
        "topk_idx first dimension must be either num_kv_heads or num_q_heads, "
        f"got {topk_idx.shape[0]}, num_kv_heads={num_kv_heads}, "
        f"num_q_heads={num_q_heads}"
    )


# =============================================================================
# MiniMax decode local-block postprocess
# =============================================================================


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
    }
)
@triton.jit
def _append_local_block_to_topk_idx_kernel(
    topk_idx_ptr,  # [num_kv_heads, batch_size, topk]
    seq_lens_ptr,  # [batch_size]
    out_ptr,  # [num_kv_heads, batch_size, topk + 1]
    batch_size,
    topk,
    num_blocks,
    block_size: tl.constexpr,
    stride_topk_h,
    stride_topk_b,
    stride_topk_t,
    stride_out_h,
    stride_out_b,
    stride_out_t,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Append the causal local block (query-block tiled, BSQ queries/program).

    Replaces the per-query grid (batch_size programs) with a query-block-tiled
    grid (cdiv(batch_size, BSQ) programs) -- at prefill scale (batch_size=
    total_q~4096) this cuts the program count ~BSQ-fold, eliminating the launch-
    overhead cliff that left the per-query version at 291us/call (4096 trivial
    programs). Each program validates BSQ queries' topk candidates and appends
    the causal local block. Same semantics as the per-query version.
    """
    pid_qb = tl.program_id(0)
    pid_h = tl.program_id(1)
    off_q = tl.arange(0, BLOCK_SIZE_Q)  # [BSQ]
    off_t = tl.arange(0, BLOCK_SIZE_T)  # [topk_pow2]
    q_tok = pid_qb * BLOCK_SIZE_Q + off_q  # [BSQ]
    q_valid = q_tok < batch_size

    # Per-query causal position + local block.
    seq_len = tl.load(seq_lens_ptr + q_tok, mask=q_valid, other=1).to(tl.int32)
    query_pos = tl.maximum(seq_len - 1, 0)
    local_blk = tl.minimum(query_pos // block_size, num_blocks - 1)

    # Load candidates [BSQ, topk] and validate.
    in_off = (
        pid_h * stride_topk_h
        + q_tok[:, None] * stride_topk_b
        + off_t[None, :] * stride_topk_t
    )
    cand = tl.load(
        topk_idx_ptr + in_off,
        mask=q_valid[:, None] & (off_t[None, :] < topk),
        other=-1,
    ).to(tl.int32)
    valid = (
        (cand >= 0) & (cand < num_blocks) & (cand * block_size <= query_pos[:, None])
    )
    cand_out = tl.where(valid, cand, -1)

    # Store validated candidates [BSQ, topk].
    out_off = (
        pid_h * stride_out_h
        + q_tok[:, None] * stride_out_b
        + off_t[None, :] * stride_out_t
    )
    tl.store(
        out_ptr + out_off, cand_out, mask=q_valid[:, None] & (off_t[None, :] < topk)
    )

    # Append local block at slot topk: -1 if already present (dedup).
    local_present = tl.sum((cand_out == local_blk[:, None]).to(tl.int32), axis=1) > 0
    out_local = tl.where(local_present, -1, local_blk)
    tl.store(
        out_ptr + pid_h * stride_out_h + q_tok * stride_out_b + topk * stride_out_t,
        tl.where(q_valid, out_local, -1),
        mask=q_valid,
    )


@torch.no_grad()
def append_local_block_to_topk_idx(
    topk_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    num_blocks: int,
) -> torch.Tensor:
    """Fuse MiniMax's ``init=0, local=1`` decode top-k postprocess.

    The generic fallback first permutes candidates to ``[B, KVH, TopK]``, then
    validates them and appends the causal local block through several PyTorch
    operators, before permuting back. This NPU path consumes and produces the
    GQA kernel layout directly. It intentionally preserves candidate order and
    only removes a local block when that exact valid candidate already exists.

    Query-block-tiled (BSQ=16): at prefill scale (batch_size=total_q~4096) this
    cuts the grid from 4096 to 256 programs (-83% launch overhead). At decode
    scale (batch_size~40) the grid is small enough that tiling is neutral.
    """
    assert topk_idx.ndim == 3
    assert topk_idx.dtype == torch.int32
    assert topk_idx.is_contiguous()
    assert seq_lens.ndim == 1
    assert seq_lens.shape[0] == topk_idx.shape[1]
    assert seq_lens.is_contiguous()
    assert block_size > 0
    assert num_blocks > 0

    num_kv_heads, batch_size, topk = topk_idx.shape
    out = torch.empty(
        (num_kv_heads, batch_size, topk + 1),
        dtype=topk_idx.dtype,
        device=topk_idx.device,
    )
    BSQ = 16
    grid = (triton.cdiv(batch_size, BSQ), num_kv_heads)
    _append_local_block_to_topk_idx_kernel[grid](
        topk_idx,
        seq_lens,
        out,
        batch_size,
        topk,
        num_blocks,
        block_size=block_size,
        stride_topk_h=topk_idx.stride(0),
        stride_topk_b=topk_idx.stride(1),
        stride_topk_t=topk_idx.stride(2),
        stride_out_h=out.stride(0),
        stride_out_b=out.stride(1),
        stride_out_t=out.stride(2),
        BLOCK_SIZE_Q=BSQ,
        num_warps=4,
        num_stages=1,
    )
    return out


# =============================================================================
# Sparse BNSD Decode Kernel
# =============================================================================


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    }
)
@triton.jit
def _gqa_share_sparse_decode_bnsd_kernel(
    q_ptr,  # [B, QH, D]
    sink_ptr,  # optional [QH, D]
    k_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    v_cache_ptr,  # [NBLOCKS, BLOCK, KVH, D]
    block_table_ptr,  # [B, max_num_blocks] or typed direct-map placeholder
    req_to_token_ptr,  # [num_requests, max_context] in direct-map mode
    req_pool_indices_ptr,  # [B] in direct-map mode
    idx_ptr,  # [KVH, B, max_topk]
    o_ptr,  # [C, B, QH, D]
    lse_ptr,  # [C, B, QH]
    seq_lens,  # [B]
    # shape
    batch_size,
    gqa_group_size,
    head_dim,
    max_topk,
    max_kv_len,
    # block/scaling
    block_size: tl.constexpr,
    sm_scale,
    # strides
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_sink_h,
    stride_sink_d,
    stride_k_block,
    stride_k_offset,
    stride_k_h,
    stride_k_d,
    stride_v_block,
    stride_v_offset,
    stride_v_h,
    stride_v_d,
    stride_bt_b,
    stride_bt_n,
    stride_rtt_r,
    stride_rtt_t,
    max_req_to_token_cols,
    num_pages,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    # meta
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    CHUNK_SIZE_T: tl.constexpr,
    # Number of selected (topk) blocks gathered into one K/V tile + one dot per
    # loop step. 1 == the original per-block path (bit-identical). 2/4 amortise
    # the per-step scalar overhead (idx gather, page lookup, address math,
    # softmax update) that leaves the prefill main-attention launch
    # overhead-bound at total_q~3072. Same block set per query -> same math,
    # only the online-softmax grouping changes (fp-tail level).
    BLOCKS_PER_STEP: tl.constexpr,
    # Hoist the whole chunk's topk-idx load + page-id gather into a vectorized
    # prologue ([BLOCK_SIZE_T] lanes, one memory round trip) and recover the
    # per-step scalars with a where+sum select, instead of the per-step serial
    # idx-load -> page-gather -> K-load dependency chain. Same blocks, same
    # order, same math -- only address generation is restructured.
    PREFETCH_IDX: tl.constexpr,
    HAS_SINK: tl.constexpr,
    USE_DIRECT_PAGE_LOOKUP: tl.constexpr,
    SANITIZE_PAGE_IDS: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_N >= block_size)

    pid_bc = tl.program_id(0)
    pid_kh = tl.program_id(1)

    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size

    seq_len = tl.minimum(tl.load(seq_lens + pid_b).to(tl.int32), max_kv_len)

    # TopK list base for this KV head and request.
    #
    # Do NOT compute real_topk with tl.sum(topk_vals >= 0) here. On Ascend this
    # pattern can compile but appear to behave like only one block is active,
    # producing an output magnitude similar to attending a single block. Instead
    # iterate over the fixed per-chunk topk range and mask out -1 entries.
    idx_base = idx_ptr + pid_kh * stride_ti_h + pid_b * stride_ti_b

    chunk_start_topk = pid_c * CHUNK_SIZE_T

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)

    dim_mask = off_d < head_dim

    # Q: [H, D]
    q_offsets = (
        pid_b * stride_q_b
        + (pid_h + off_h[:, None]) * stride_q_h
        + off_d[None, :] * stride_q_d
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
        other=0.0,
    )

    # Sink belongs only to chunk 0 so it is counted once across split-topk chunks.
    if HAS_SINK and pid_c == 0:
        sink_offsets = (pid_h + off_h[:, None]) * stride_sink_h + off_d[
            None, :
        ] * stride_sink_d
        sink = tl.load(
            sink_ptr + sink_offsets,
            mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        qsink = tl.sum(q.to(tl.float32) * sink, axis=1) * sm_scale
        m_i = qsink
        lse_i = qsink
    else:
        _neg = -1.0e30
        m_i = tl.full((BLOCK_SIZE_H,), _neg, dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_H,), _neg, dtype=tl.float32)

    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0.0, dtype=tl.float32)

    # req_idx (req_pool_indices[pid_b]) is loop-invariant, but hoisting it before
    # the loop is NEUTRAL (~215us hoisted vs ~211us in-loop, min-of-3; the
    # redundant scalar load is negligible). An earlier 7057us reading was box
    # degradation (post-crash NPU state), NOT a hoist codegen regression. Keep
    # in-loop (simpler; no benefit to hoist).
    # Iterate over the fixed topk slice assigned to this chunk. The actual valid
    # length is encoded by -1 sentinels in topk_idx.
    if BLOCKS_PER_STEP == 1:
        off_t_pf = tl.arange(0, BLOCK_SIZE_T)
        if PREFETCH_IDX:
            # Vectorized prologue: one gather for all of this chunk's selected
            # blocks, one for their physical page ids. Afterwards the K/V loads
            # in the loop depend only on register data, so the memory pipeline
            # can run ahead instead of serializing idx->page->K per step.
            chunk_end_topk_pf = tl.minimum(chunk_start_topk + CHUNK_SIZE_T, max_topk)
            topk_pos_all = chunk_start_topk + off_t_pf
            logical_all = tl.load(
                idx_base + topk_pos_all * stride_ti_t,
                mask=topk_pos_all < chunk_end_topk_pf,
                other=-1,
            ).to(tl.int32)
            valid_pf = logical_all >= 0
            safe_pf = tl.maximum(logical_all, 0)
            if USE_DIRECT_PAGE_LOOKUP:
                req_idx_pf = tl.load(req_pool_indices_ptr + pid_b).to(tl.int64)
                token_cols_pf = tl.minimum(
                    safe_pf * block_size, max_req_to_token_cols - 1
                )
                token_slots_pf = tl.load(
                    req_to_token_ptr
                    + req_idx_pf * stride_rtt_r
                    + token_cols_pf * stride_rtt_t,
                    mask=valid_pf,
                    other=0,
                ).to(tl.int64)
                phys_pf = token_slots_pf // block_size
                if SANITIZE_PAGE_IDS:
                    phys_pf = tl.minimum(tl.maximum(phys_pf, 0), num_pages - 1)
            else:
                phys_pf = tl.load(
                    block_table_ptr + pid_b * stride_bt_b + safe_pf * stride_bt_n,
                    mask=valid_pf,
                    other=0,
                ).to(tl.int64)
            phys_pf32 = phys_pf.to(tl.int32)
        for step in tl.range(CHUNK_SIZE_T):
            if PREFETCH_IDX:
                # Dynamic register-vector index via where+sum ([T] int ops).
                logical_block = tl.sum(
                    tl.where(off_t_pf == step, logical_all, 0), axis=0
                )
                physical_block = tl.sum(
                    tl.where(off_t_pf == step, phys_pf32, 0), axis=0
                ).to(tl.int64)
                valid_block = logical_block >= 0
            else:
                topk_pos = chunk_start_topk + step
                in_topk_range = topk_pos < max_topk

                logical_block = tl.load(
                    idx_base + topk_pos * stride_ti_t,
                    mask=in_topk_range,
                    other=-1,
                ).to(tl.int32)
                valid_block = logical_block >= 0

                if USE_DIRECT_PAGE_LOOKUP:
                    req_idx = tl.load(req_pool_indices_ptr + pid_b).to(tl.int64)
                    safe_logical_block = tl.maximum(logical_block, 0)
                    token_col = tl.minimum(
                        safe_logical_block * block_size, max_req_to_token_cols - 1
                    )
                    token_slot = tl.load(
                        req_to_token_ptr
                        + req_idx * stride_rtt_r
                        + token_col * stride_rtt_t,
                        mask=valid_block,
                        other=0,
                    ).to(tl.int64)
                    physical_block = token_slot // block_size
                    if SANITIZE_PAGE_IDS:
                        physical_block = tl.minimum(
                            tl.maximum(physical_block, 0), num_pages - 1
                        )
                else:
                    physical_block = tl.load(
                        block_table_ptr
                        + pid_b * stride_bt_b
                        + logical_block * stride_bt_n,
                        mask=valid_block,
                        other=0,
                    ).to(tl.int64)

            pos = logical_block * block_size + off_n
            pos_mask = valid_block & (pos < seq_len)

            # K: [D, N]
            k_offsets = (
                physical_block * stride_k_block
                + off_n[None, :] * stride_k_offset
                + pid_kh * stride_k_h
                + off_d[:, None] * stride_k_d
            )
            k = tl.load(
                k_cache_ptr + k_offsets,
                mask=dim_mask[:, None] & pos_mask[None, :],
                other=0.0,
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
                mask=pos_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )

            # [H, D] @ [D, N] -> [H, N]
            qk = tl.dot(q, k) * sm_scale
            qk = tl.where(pos_mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            # Direct path (no valid_block guard): m_ij is finite (finite-floor
            # init or qsink), so a sentinel slot's qk=-inf -> exp(-inf - m_ij)
            # = 0 (p=0) and exp(m_i - m_ij)=1 (acc_o no-op) naturally.
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            acc_o = acc_o * tl.exp(m_i - m_ij)[:, None] + tl.dot(p.to(v.dtype), v)
            lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)
            m_i = m_ij
    else:
        # Multi-block path: gather BLOCKS_PER_STEP selected blocks into one
        # [D, BPS*block_size] K tile / [BPS*block_size, D] V tile per step. All
        # per-column vectors (Triton cannot index a [BPS] tensor by column), so
        # the idx/page gathers are per-column (redundant x block_size but L2
        # hits) -- the same trick as _prefill_bnsd_score_kernel's P3 path.
        sub_id = off_n // block_size  # [N] which selected block in this step
        inn = off_n % block_size  # [N] token offset within that block
        chunk_end_topk = tl.minimum(chunk_start_topk + CHUNK_SIZE_T, max_topk)
        num_steps = tl.cdiv(chunk_end_topk - chunk_start_topk, BLOCKS_PER_STEP)
        for step in tl.range(num_steps, num_stages=1, disallow_acc_multi_buffer=True):
            topk_pos_col = chunk_start_topk + step * BLOCKS_PER_STEP + sub_id
            logical_block_col = tl.load(
                idx_base + topk_pos_col * stride_ti_t,
                mask=topk_pos_col < chunk_end_topk,
                other=-1,
            ).to(tl.int32)
            valid_col = logical_block_col >= 0
            safe_logical_col = tl.maximum(logical_block_col, 0)

            if USE_DIRECT_PAGE_LOOKUP:
                req_idx = tl.load(req_pool_indices_ptr + pid_b).to(tl.int64)
                token_col = tl.minimum(
                    safe_logical_col * block_size, max_req_to_token_cols - 1
                )
                token_slot = tl.load(
                    req_to_token_ptr
                    + req_idx * stride_rtt_r
                    + token_col * stride_rtt_t,
                    mask=valid_col,
                    other=0,
                ).to(tl.int64)
                physical_block_col = token_slot // block_size
                if SANITIZE_PAGE_IDS:
                    physical_block_col = tl.minimum(
                        tl.maximum(physical_block_col, 0), num_pages - 1
                    )
            else:
                physical_block_col = tl.load(
                    block_table_ptr
                    + pid_b * stride_bt_b
                    + safe_logical_col * stride_bt_n,
                    mask=valid_col,
                    other=0,
                ).to(tl.int64)

            pos = logical_block_col * block_size + inn
            pos_mask = valid_col & (pos < seq_len)

            # K: [D, BPS*block_size]
            k_offsets = (
                physical_block_col[None, :] * stride_k_block
                + inn[None, :] * stride_k_offset
                + pid_kh * stride_k_h
                + off_d[:, None] * stride_k_d
            )
            k = tl.load(
                k_cache_ptr + k_offsets,
                mask=dim_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )

            # [H, D] @ [D, BPS*block_size] -> [H, BPS*block_size]
            qk = tl.dot(q, k) * sm_scale
            qk = tl.where(pos_mask[None, :], qk, float("-inf"))

            # A step with at least one valid column behaves exactly like the
            # per-block path on those columns (invalid columns stay -inf ->
            # p=0); an all-invalid step must not touch the accumulator (it
            # would produce -inf - -inf = nan), mirroring valid_block above.
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            # Direct path: m_ij finite (finite-floor init or qsink) -> an
            # all-invalid step's qk=-inf yields exp(-inf - m_ij)=0 (p=0) and
            # exp(m_i - m_ij)=1 (acc_o no-op) naturally; no has_valid guard.
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            # V load is sequenced AFTER the qk/p phase so its UB live range
            # starts as K's ends (K and V tiles together overflow the 192KB UB
            # at BLOCK_SIZE_N >= 256 otherwise).
            v_offsets = (
                physical_block_col[:, None] * stride_v_block
                + inn[:, None] * stride_v_offset
                + pid_kh * stride_v_h
                + off_d[None, :] * stride_v_d
            )
            v = tl.load(
                v_cache_ptr + v_offsets,
                mask=pos_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )

            acc_o = acc_o * tl.exp(m_i - m_ij)[:, None] + tl.dot(p.to(v.dtype), v)
            lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)
            m_i = m_ij

    # Final scale.
    # Empty chunks keep lse_i=-inf and should output clean zeros.
    scale = tl.where(
        lse_i > float("-inf"),
        tl.exp(m_i - lse_i),
        tl.zeros_like(lse_i),
    )
    acc_o = acc_o * scale[:, None]

    # Store partial output: [C, B, QH, D]
    o_offsets = (
        pid_c * stride_o_c
        + pid_b * stride_o_b
        + (pid_h + off_h[:, None]) * stride_o_h
        + off_d[None, :] * stride_o_d
    )
    tl.store(
        o_ptr + o_offsets,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=(off_h[:, None] < gqa_group_size) & (off_d[None, :] < head_dim),
    )

    l_offsets = pid_c * stride_l_c + pid_b * stride_l_b + (pid_h + off_h) * stride_l_h
    tl.store(
        lse_ptr + l_offsets,
        lse_i.to(lse_ptr.dtype.element_ty),
        mask=off_h < gqa_group_size,
    )


# =============================================================================
# Merge split-topk sparse attention output
# =============================================================================


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _merge_topk_attn_out_bnsd_kernel(
    o_ptr,  # [C, B, QH, D]
    lse_ptr,  # [C, B, QH]
    out_ptr,  # [B, QH, D]
    head_dim,
    # strides
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    # meta
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    off_d = tl.arange(0, BLOCK_SIZE_D)

    m = tl.full((), float("-inf"), dtype=tl.float32)
    l = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.full((BLOCK_SIZE_D,), 0.0, dtype=tl.float32)

    for c in tl.static_range(0, NUM_TOPK_CHUNKS):
        lse_c = tl.load(
            lse_ptr + c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h
        )

        o_c = tl.load(
            o_ptr
            + c * stride_o_c
            + pid_b * stride_o_b
            + pid_h * stride_o_h
            + off_d * stride_o_d,
            mask=off_d < head_dim,
            other=0.0,
        ).to(tl.float32)

        # Avoid -inf - -inf -> NaN for all-empty chunks.
        valid = lse_c > float("-inf")
        m_new = tl.maximum(m, lse_c)

        scale_old = tl.where(
            m > float("-inf"),
            tl.exp(m - m_new),
            tl.zeros_like(m),
        )
        scale_new = tl.where(
            valid,
            tl.exp(lse_c - m_new),
            tl.zeros_like(lse_c),
        )

        acc = acc * scale_old + o_c * scale_new
        l = l * scale_old + scale_new
        m = m_new

    out = tl.where(l > 0.0, acc / l, acc)

    tl.store(
        out_ptr + pid_b * stride_out_b + pid_h * stride_out_h + off_d * stride_out_d,
        out.to(out_ptr.dtype.element_ty),
        mask=off_d < head_dim,
    )


# =============================================================================
# Python Wrapper
# =============================================================================


@torch.no_grad()
def flash_decode_bnsd_with_gqa_share_sparse(
    q: torch.Tensor,  # [batch_size, num_q_heads, head_dim]
    sink: Optional[torch.Tensor],  # optional [num_q_heads, head_dim]
    k_cache_bnsd: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache_bnsd: torch.Tensor,  # same shape
    block_table: Optional[torch.Tensor],  # [batch_size, max_num_blocks]
    seq_lens: torch.Tensor,  # [batch_size]
    block_size: int,
    topk_idx: torch.Tensor,  # [num_kv_heads or num_q_heads, batch_size, topk]
    sm_scale: Optional[float] = None,
    num_topk_chunks: Optional[int] = None,
    max_num_topk_chunks: int = 8,
    req_to_token: Optional[torch.Tensor] = None,
    req_pool_indices: Optional[torch.Tensor] = None,
    max_num_blocks: Optional[int] = None,
    num_pages: Optional[int] = None,
    sanitize_page_ids: bool = False,
    # Selected blocks fused into one K/V tile + dot per loop step (prefill
    # main-attention anti-overhead lever; 1 keeps the validated per-block
    # path). Must be a power of two so BLOCK_SIZE_N stays aligned.
    topk_blocks_per_step: int = 1,
    # Hoist the chunk's topk-idx + page-id gathers into a vectorized prologue
    # (breaks the per-step idx->page->K load dependency chain). A/B lever for
    # the latency/issue-bound prefill main-attention launch.
    prefetch_idx: bool = False,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    # Native Ascend aclnn op gate (auto-detected via _is_native_sparse_available).
    # True -> route to npu_sparse_attention_score; False -> Triton split-K path.
    use_native: bool = False,
) -> torch.Tensor:
    """Sparse decode attention using BNSD KV cache and precomputed topk blocks.

    This is the BNSD/Ascend-friendly counterpart of SGLang's
    flash_decode_with_gqa_share_sparse.

    Args:
        q:
            [batch_size, num_q_heads, head_dim].
        sink:
            Optional [num_q_heads, head_dim].
        k_cache_bnsd / v_cache_bnsd:
            [num_blocks, block_size, num_kv_heads, head_dim].
        block_table:
            [batch_size, max_num_blocks].
        seq_lens:
            [batch_size].
        block_size:
            KV block size.
        topk_idx:
            Prefer [num_kv_heads, batch_size, topk]. If [num_q_heads, batch_size,
            topk] is provided, the first q-head of each GQA group is used.
        num_topk_chunks:
            If None, choose dynamically. Otherwise must be power-of-two.

    Returns:
        o:
            [batch_size, num_q_heads, head_dim].
    """
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache_bnsd.dtype == q.dtype
    assert v_cache_bnsd.dtype == q.dtype
    assert k_cache_bnsd.shape == v_cache_bnsd.shape
    # Native Ascend block-sparse decode main (~3.3x faster than the Triton
    # split-K path on Ascend910_9362). Placed BEFORE the direct-page-lookup
    # asserts so the backend's hoisted block_table (passed alongside req_to_token)
    # does not trip `assert block_table is None`.
    #
    # cuda-graph replay-safe: the aclnn op runs through the STANDARD EXEC_NPU_CMD
    # (workspace from the NPU caching allocator -> graph memory pool, so its
    # address is pinned across replay), and actual_seq_lengths_kv is sglang's
    # static int32 seq_lens buffer refreshed out-of-graph each forward. The op's
    # host tiling is shape-derived (totalTaskNum = totalQTokens*kvHeads) and thus
    # identical at capture and replay. Engages when use_native=True (caller-gated
    # per path); False falls through to the Triton split-K path.
    if use_native:
        _nkvh = k_cache_bnsd.shape[2]
        if topk_idx.shape[0] == _nkvh and (
            block_table is not None or req_to_token is not None
        ):
            return _native_decode_main(
                q,
                k_cache_bnsd,
                v_cache_bnsd,
                topk_idx,
                seq_lens,
                block_size,
                sm_scale,
                block_table,
                req_to_token,
                req_pool_indices,
                max_num_blocks,
                _nkvh,
                q.shape[2],
            )

    use_direct_page_lookup = req_to_token is not None
    assert (req_pool_indices is not None) == use_direct_page_lookup
    if use_direct_page_lookup:
        assert block_table is None
        assert req_to_token.ndim == 2
        assert req_to_token.dtype in (torch.int32, torch.int64)
        assert req_pool_indices.ndim == 1
        assert req_pool_indices.dtype in (torch.int32, torch.int64)
        assert max_num_blocks is not None and max_num_blocks > 0
        assert num_pages is not None and num_pages > 0
    else:
        assert block_table is not None
        assert block_table.dtype in (torch.int32, torch.int64)

    batch_size, num_q_heads, head_dim = q.shape
    _, block_size_from_cache, num_kv_heads, cache_head_dim = k_cache_bnsd.shape

    assert block_size_from_cache == block_size
    assert cache_head_dim == head_dim
    assert num_q_heads % num_kv_heads == 0
    assert seq_lens.shape[0] == batch_size
    assert topk_idx.shape[1] == batch_size
    if use_direct_page_lookup:
        assert req_pool_indices.shape[0] == batch_size
        assert max_num_blocks * block_size <= req_to_token.shape[1]
        page_source = req_to_token
        page_source_rows = req_pool_indices
        max_kv_len = int(max_num_blocks) * block_size
        direct_num_pages = int(num_pages)
    else:
        assert block_table.shape[0] == batch_size
        page_source = block_table
        page_source_rows = seq_lens
        max_kv_len = block_table.shape[1] * block_size
        direct_num_pages = 1

    gqa_group_size = num_q_heads // num_kv_heads

    topk_idx = _normalize_topk_idx_for_gqa(
        topk_idx,
        num_q_heads,
        num_kv_heads,
        gqa_group_size,
    )

    max_topk = topk_idx.shape[2]

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    if num_topk_chunks is None:
        num_topk_chunks = _choose_num_topk_chunks(
            batch_size,
            num_kv_heads,
            max_topk,
            max_num_topk_chunks=max_num_topk_chunks,
        )
    else:
        num_topk_chunks = int(num_topk_chunks)

    assert num_topk_chunks >= 1
    assert (num_topk_chunks & (num_topk_chunks - 1)) == 0
    assert num_topk_chunks <= max(1, max_topk)

    chunk_size_topk = (max_topk + num_topk_chunks - 1) // num_topk_chunks
    # Ascend BiSheng can crash at ConvertLinalgRToBinary when CHUNK_SIZE_T=1
    # in this sparse-decode kernel:
    #   LLVM ERROR: operation destroyed but still has uses
    # Use a minimum static topk loop width of 2. Extra iterations are safely
    # masked by ``topk_pos < max_topk`` and ``logical_block >= 0``.
    # This keeps correctness unchanged while avoiding the backend corner case.
    chunk_size_topk = max(2, chunk_size_topk)

    blocks_per_step = max(1, int(topk_blocks_per_step))
    assert (blocks_per_step & (blocks_per_step - 1)) == 0
    # When several topk blocks are fused per step, the static loop width must
    # cover at least one fused tile so every selected block is visited.
    chunk_size_topk = max(chunk_size_topk, blocks_per_step)

    # Single-chunk fast path: with NUM_TOPK_CHUNKS==1 the decode kernel writes the
    # already-final-normalized output (it applies the final exp(m_i - lse_i) scale
    # before storing), so the merge kernel would be a no-op copy. Alias o_partial
    # to the final output buffer (pid_c is always 0 -> pid_c*stride_o_c == 0) and
    # skip the merge launch + the [C,B,QH,D] temp allocation.
    single_chunk = num_topk_chunks == 1
    out = torch.empty_like(q)
    if single_chunk:
        o_partial = out.view(1, batch_size, num_q_heads, head_dim)
    else:
        o_partial = torch.empty(
            (num_topk_chunks, batch_size, num_q_heads, head_dim),
            dtype=q.dtype,
            device=q.device,
        )
    # lse_partial is always written by the kernel; unused on the single-chunk path
    # but required as a store target (small, [C,B,QH]).
    lse_partial = torch.empty(
        (num_topk_chunks, batch_size, num_q_heads),
        dtype=torch.float32,
        device=q.device,
    )

    # Triton still type-checks pointer arguments in constexpr-dead branches on
    # some Ascend builds. Do not pass Python None as sink_ptr. Instead pass any
    # typed tensor pointer and control the real behavior with HAS_SINK.
    sink_arg = sink if sink is not None else q

    grid = (batch_size * num_topk_chunks, num_kv_heads)
    _gqa_share_sparse_decode_bnsd_kernel[grid](
        q,
        sink_arg,
        k_cache_bnsd,
        v_cache_bnsd,
        page_source,
        page_source,
        page_source_rows,
        topk_idx,
        o_partial,
        lse_partial,
        seq_lens,
        batch_size,
        gqa_group_size,
        head_dim,
        max_topk,
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
        page_source.stride(0),
        page_source.stride(1),
        req_to_token.stride(0) if use_direct_page_lookup else 0,
        req_to_token.stride(1) if use_direct_page_lookup else 0,
        req_to_token.shape[1] if use_direct_page_lookup else 1,
        direct_num_pages,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        BLOCK_SIZE_N=block_size * blocks_per_step,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        CHUNK_SIZE_T=chunk_size_topk,
        BLOCKS_PER_STEP=blocks_per_step,
        PREFETCH_IDX=prefetch_idx and blocks_per_step == 1,
        HAS_SINK=sink is not None,
        USE_DIRECT_PAGE_LOOKUP=use_direct_page_lookup,
        SANITIZE_PAGE_IDS=sanitize_page_ids,
        num_warps=_SPARSE_DECODE_NW if num_warps is None else num_warps,
        num_stages=_SPARSE_DECODE_NS if num_stages is None else num_stages,
    )

    if not single_chunk:
        merge_grid = (batch_size, num_q_heads)
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
