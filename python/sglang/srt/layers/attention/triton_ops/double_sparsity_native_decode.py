"""Native Double Sparsity sparse-decode kernels.

Replaces the FA3 page-table sparse path (`DSFlashAttentionAdaptor` +
`_ds_select_stage2_merge_kernel` + `ds_union_per_batch`) with a self-
contained sparse-decode pipeline seeded from PR #22992
(`flash_decode_sparse_attention_fwd`).

Pipeline per decode layer:
  1. **Score** (Triton): per `(bs, kv_head, BLOCK_T)` program, load Q_label,
     load K_label at `req_to_token[bs, t]` for `t` in the tile, write
     dot-product score into `att_out_approx[kv_head, bs, t]`. Positions in
     `[seq_len - 1, max_ctx)` and in the sink / recent windows are masked
     to -inf so they cannot be picked by topk.
  2. **Top-k** (`torch.topk`): one CUB call selects `top_k` logical positions
     per `(bs, kv_head)`. Output is `topk_logical[bs, kv_head, top_k]`.
  3. **Build selected_physical** (torch): map top-k logical → physical via
     `req_to_token`, append the sink range and recent range as physical ids
     (the recent range always covers the current decode position because
     `recent_tokens >= 1` is enforced at startup).
  4. **Sparse attention** (Triton, 2 stages): the v1 split-K + reduce
     decode kernel, lookup-free (`selected_physical` is direct physical ids,
     not logical). One selected set per `(bs, kv_head)`, shared by all Q
     heads in the GQA group.

Selection cost is bounded by `top_k`, not `seq_len`; sparse attention cost
is bounded by `total_selected = top_k + sink_tokens + recent_tokens`, not
`seq_len`. Both are the headline DS properties.

History-only scoring invariant: `K_label[layer L][seq_len - 1]` is written
by *this* layer's `attention_end`, AFTER attention runs. Selection at
layer L's `attention_begin` must therefore exclude position `seq_len - 1`
(it is unconditionally re-added via the recency window).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

NEG_INF = -float("inf")


# --------------------------------------------------------------------------- #
# Stage 0: score K_label against Q_label, mask sink/recent/oob to -inf.        #
# --------------------------------------------------------------------------- #


@triton.jit
def _ds_native_score_kernel(
    Q_Label,  # [bs, H_kv, S]                          fp32 / bf16
    K_Label_Buffer,  # [T_pool, H_kv, S]                      bf16 / fp32
    Req_to_tokens,  # [bs, max_ctx]                          int32
    B_Seqlen,  # [bs]                                   int64
    Att_Out,  # [bs, H_kv, max_ctx]                    fp32
    stride_qbs,
    stride_qh,
    stride_klt,
    stride_klh,
    stride_rt_bs,
    stride_ab,
    stride_ah,
    sm_scale,
    sink_tokens,  # int — runtime; mask [0, sink) to -inf
    recent_tokens,  # int — runtime; mask [seq - recent, max_ctx) to -inf
    max_ctx,  # int — runtime
    S: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """One program per (bs, kv_head, block_t). Scores K_label · Q_label.

    Masks invalid positions to -inf so the downstream `torch.topk` cannot
    pick them. Positions excluded:
      * `t >= seq_len - 1`             (history only; current decode pos
                                        is added back via recency)
      * `t < sink_tokens`              (always-keep; appended later)
      * `t >= seq_len - recent_tokens` (always-keep; appended later)
    """
    bs_idx = tl.program_id(0)
    kv_idx = tl.program_id(1)
    blk = tl.program_id(2)

    t_offs = blk * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    seq_len = tl.load(B_Seqlen + bs_idx).to(tl.int64)
    history_len = seq_len - 1  # exclude current decode pos
    rec_lo = seq_len - recent_tokens  # may be negative if seq < recent
    rec_lo = tl.maximum(rec_lo, 0)

    in_history = t_offs < history_len
    in_sink = t_offs < sink_tokens
    in_recent = t_offs >= rec_lo
    valid = in_history & (~in_sink) & (~in_recent)

    # Load K_label rows for this tile (one row per t).
    rpi_offs = bs_idx * stride_rt_bs + t_offs
    phys = tl.load(Req_to_tokens + rpi_offs, mask=in_history, other=0).to(tl.int64)

    s_offs = tl.arange(0, S)
    q_offs = bs_idx * stride_qbs + kv_idx * stride_qh + s_offs
    q_label = tl.load(Q_Label + q_offs).to(tl.float32)  # [S]

    kl_offs = phys[:, None] * stride_klt + kv_idx * stride_klh + s_offs[None, :]
    kl = tl.load(
        K_Label_Buffer + kl_offs,
        mask=in_history[:, None],
        other=0.0,
    ).to(
        tl.float32
    )  # [BLOCK_T, S]

    scores = tl.sum(kl * q_label[None, :], axis=1) * sm_scale
    scores = tl.where(valid, scores, -float("inf"))  # mask sink/recent/oob → -inf

    out_offs = bs_idx * stride_ab + kv_idx * stride_ah + t_offs
    tl.store(Att_Out + out_offs, scores)


def _launch_score(
    *,
    q_label: torch.Tensor,  # [bs, H_kv, S]
    k_label_layer: torch.Tensor,  # [T_pool, H_kv, S]
    req_to_token_indexed: torch.Tensor,  # [bs, max_ctx]
    seq_lens: torch.Tensor,  # [bs] int64
    att_out: torch.Tensor,  # [H_kv, bs, max_ctx] fp32
    sm_scale: float,
    sink_tokens: int,
    recent_tokens: int,
    block_t: int = 128,
) -> None:
    bs, h_kv, s = q_label.shape
    max_ctx = req_to_token_indexed.shape[1]
    num_blocks = triton.cdiv(max_ctx, block_t)
    grid = (bs, h_kv, num_blocks)
    # att_out layout: [bs, H_kv, max_ctx]. Kernel takes (stride_ab, stride_ah).
    _ds_native_score_kernel[grid](
        q_label,
        k_label_layer,
        req_to_token_indexed,
        seq_lens,
        att_out,
        q_label.stride(0),
        q_label.stride(1),
        k_label_layer.stride(0),
        k_label_layer.stride(1),
        req_to_token_indexed.stride(0),
        att_out.stride(0),
        att_out.stride(1),
        sm_scale,
        sink_tokens,
        recent_tokens,
        max_ctx,
        S=s,
        BLOCK_T=block_t,
        num_warps=4,
        num_stages=2,
    )


# --------------------------------------------------------------------------- #
# Build selected_physical: top-k physical + sink range + recent range.         #
# --------------------------------------------------------------------------- #


@triton.jit
def _ds_native_build_selected_physical_kernel(
    Topk_Logical,  # [bs, H_kv, TOP_K]                       int32
    Req_to_tokens,  # [bs, max_ctx]                           int32
    B_Seqlen,  # [bs]                                    int64
    Out,  # [bs, H_kv, TOTAL]                       int32
    stride_tl_b,
    stride_tl_h,
    stride_rt_b,
    stride_o_b,
    stride_o_h,
    sink_tokens,  # runtime int
    recent_tokens,  # runtime int
    max_ctx,  # runtime int
    TOP_K: tl.constexpr,
    TOP_K_PADDED: tl.constexpr,  # next-pow2(TOP_K)
    TOTAL: tl.constexpr,
    SINK_RECENT_SLACK: tl.constexpr,  # = TOTAL - TOP_K
    SINK_RECENT_PADDED: tl.constexpr,  # next-pow2(SINK_RECENT_SLACK)
):
    """One program per (bs, kv_head). Writes the [TOTAL] row.

    Layout (matches torch reference):
      [0     : TOP_K)               = req_to_token[b, topk_logical[b, h, :TOP_K]]
      [TOP_K : TOP_K+sink)          = req_to_token[b, 0..sink_tokens)
      [TOP_K+sink : TOTAL)          = req_to_token[b, seq-recent..seq)
    """
    bs_idx = tl.program_id(0)
    kv_idx = tl.program_id(1)

    out_base = bs_idx * stride_o_b + kv_idx * stride_o_h
    rt_base = bs_idx * stride_rt_b
    tl_base = bs_idx * stride_tl_b + kv_idx * stride_tl_h

    # 1. top-k physical: indirect gather req_to_token[b, topk_logical[b, h, k]]
    k_offs = tl.arange(0, TOP_K_PADDED)
    k_valid = k_offs < TOP_K
    logical = tl.load(Topk_Logical + tl_base + k_offs, mask=k_valid, other=0).to(
        tl.int64
    )
    logical = tl.maximum(
        logical, 0
    )  # defensive: clamp -1 sentinels (none here under v0)
    phys = tl.load(Req_to_tokens + rt_base + logical, mask=k_valid, other=0).to(
        tl.int32
    )
    tl.store(Out + out_base + k_offs, phys, mask=k_valid)

    # 2. sink + recent physical: req_to_token[b, [0..sink)] ∪ [seq-recent..seq).
    # Iterate over SINK_RECENT_PADDED (next-pow2 of SINK_RECENT_SLACK = TOTAL-TOP_K)
    # and mask out the tail; sink/recent counts are runtime.
    sr_offs = tl.arange(0, SINK_RECENT_PADDED)
    sr_in_range = sr_offs < SINK_RECENT_SLACK
    is_sink = sr_in_range & (sr_offs < sink_tokens)
    is_recent = (
        sr_in_range & (sr_offs >= sink_tokens) & (sr_offs < sink_tokens + recent_tokens)
    )

    # Sink logical position = sr_offs (when is_sink).
    sink_pos = sr_offs
    sink_phys = tl.load(
        Req_to_tokens + rt_base + sink_pos,
        mask=is_sink,
        other=0,
    ).to(tl.int32)

    seq_len = tl.load(B_Seqlen + bs_idx).to(tl.int64)
    rec_lo = tl.maximum(seq_len - recent_tokens, 0)
    rec_idx = sr_offs - sink_tokens  # 0..recent (only valid when is_recent)
    rec_pos = rec_lo + rec_idx
    rec_phys = tl.load(
        Req_to_tokens + rt_base + rec_pos,
        mask=is_recent,
        other=0,
    ).to(tl.int32)

    combined = tl.where(is_sink, sink_phys, rec_phys)
    tl.store(Out + out_base + TOP_K + sr_offs, combined, mask=sr_in_range)


def _build_selected_physical(
    *,
    topk_logical: torch.Tensor,  # [bs, H_kv, top_k] int32 (from torch.topk)
    req_to_token_indexed: torch.Tensor,  # [bs, max_ctx] int32
    seq_lens: torch.Tensor,  # [bs] int64
    sink_tokens: int,
    recent_tokens: int,
    out: torch.Tensor,  # [bs, H_kv, total_selected] int32 — preallocated
) -> None:
    """Concatenate top-k physical + sink physical + recent physical, in-place.

    Single Triton kernel — torch-op equivalent (kept in
    `_build_selected_physical_torch_ref` for parity testing) costs ~110
    µs per call at bs=1/h_kv=1 due to Python kernel-launch overhead from
    the ~20 small torch ops it expands into. The fused kernel runs the
    same work in one program per (bs, h_kv).

    Layout in `out[:, :, :]`:
        [0                : top_k)                  = top-k history physical
        [top_k            : top_k + sink_tokens)    = sink physical (positions 0..sink)
        [top_k + sink     : top_k + sink + recent)  = recent physical (last `recent` positions)

    Sink/recent positions are always-keep; the score kernel masked them out
    of the topk candidates, so there is no double-counting. Recent always
    covers the current decode position because `recent_tokens >= 1`.
    """
    if not topk_logical.is_cuda:
        _build_selected_physical_torch_ref(
            topk_logical=topk_logical,
            req_to_token_indexed=req_to_token_indexed,
            seq_lens=seq_lens,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            out=out,
        )
        return

    bs, h_kv, top_k = topk_logical.shape
    total = out.shape[2]
    max_ctx = req_to_token_indexed.shape[1]
    slack = total - top_k
    if slack < sink_tokens + recent_tokens:
        raise ValueError(
            f"out has slack {slack} < sink+recent = {sink_tokens + recent_tokens}"
        )
    grid = (bs, h_kv)
    _ds_native_build_selected_physical_kernel[grid](
        topk_logical,
        req_to_token_indexed,
        seq_lens,
        out,
        topk_logical.stride(0),
        topk_logical.stride(1),
        req_to_token_indexed.stride(0),
        out.stride(0),
        out.stride(1),
        sink_tokens,
        recent_tokens,
        max_ctx,
        TOP_K=top_k,
        TOP_K_PADDED=triton.next_power_of_2(top_k),
        TOTAL=total,
        SINK_RECENT_SLACK=slack,
        SINK_RECENT_PADDED=triton.next_power_of_2(slack),
        num_warps=1,
        num_stages=1,
    )


def _build_selected_physical_torch_ref(
    *,
    topk_logical: torch.Tensor,
    req_to_token_indexed: torch.Tensor,
    seq_lens: torch.Tensor,
    sink_tokens: int,
    recent_tokens: int,
    out: torch.Tensor,
) -> None:
    """Torch reference for the build kernel — used by tests + CPU paths.

    Kept verbatim from the v0 implementation so the parity test catches
    drift between the kernel and the reference.
    """
    bs, h_kv, top_k = topk_logical.shape
    device = out.device
    max_ctx = req_to_token_indexed.shape[1]

    # 1. top-k physical: gather req_to_token[b, topk_logical[b, h, k]]
    r2t_expand = req_to_token_indexed.unsqueeze(1).expand(bs, h_kv, max_ctx)
    topk_phys = torch.gather(r2t_expand, 2, topk_logical.long().clamp_min(0)).to(
        torch.int32
    )
    out[:, :, :top_k].copy_(topk_phys)

    if sink_tokens > 0:
        sink_phys = req_to_token_indexed[:, :sink_tokens].to(torch.int32)
        out[:, :, top_k : top_k + sink_tokens].copy_(
            sink_phys.unsqueeze(1).expand(bs, h_kv, sink_tokens)
        )

    if recent_tokens > 0:
        rec_offset = torch.arange(recent_tokens, device=device, dtype=torch.int64)
        rec_pos = (
            seq_lens.to(device).to(torch.int64).unsqueeze(1)
            - recent_tokens
            + rec_offset
        )
        rec_pos.clamp_(0, max_ctx - 1)
        rec_phys = torch.gather(req_to_token_indexed, 1, rec_pos.to(torch.int64)).to(
            torch.int32
        )
        out[:, :, top_k + sink_tokens : top_k + sink_tokens + recent_tokens].copy_(
            rec_phys.unsqueeze(1).expand(bs, h_kv, recent_tokens)
        )


# --------------------------------------------------------------------------- #
# Stage 2: sparse attention split-K (per (bs, q_head, block)).                 #
# --------------------------------------------------------------------------- #


@triton.jit
def _ds_native_sparse_attn_stage2_kernel(
    Q,  # [bs, H_q, D]
    K_Buffer,  # [T_pool, H_kv, D]
    V_Buffer,  # [T_pool, H_kv, D]
    Selected_Physical,  # [bs, H_kv, total_selected]               int32
    Mid_O,  # [bs, H_q, num_blocks, D]                 fp32
    Mid_O_LogExpSum,  # [bs, H_q, num_blocks]                    fp32
    sm_scale,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_sel_bs,
    stride_sel_h,
    stride_mob,
    stride_moh,
    stride_mos,
    stride_meb,
    stride_meh,
    GQA_GROUP: tl.constexpr,  # H_q // H_kv
    TOTAL_SELECTED: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per (bs, q_head, block_seq) program. Online softmax over one tile of
    the selected set, partial results accumulated into Mid_O / Mid_O_LogExpSum
    for stage-3 reduction.
    """
    bs_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    seq_blk = tl.program_id(2)
    kv_head = q_head // GQA_GROUP

    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_off = bs_idx * stride_qbs + q_head * stride_qh + offs_d
    q = tl.load(Q + q_off).to(tl.float32)  # [D]

    blk_start = seq_blk * BLOCK_SEQ
    blk_end = tl.minimum(blk_start + BLOCK_SEQ, TOTAL_SELECTED)

    sel_base = bs_idx * stride_sel_bs + kv_head * stride_sel_h

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for n_start in range(blk_start, blk_end, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        n_valid = n_offs < blk_end
        phys = tl.load(
            Selected_Physical + sel_base + n_offs,
            mask=n_valid,
            other=0,
        ).to(tl.int64)

        kv_offs = phys[:, None] * stride_kbs + kv_head * stride_kh + offs_d[None, :]
        k = tl.load(K_Buffer + kv_offs, mask=n_valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(V_Buffer + kv_offs, mask=n_valid[:, None], other=0.0).to(tl.float32)

        att = tl.sum(q[None, :] * k, axis=1) * sm_scale  # [BLOCK_N]
        att = tl.where(n_valid, att, -float("inf"))

        cur_max = tl.max(att, axis=0)
        new_max = tl.maximum(cur_max, max_logic)
        exp_att = tl.exp(att - new_max)
        scale = tl.exp(max_logic - new_max)
        acc = acc * scale + tl.sum(exp_att[:, None] * v, axis=0)
        sum_exp = sum_exp * scale + tl.sum(exp_att, axis=0)
        max_logic = new_max

    # Write partial result. Stage 3 reduces across blocks.
    safe_sum = tl.where(sum_exp > 0, sum_exp, 1.0)
    mid_off = bs_idx * stride_mob + q_head * stride_moh + seq_blk * stride_mos + offs_d
    log_off = bs_idx * stride_meb + q_head * stride_meh + seq_blk
    tl.store(Mid_O + mid_off, acc / safe_sum)
    tl.store(Mid_O_LogExpSum + log_off, max_logic + tl.log(safe_sum))


# --------------------------------------------------------------------------- #
# Stage 3: reduce partial results across the split-K blocks.                   #
# --------------------------------------------------------------------------- #


@triton.jit
def _ds_native_sparse_attn_stage3_kernel(
    Mid_O,  # [bs, H_q, num_blocks, D] fp32
    Mid_O_LogExpSum,  # [bs, H_q, num_blocks]    fp32
    O,  # [bs, H_q, D]
    stride_mob,
    stride_moh,
    stride_mos,
    stride_meb,
    stride_meh,
    stride_obs,
    stride_oh,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    bs_idx = tl.program_id(0)
    q_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for blk in range(NUM_BLOCKS):
        mid_off = bs_idx * stride_mob + q_head * stride_moh + blk * stride_mos + offs_d
        log_off = bs_idx * stride_meb + q_head * stride_meh + blk
        tv = tl.load(Mid_O + mid_off)
        tlog = tl.load(Mid_O_LogExpSum + log_off)
        new_max = tl.maximum(tlog, max_logic)
        old_scale = tl.exp(max_logic - new_max)
        exp_l = tl.exp(tlog - new_max)
        acc = acc * old_scale + exp_l * tv
        sum_exp = sum_exp * old_scale + exp_l
        max_logic = new_max

    safe_sum = tl.where(sum_exp > 0, sum_exp, 1.0)
    tl.store(O + bs_idx * stride_obs + q_head * stride_oh + offs_d, acc / safe_sum)


def _launch_sparse_attn(
    *,
    q: torch.Tensor,  # [bs, H_q, D]
    k_buffer: torch.Tensor,  # [T_pool, H_kv, D]
    v_buffer: torch.Tensor,  # [T_pool, H_kv, D]
    selected_physical: torch.Tensor,  # [bs, H_kv, total_selected] int32
    mid_out: torch.Tensor,  # [bs, H_q, num_blocks, D]   fp32
    mid_o_logexpsum: torch.Tensor,  # [bs, H_q, num_blocks]      fp32
    output: torch.Tensor,  # [bs, H_q, D]
    sm_scale: float,
    block_seq: int = 128,
    block_n: int = 16,
) -> None:
    bs, h_q, d = q.shape
    _, h_kv, _ = k_buffer.shape
    gqa_group = h_q // h_kv
    total_selected = selected_physical.shape[2]
    num_blocks = triton.cdiv(total_selected, block_seq)
    assert (
        mid_out.shape[2] >= num_blocks
    ), f"mid_out scratch has {mid_out.shape[2]} blocks, need {num_blocks}"

    grid2 = (bs, h_q, num_blocks)
    _ds_native_sparse_attn_stage2_kernel[grid2](
        q,
        k_buffer,
        v_buffer,
        selected_physical,
        mid_out,
        mid_o_logexpsum,
        sm_scale,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        selected_physical.stride(0),
        selected_physical.stride(1),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_o_logexpsum.stride(0),
        mid_o_logexpsum.stride(1),
        GQA_GROUP=gqa_group,
        TOTAL_SELECTED=total_selected,
        BLOCK_DMODEL=d,
        BLOCK_SEQ=block_seq,
        BLOCK_N=block_n,
        num_warps=1,
        num_stages=2,
    )

    grid3 = (bs, h_q)
    _ds_native_sparse_attn_stage3_kernel[grid3](
        mid_out,
        mid_o_logexpsum,
        output,
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_o_logexpsum.stride(0),
        mid_o_logexpsum.stride(1),
        output.stride(0),
        output.stride(1),
        NUM_BLOCKS=num_blocks,
        BLOCK_DMODEL=d,
        num_warps=4,
        num_stages=1,
    )


# --------------------------------------------------------------------------- #
# Public orchestrator.                                                         #
# --------------------------------------------------------------------------- #


def ds_native_sparse_decode(
    *,
    q: torch.Tensor,  # [bs, H_q, D]      query (current decode token)
    k_buffer: torch.Tensor,  # [T_pool, H_kv, D] KV pool's K (already includes new K)
    v_buffer: torch.Tensor,  # [T_pool, H_kv, D] KV pool's V (already includes new V)
    k_label_layer: torch.Tensor,  # [T_pool, H_kv, S] DS side cache (history only)
    q_label: torch.Tensor,  # [bs, H_kv, S]     pre-computed Q_label (GQA-reduced)
    req_to_token_indexed: torch.Tensor,  # [bs, max_ctx]     pre-indexed per request
    seq_lens: torch.Tensor,  # [bs] int64
    top_k: int,  # tokens scored & selected from history
    sink_tokens: int,
    recent_tokens: int,
    sm_scale: float,
    # preallocated scratch (none allocated inside — caller owns lifecycle)
    att_out_approx: torch.Tensor,  # [H_kv, bs, max_ctx]              fp32
    selected_physical: torch.Tensor,  # [bs, H_kv, total_selected]       int32
    mid_out: torch.Tensor,  # [bs, H_q, max_num_blocks, D]     fp32
    mid_o_logexpsum: torch.Tensor,  # [bs, H_q, max_num_blocks]        fp32
    output: torch.Tensor,  # [bs, H_q, D]
    score_block_t: int = 128,
    attn_block_seq: int = 128,
    attn_block_n: int = 16,
    topk_logical_scratch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full DS native sparse-decode forward.

    Returns `output` (same tensor that was passed in).

    Caller invariants:
      * `q_label` already gathered + GQA-reduced (use `_compute_q_label` from
        `select_kernels.py`).
      * `k_buffer` / `v_buffer` already include this step's new K/V for the
        current decode position (caller must call `set_kv_buffer` first).
      * `k_label_layer` covers history `[0, seq_len - 1)` for every active
        request; the current decode token's K_label is NOT required yet (and
        is excluded from scoring via `history_len = seq_len - 1`).
      * `total_selected = top_k + sink_tokens + recent_tokens` matches the
        third dim of `selected_physical`.
      * `seq_lens >= max(top_k + sink_tokens + recent_tokens, 1)` for every
        active row — caller gates this via `min_seq_len`.
    """
    bs, h_q, d = q.shape
    _, h_kv, _ = k_buffer.shape
    total_selected = selected_physical.shape[2]
    expected = top_k + sink_tokens + recent_tokens
    if total_selected != expected:
        raise ValueError(
            f"selected_physical[..., {total_selected}] != top_k + sink + recent = {expected}"
        )

    # 1. Score. att_out_approx layout is [bs, H_kv, max_ctx] so that
    # torch.topk on dim=-1 yields [bs, H_kv, top_k] directly — no
    # transpose-then-contiguous between selection and build.
    _launch_score(
        q_label=q_label,
        k_label_layer=k_label_layer,
        req_to_token_indexed=req_to_token_indexed,
        seq_lens=seq_lens,
        att_out=att_out_approx,
        sm_scale=sm_scale,
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
        block_t=score_block_t,
    )

    # 2. Top-k. Sink/recent/oob positions were masked to -inf in step 1.
    # When `topk_logical_scratch` is provided, write indices into it in
    # place; otherwise let torch.topk allocate (test/CPU paths).
    topk_logical = torch.topk(att_out_approx, top_k, dim=-1, sorted=False).indices
    if topk_logical.dtype != torch.int32:
        # torch.topk returns int64 indices; the build kernel expects int32.
        # The conversion is a small alloc, ~5-10 µs at bs=1.
        topk_logical = topk_logical.to(torch.int32)

    # 3. Build selected_physical (top-k + sink + recent), in-place.
    _build_selected_physical(
        topk_logical=topk_logical,
        req_to_token_indexed=req_to_token_indexed,
        seq_lens=seq_lens,
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
        out=selected_physical,
    )

    # 4. Sparse attention
    _launch_sparse_attn(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        selected_physical=selected_physical,
        mid_out=mid_out,
        mid_o_logexpsum=mid_o_logexpsum,
        output=output,
        sm_scale=sm_scale,
        block_seq=attn_block_seq,
        block_n=attn_block_n,
    )
    return output


__all__ = [
    "ds_native_sparse_decode",
    "_build_selected_physical",
    "_launch_score",
    "_launch_sparse_attn",
]
