from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.minimax_sparse_ops.common.index import (
    topk_index_reduce,
)


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_N": lambda args: triton.next_power_of_2(args["block_size"]),
    }
)
@triton.jit
def _prefill_npu_score_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    sink_ptr,
    idx_o_ptr,
    score_ptr,
    req_to_token_ptr,
    query_seq_lens_ptr,
    query_req_indices_ptr,
    total_q: tl.constexpr,
    num_heads: tl.constexpr,
    gqa_group_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_seqblock: tl.constexpr,
    block_size: tl.constexpr,
    sm_scale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_sink_h,
    stride_sink_d,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_s_h,
    stride_s_q,
    stride_s_k,
    stride_r2t_b,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    HAS_VALUE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
):
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")

    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kh = pid_h // gqa_group_size

    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    dim_mask = off_d < head_dim

    seq_len = tl.load(query_seq_lens_ptr + pid_q).to(tl.int32)
    req_idx = tl.load(query_req_indices_ptr + pid_q).to(tl.int64)

    q = tl.load(
        q_ptr + pid_q * stride_q_n + pid_h * stride_q_h + off_d * stride_q_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    if HAS_SINK:
        sink = tl.load(
            sink_ptr + pid_h * stride_sink_h + off_d * stride_sink_d,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        qsink = tl.sum(q * sink, axis=0) * sm_scale
        m_i = qsink
        l_i = tl.full((), 1.0, dtype=tl.float32)
    else:
        m_i = tl.full((), float("-inf"), dtype=tl.float32)
        l_i = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.full((BLOCK_SIZE_D,), 0.0, dtype=tl.float32)

    for block_idx in tl.range(0, max_seqblock):
        valid_block = block_idx * block_size < seq_len
        pos = block_idx * block_size + off_n
        pos_mask = valid_block & (pos < seq_len)
        slots = tl.load(
            req_to_token_ptr + req_idx * stride_r2t_b + pos,
            mask=pos_mask,
            other=0,
        ).to(tl.int64)

        k = tl.load(
            k_cache_ptr
            + slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d,
            mask=dim_mask[:, None] & pos_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        qk = tl.sum(q[:, None] * k, axis=0) * sm_scale
        qk = tl.where(pos_mask, qk, -1.0e30)

        sub_max = tl.max(qk, axis=0)
        if SCORE_TYPE == "max":
            score = sub_max
        else:
            score = sub_max + tl.log(tl.sum(tl.exp(qk - sub_max), axis=0))
            score = tl.where(score != score, float("-inf"), score)

        tl.store(
            score_ptr
            + pid_h * stride_s_h
            + pid_q * stride_s_q
            + block_idx * stride_s_k,
            score,
            mask=valid_block,
        )

        if HAS_VALUE:
            v = tl.load(
                v_cache_ptr
                + slots[:, None] * stride_v_s
                + pid_kh * stride_v_h
                + off_d[None, :] * stride_v_d,
                mask=pos_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            m_new = tl.maximum(m_i, sub_max)
            p = tl.where(pos_mask, tl.exp(qk - m_new), 0.0)
            l_new = tl.sum(p, axis=0)
            acc_scale = tl.where(m_i > float("-inf"), tl.exp(m_i - m_new), 0.0)
            acc_new = acc * acc_scale + tl.sum(p[:, None] * v, axis=0)
            l_i_new = l_i * acc_scale + l_new

            acc = tl.where(valid_block, acc_new, acc)
            l_i = tl.where(valid_block, l_i_new, l_i)
            m_i = tl.where(valid_block, m_new, m_i)

    if HAS_VALUE:
        out = tl.where(l_i > 0.0, acc / l_i, acc)
        tl.store(
            idx_o_ptr + pid_q * stride_o_n + pid_h * stride_o_h + off_d * stride_o_d,
            out.to(idx_o_ptr.dtype.element_ty),
            mask=dim_mask,
        )


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
    }
)
@triton.jit
def _prefill_npu_topk_kernel(
    score_ptr,
    topk_idx_ptr,
    query_seq_lens_ptr,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    max_seqblock: tl.constexpr,
    stride_s_h,
    stride_s_q,
    stride_s_k,
    stride_t_h,
    stride_t_q,
    stride_t_k,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)

    seq_len = tl.load(query_seq_lens_ptr + pid_q).to(tl.int32)
    num_blocks = tl.cdiv(seq_len, block_size)

    off_t = tl.arange(0, BLOCK_SIZE_T)
    valid_topk_lane = off_t < topk
    top_scores = tl.where(
        valid_topk_lane,
        tl.full((BLOCK_SIZE_T,), -1.0e30, dtype=tl.float32),
        tl.full((BLOCK_SIZE_T,), 1.0e30, dtype=tl.float32),
    )
    top_indices = tl.full((BLOCK_SIZE_T,), -1, dtype=tl.int32)

    for block_idx in tl.range(0, max_seqblock):
        valid_block = block_idx < num_blocks
        score = tl.load(
            score_ptr
            + pid_h * stride_s_h
            + pid_q * stride_s_q
            + block_idx * stride_s_k,
            mask=valid_block,
            other=-1.0e30,
        ).to(tl.float32)
        score = tl.where(score != score, -1.0e30, score)

        min_score = tl.min(top_scores, axis=0)
        candidate_pos = tl.where(
            (top_scores == min_score) & valid_topk_lane,
            off_t,
            tl.full((BLOCK_SIZE_T,), BLOCK_SIZE_T, dtype=tl.int32),
        )
        min_pos = tl.min(candidate_pos, axis=0)
        do_replace = valid_block & (score > min_score)
        replace_mask = off_t == min_pos

        top_scores = tl.where(replace_mask & do_replace, score, top_scores)
        top_indices = tl.where(replace_mask & do_replace, block_idx, top_indices)

    tl.store(
        topk_idx_ptr
        + pid_h * stride_t_h
        + pid_q * stride_t_q
        + off_t * stride_t_k,
        top_indices.to(topk_idx_ptr.dtype.element_ty),
        mask=off_t < topk,
    )


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_N": lambda args: triton.next_power_of_2(args["block_size"]),
    }
)
@triton.jit
def _prefill_npu_sparse_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    sink_ptr,
    topk_idx_ptr,
    out_ptr,
    req_to_token_ptr,
    query_seq_lens_ptr,
    query_req_indices_ptr,
    gqa_group_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_topk: tl.constexpr,
    block_size: tl.constexpr,
    sm_scale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_sink_h,
    stride_sink_d,
    stride_t_h,
    stride_t_q,
    stride_t_k,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_r2t_b,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_h = pid_kh * gqa_group_size

    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_N)

    head_mask = off_h < gqa_group_size
    dim_mask = off_d < head_dim

    seq_len = tl.load(query_seq_lens_ptr + pid_q).to(tl.int32)
    req_idx = tl.load(query_req_indices_ptr + pid_q).to(tl.int64)

    q = tl.load(
        q_ptr
        + pid_q * stride_q_n
        + (pid_h + off_h[:, None]) * stride_q_h
        + off_d[None, :] * stride_q_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    if HAS_SINK:
        sink = tl.load(
            sink_ptr
            + (pid_h + off_h[:, None]) * stride_sink_h
            + off_d[None, :] * stride_sink_d,
            mask=head_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        qsink = tl.sum(q.to(tl.float32) * sink, axis=1) * sm_scale
        m_i = qsink
        l_i = tl.full((BLOCK_SIZE_H,), 1.0, dtype=tl.float32)
    else:
        m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_SIZE_H,), 0.0, dtype=tl.float32)

    acc = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0.0, dtype=tl.float32)

    for topk_pos in tl.range(0, max_topk):
        logical_block = tl.load(
            topk_idx_ptr
            + pid_kh * stride_t_h
            + pid_q * stride_t_q
            + topk_pos * stride_t_k
        ).to(tl.int32)
        valid_block = logical_block >= 0
        pos = logical_block * block_size + off_n
        pos_mask = valid_block & (pos < seq_len)
        slots = tl.load(
            req_to_token_ptr + req_idx * stride_r2t_b + pos,
            mask=pos_mask,
            other=0,
        ).to(tl.int64)

        k = tl.load(
            k_cache_ptr
            + slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d,
            mask=dim_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            v_cache_ptr
            + slots[:, None] * stride_v_s
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d,
            mask=pos_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        qk = tl.dot(q, k) * sm_scale
        qk = tl.where(pos_mask[None, :], qk, -1.0e30)
        block_m = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, block_m)
        p = tl.where(pos_mask[None, :], tl.exp(qk - m_new[:, None]), 0.0)
        l_new = tl.sum(p, axis=1)
        acc_scale = tl.where(m_i > float("-inf"), tl.exp(m_i - m_new), 0.0)
        acc_new = acc * acc_scale[:, None] + tl.dot(p.to(v.dtype), v)
        l_i_new = l_i * acc_scale + l_new

        acc = tl.where(valid_block, acc_new, acc)
        l_i = tl.where(valid_block, l_i_new, l_i)
        m_i = tl.where(valid_block, m_new, m_i)

    out = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], acc)
    tl.store(
        out_ptr
        + pid_q * stride_o_n
        + (pid_h + off_h[:, None]) * stride_o_h
        + off_d[None, :] * stride_o_d,
        out.to(out_ptr.dtype.element_ty),
        mask=head_mask[:, None] & dim_mask[None, :],
    )


def _build_prefill_query_meta(
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefix_lens: torch.Tensor,
    total_q: int,
):
    if total_q == 0:
        return cu_seqlens.new_empty((0,), dtype=torch.int32)

    extend_lens = torch.diff(cu_seqlens).to(torch.long)
    batch_size = extend_lens.shape[0]
    device = cu_seqlens.device
    batch_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=device, dtype=torch.long), extend_lens
    )
    q_starts = torch.repeat_interleave(cu_seqlens[:-1].to(torch.long), extend_lens)
    q_offsets = torch.arange(total_q, device=device, dtype=torch.long) - q_starts
    query_seq_lens = (
        prefix_lens.to(device=device, dtype=torch.long)[batch_ids] + q_offsets + 1
    ).to(torch.int32)
    query_req_indices = slot_ids.to(device=device, dtype=torch.long)[batch_ids]
    return query_seq_lens, query_req_indices


@torch.no_grad()
def flash_prefill_npu_with_topk_index(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: Optional[torch.Tensor],
    sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    query_seq_lens: torch.Tensor,
    query_req_indices: torch.Tensor,
    max_seqlen_k: int,
    block_size: int,
    topk: int,
    init_blocks: int = 0,
    local_blocks: int = 0,
    sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    assert score_type in ("max", "lse")
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache.dtype == q.dtype
    assert q.shape[0] == query_seq_lens.shape[0] == query_req_indices.shape[0]
    if init_blocks != 0 or local_blocks != 0:
        raise NotImplementedError(
            "NPU prefill topk index kernel emits pure top-k only; "
            "forced init/local blocks are appended by the wrapper."
        )

    total_q, num_heads, head_dim = q.shape
    _, num_kv_heads, cache_head_dim = k_cache.shape
    assert cache_head_dim == head_dim
    assert num_heads % num_kv_heads == 0
    if not disable_index_value:
        assert v_cache is not None
        assert v_cache.dtype == q.dtype
        assert v_cache.shape == k_cache.shape

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    max_seqblock = (max_seqlen_k + block_size - 1) // block_size
    score = torch.full(
        (num_heads, total_q, max_seqblock),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    idx_o = None
    if not disable_index_value:
        idx_o = torch.empty_like(q)

    grid = (total_q, num_heads)
    _prefill_npu_score_kernel[grid](
        q,
        k_cache,
        v_cache if v_cache is not None else k_cache,
        sink if sink is not None else q,
        idx_o if idx_o is not None else q,
        score,
        req_to_token,
        query_seq_lens,
        query_req_indices,
        total_q,
        num_heads,
        num_heads // num_kv_heads,
        head_dim,
        max_seqblock,
        block_size,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0) if v_cache is not None else 0,
        v_cache.stride(1) if v_cache is not None else 0,
        v_cache.stride(2) if v_cache is not None else 0,
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        idx_o.stride(0) if idx_o is not None else 0,
        idx_o.stride(1) if idx_o is not None else 0,
        idx_o.stride(2) if idx_o is not None else 0,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        req_to_token.stride(0),
        HAS_VALUE=not disable_index_value,
        HAS_SINK=sink is not None,
        SCORE_TYPE=score_type,
        num_warps=4,
        num_stages=2,
    )

    topk_idx = torch.empty(
        (num_heads, total_q, topk),
        dtype=torch.int32,
        device=q.device,
    )
    _prefill_npu_topk_kernel[grid](
        score,
        topk_idx,
        query_seq_lens,
        block_size,
        topk,
        max_seqblock,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        num_warps=1,
        num_stages=1,
    )
    return idx_o, topk_idx


@torch.no_grad()
def flash_prefill_npu_with_gqa_share_sparse(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    query_seq_lens: torch.Tensor,
    query_req_indices: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache.dtype == q.dtype
    assert v_cache.dtype == q.dtype
    assert k_cache.shape == v_cache.shape

    total_q, num_q_heads, head_dim = q.shape
    _, num_kv_heads, cache_head_dim = k_cache.shape
    assert cache_head_dim == head_dim
    assert num_q_heads % num_kv_heads == 0
    assert topk_idx.shape[0] == num_kv_heads
    assert topk_idx.shape[1] == total_q

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    out = torch.empty_like(q)
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_q_heads // num_kv_heads
    grid = (total_q, num_kv_heads)
    _prefill_npu_sparse_kernel[grid](
        q,
        k_cache,
        v_cache,
        sink if sink is not None else q,
        topk_idx,
        out,
        req_to_token,
        query_seq_lens,
        query_req_indices,
        gqa_group_size,
        head_dim,
        max_topk,
        block_size,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        req_to_token.stride(0),
        HAS_SINK=sink is not None,
        num_warps=4,
        num_stages=2,
    )
    return out


def _merge_prefill_sparse_blocks(
    topk_idx: torch.Tensor,
    query_seq_lens: torch.Tensor,
    block_size: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    total_forced = init_blocks + local_blocks
    if total_forced <= 0:
        return topk_idx

    topk_by_query = topk_idx.permute(1, 0, 2).contiguous()
    num_queries, num_heads, topk = topk_by_query.shape
    total = topk + total_forced
    device = topk_by_query.device
    dtype = topk_by_query.dtype

    query_positions = (query_seq_lens.to(torch.long) - 1).clamp(min=0)
    num_blocks = torch.div(
        query_seq_lens.to(torch.long) + block_size - 1,
        block_size,
        rounding_mode="floor",
    )
    qpos_3d = query_positions[:, None, None]
    num_blocks_3d = num_blocks[:, None, None]

    if init_blocks == 0 and local_blocks == 1:
        local = (query_positions // block_size).clamp(min=0)
        local = torch.minimum(local, (num_blocks - 1).clamp(min=0))
        local = local.to(dtype).view(num_queries, 1, 1).expand(-1, num_heads, -1)
        valid_topk = (topk_by_query >= 0) & (
            topk_by_query.to(torch.long) < num_blocks_3d
        )
        valid_topk = valid_topk & (
            topk_by_query.to(torch.long) * block_size <= qpos_3d
        )
        local_duplicate = ((topk_by_query == local) & valid_topk).any(
            dim=-1, keepdim=True
        )
        valid_local = (local >= 0) & (local.to(torch.long) < num_blocks_3d)
        valid_local = (
            valid_local
            & (local.to(torch.long) * block_size <= qpos_3d)
            & ~local_duplicate
        )
        merged = torch.cat(
            [
                torch.where(
                    valid_topk, topk_by_query, torch.full_like(topk_by_query, -1)
                ),
                torch.where(valid_local, local, torch.full_like(local, -1)),
            ],
            dim=-1,
        )
        return merged.permute(1, 0, 2).contiguous()

    forced_parts = []
    if init_blocks > 0:
        forced_parts.append(
            torch.arange(init_blocks, device=device, dtype=dtype)
            .view(1, 1, -1)
            .expand(num_queries, num_heads, -1)
        )
    if local_blocks > 0:
        offsets = torch.arange(local_blocks, device=device, dtype=torch.long)
        block_ids = query_positions // block_size
        first = (block_ids - local_blocks + 1).clamp(min=0)
        forced_parts.append(
            (first[:, None] + offsets[None, :])
            .to(dtype)
            .view(num_queries, 1, -1)
            .expand(-1, num_heads, -1)
        )

    forced = torch.cat(forced_parts, dim=-1)
    candidates = torch.cat([forced, topk_by_query], dim=-1)
    valid = (candidates >= 0) & (candidates.to(torch.long) < num_blocks_3d)
    valid = valid & (candidates.to(torch.long) * block_size <= qpos_3d)
    invalid_value = num_blocks_3d.expand_as(candidates).to(dtype)

    sorted_candidates = torch.sort(
        torch.where(valid, candidates, invalid_value), dim=-1
    ).values
    sorted_valid = sorted_candidates.to(torch.long) < num_blocks_3d
    previous = torch.cat(
        [
            torch.full_like(sorted_candidates[..., :1], -1),
            sorted_candidates[..., :-1],
        ],
        dim=-1,
    )
    keep = sorted_valid & (sorted_candidates != previous)
    ranks = torch.cumsum(keep.to(torch.int32), dim=-1) - 1
    output = torch.full(
        (num_queries, num_heads, total + 1),
        -1,
        dtype=dtype,
        device=device,
    )
    overflow_rank = torch.full_like(ranks, total)
    scatter_index = torch.where(
        keep & (ranks < total), ranks, overflow_rank
    ).long()
    scatter_src = torch.where(keep, sorted_candidates, -1)
    output.scatter_(2, scatter_index, scatter_src)
    return output[:, :, :total].permute(1, 0, 2).contiguous()


@torch.no_grad()
def minimax_sparse_prefill_npu_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: Optional[torch.Tensor],
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v_cache: Optional[torch.Tensor],
    idx_sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    all_seqblock_q: Optional[int] = None,
    seqlens_cpu: Optional[List[int]] = None,
    page_size: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Run NPU prefill with prefill-native Triton kernels.

    This path keeps prefill on the slot/req_to_token kernels instead of
    representing every prefill token as a decode query. It still selects pure
    top-k blocks first and appends forced init/local blocks afterwards to match
    the NPU PyTorch fallback semantics used by the debug comparator.
    """
    del seq_lens, max_seqlen_q, cu_seqblocks_q, max_seqblock_q
    del all_seqblock_q, seqlens_cpu, page_size

    if block_size_q != 1:
        raise NotImplementedError(
            "MiniMax-M3 NPU Triton native prefill currently requires "
            f"block_size_q=1 (got block_size_q={block_size_q})."
        )
    if q.shape[0] == 0:
        idx_o = None if disable_index_value else idx_q.new_empty(idx_q.shape)
        return idx_o, q.new_empty(q.shape)

    query_seq_lens, query_req_indices = _build_prefill_query_meta(
        req_to_token,
        slot_ids,
        cu_seqlens,
        prefix_lens,
        q.shape[0],
    )

    idx_o, topk_idx = flash_prefill_npu_with_topk_index(
        q=idx_q,
        k_cache=idx_k_cache,
        v_cache=idx_v_cache,
        sink=idx_sink,
        req_to_token=req_to_token,
        query_seq_lens=query_seq_lens,
        query_req_indices=query_req_indices,
        max_seqlen_k=max_seqlen_k,
        block_size=block_size_k,
        topk=topk,
        init_blocks=0,
        local_blocks=0,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
    )

    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]

    topk_idx = _merge_prefill_sparse_blocks(
        topk_idx,
        query_seq_lens,
        block_size_k,
        init_blocks,
        local_blocks,
    )

    if num_idx_heads > num_kv_heads:
        idx_group_size = num_idx_heads // num_kv_heads
        topk_idx = topk_index_reduce(
            topk_idx.view(num_kv_heads, idx_group_size, -1, topk_idx.shape[-1]),
            dim=1,
        )

    o = flash_prefill_npu_with_gqa_share_sparse(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        sink=sink,
        req_to_token=req_to_token,
        query_seq_lens=query_seq_lens,
        query_req_indices=query_req_indices,
        topk_idx=topk_idx,
        block_size=block_size_k,
        sm_scale=sm_scale,
    )
    return idx_o, o
