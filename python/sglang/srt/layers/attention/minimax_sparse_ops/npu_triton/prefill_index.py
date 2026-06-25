from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


_PREFILL_NPU_SCORE_BLOCK_SIZE_N = 64


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_N": lambda args: _PREFILL_NPU_SCORE_BLOCK_SIZE_N,
    }
)
@triton.jit
def _prefill_npu_msa_index_score_kernel(
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
    tl.static_assert(block_size % BLOCK_SIZE_N == 0)

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
        score_max = tl.full((), -1.0e30, dtype=tl.float32)
        score_lse_m = tl.full((), -1.0e30, dtype=tl.float32)
        score_lse_l = tl.full((), 0.0, dtype=tl.float32)

        for inner_start in tl.static_range(0, block_size, BLOCK_SIZE_N):
            pos = block_idx * block_size + inner_start + off_n
            pos_mask = valid_block & (pos < seq_len)
            valid_chunk = tl.sum(pos_mask.to(tl.int32), axis=0) > 0
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
                score_max = tl.where(
                    valid_chunk, tl.maximum(score_max, sub_max), score_max
                )
            else:
                m_new = tl.maximum(score_lse_m, sub_max)
                l_new = tl.sum(tl.exp(qk - m_new), axis=0)
                old_scale = tl.where(
                    score_lse_m > float("-inf"),
                    tl.exp(score_lse_m - m_new),
                    0.0,
                )
                score_lse_l_new = score_lse_l * old_scale + l_new
                score_lse_m = tl.where(valid_chunk, m_new, score_lse_m)
                score_lse_l = tl.where(valid_chunk, score_lse_l_new, score_lse_l)

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

                acc = tl.where(valid_block & valid_chunk, acc_new, acc)
                l_i = tl.where(valid_block & valid_chunk, l_i_new, l_i)
                m_i = tl.where(valid_block & valid_chunk, m_new, m_i)

        if SCORE_TYPE == "max":
            score = score_max
        else:
            score = score_lse_m + tl.log(score_lse_l)
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
def _prefill_npu_msa_index_topk_kernel(
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


@torch.no_grad()
def flash_prefill_npu_msa_index(
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
            "NPU MSA prefill index kernel emits pure top-k only; "
            "forced init/local blocks are appended by the prefill wrapper."
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
    _prefill_npu_msa_index_score_kernel[grid](
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
    _prefill_npu_msa_index_topk_kernel[grid](
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
