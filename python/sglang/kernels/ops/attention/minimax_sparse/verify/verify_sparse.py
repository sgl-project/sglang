# Copyright 2025 XunhaoLai. All rights reserved.
#
# EAGLE3 verify 专用 sparse attention 入口 (Path 1 Strategy B, 固定长度版).
#
# 复制 minimax_sparse.minimax_sparse_prefill 的结构, 改造点:
#   1. Step1 用 flash_verify_prefill_with_topk_index (score buffer 第3维固定上界)
#   2. Step3 用 flash_verify_prefill_with_gqa_share_sparse (OOB 双保险)
#   3. 新增 max_seqblock_k_upper: int 参数, = cdiv(context_len, block_size_k),
#      由 backend 计算并传入 (静态 Python int, graph-safe)
#   4. Step2 (topk_index_reduce) 与 prefill 完全一致, 不变
# causal 语义 (off_q >= pos_k in Step1, off_q_k >= c in Step3) 与 prefill 完全一致.

from typing import List, Optional

import torch

from ..common.index import topk_index_reduce
from .flash_with_topk_idx import flash_verify_prefill_with_topk_index
from .topk_sparse import flash_verify_prefill_with_gqa_share_sparse


def minimax_sparse_verify_prefill(
    q: torch.Tensor,  # [total_extend_tokens, num_q_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged main)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged main)
    sink: Optional[torch.Tensor],  # [num_q_heads, qk_head_dim]
    idx_q: torch.Tensor,  # [total_extend_tokens, num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim] (paged index)
    idx_v_cache: Optional[torch.Tensor],  # [max_slots, 1, idx_head_dim] (paged index); None when disable_index_value
    idx_sink: Optional[torch.Tensor],  # [num_idx_heads, idx_head_dim]
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch_size, ]
    cu_seqlens: torch.Tensor,  # [batch_size + 1, ] (Q-side cumulative)
    seq_lens: torch.Tensor,  # [batch_size, ] total K length (prefix + chunk)
    prefix_lens: torch.Tensor,  # [batch_size, ]
    max_seqlen_q: int,
    max_seqlen_k: int,
    max_seqblock_k_upper: int,  # FIXED upper bound on KV blocks (graph-safe)
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    seqlens_cpu: Optional[List[int]] = None,
):
    """EAGLE3 verify 专用 sparse attention (固定长度 score buffer 版).

    与 minimax_sparse_prefill 的唯一区别: Step1 的 score 张量第3维用固定上界
    max_seqblock_k_upper (= cdiv(context_len, block_size_k)) 而非
    cdiv(max_seqlen_k, block_size_k) (运行时动态), Step3 kernel 额外接受
    max_seqblock_k_upper 做 OOB 双保险. 其余参数与 minimax_sparse_prefill 一致.
    """
    # Step 1: Flash attention with topk index (using index head) — fixed-size score
    idx_o, topk_idx = flash_verify_prefill_with_topk_index(
        q=idx_q,
        k_cache=idx_k_cache,
        v_cache=idx_v_cache,
        sink=idx_sink,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        max_seqblock_k_upper=max_seqblock_k_upper,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
        seqlens_cpu=seqlens_cpu,
    )
    # Step 2: Reduce topk idx if num_idx_heads > num_kv_heads (unchanged)
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    if idx_group_size > 1:
        topk_idx = topk_index_reduce(
            topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
        )
    # Step 3: Sparse attention using topk index (main head) — OOB double-safety
    o = flash_verify_prefill_with_gqa_share_sparse(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        sink=sink,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        topk_idx=topk_idx,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqblock_k_upper=max_seqblock_k_upper,
        sm_scale=sm_scale,
        seqlens_cpu=seqlens_cpu,
    )
    return idx_o, o
