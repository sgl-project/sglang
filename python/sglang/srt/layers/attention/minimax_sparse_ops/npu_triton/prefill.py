from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import triton

from sglang.srt.layers.attention.minimax_sparse_ops.common.index import (
    topk_index_reduce,
)
from sglang.srt.layers.attention.minimax_sparse_ops.common.utils import (
    get_cu_seqblocks,
    set_triton_allocator_if_available,
)
from sglang.srt.layers.attention.minimax_sparse_ops.prefill.flash_with_topk_idx import (
    _flash_attn_fwd_with_block_score_kernel,
    _topk_index_kernel,
)
from sglang.srt.layers.attention.minimax_sparse_ops.prefill.topk_sparse import (
    flash_prefill_with_gqa_share_sparse,
)


@torch.no_grad()
def _flash_prefill_with_topk_index_npu(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: Optional[torch.Tensor],
    sink: Optional[torch.Tensor],
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
    sm_scale: Optional[float],
    score_type: str,
    disable_index_value: bool,
    cu_seqblocks_q: torch.Tensor,
    max_seqblock_q: int,
    all_seqblock_q: int,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """NPU-friendly variant of the index prefill wrapper.

    The underlying Triton kernel is shared with the generic path. This wrapper
    keeps pointer arguments typed even when index-value output is disabled.
    """
    assert score_type in (
        "max",
        "lse",
    ), f"score_type must be 'max' or 'lse', got {score_type!r}"
    set_triton_allocator_if_available()
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert k_cache.dtype == q.dtype
    assert cu_seqlens.dtype == torch.int32

    total_q, num_heads, qk_head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    if disable_index_value:
        v_cache_arg = k_cache if v_cache is None else v_cache
        v_head_dim = qk_head_dim
    else:
        assert v_cache is not None and v_cache.dtype == q.dtype
        assert v_cache.shape[1] == k_cache.shape[1]
        v_cache_arg = v_cache
        v_head_dim = v_cache.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    batch_size = cu_seqlens.shape[0] - 1
    assert qk_head_dim <= 256 and v_head_dim <= 256, "head_dim must be less than 256"
    if sink is not None:
        assert sink.shape[0] == num_heads and sink.shape[1] == qk_head_dim
    assert (
        init_blocks + local_blocks <= topk
    ), "init_blocks + local_blocks must be less than topk"
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5

    max_seqblock_k = triton.cdiv(max_seqlen_k, block_size_k)
    o = torch.empty(total_q, num_heads, v_head_dim, dtype=q.dtype, device=q.device)
    score = torch.full(
        (num_heads, total_q, max_seqblock_k),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )

    def grid(META):
        return (triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]), batch_size * num_heads)

    _flash_attn_fwd_with_block_score_kernel[grid](
        q,
        k_cache,
        v_cache_arg,
        sink,
        o,
        score,
        req_to_token,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        slot_ids,
        max_slots,
        num_heads,
        gqa_group_size,
        qk_head_dim,
        v_head_dim,
        block_size_k,
        sm_scale,
        False,
        1,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache_arg.stride(0),
        v_cache_arg.stride(1),
        v_cache_arg.stride(2),
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        req_to_token.stride(0),
        SCORE_TYPE=score_type,
        DISABLE_INDEX_VALUE=disable_index_value,
    )

    topk_idx = torch.full(
        (num_heads, all_seqblock_q, topk),
        fill_value=-1,
        device=score.device,
        dtype=torch.int32,
    )
    grid = (max_seqblock_q, batch_size, num_heads)
    _topk_index_kernel[grid](
        score,
        topk_idx,
        block_size_q,
        block_size_k,
        cu_seqlens,
        cu_seqblocks_q,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        MASK_INIT=False,
        MASK_LOCAL=False,
    )
    return (None if disable_index_value else o), topk_idx


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
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Run MiniMax-M3 sparse prefill through the NPU Triton path.

    This mirrors ``minimax_sparse_prefill`` while keeping the NPU path explicit:
    TMA is disabled and query-block metadata can be supplied by the backend once
    per forward instead of recomputed by every sparse layer.
    """
    if cu_seqblocks_q is None or max_seqblock_q is None or all_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k, seqlens_cpu
        )

    idx_o, topk_idx = _flash_prefill_with_topk_index_npu(
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
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        disable_index_value=disable_index_value,
        cu_seqblocks_q=cu_seqblocks_q,
        max_seqblock_q=max_seqblock_q,
        all_seqblock_q=all_seqblock_q,
    )

    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    if idx_group_size > 1:
        topk_idx = topk_index_reduce(
            topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
        )

    o = flash_prefill_with_gqa_share_sparse(
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
        sm_scale=sm_scale,
        use_tma=False,
        cu_seqblocks_q=cu_seqblocks_q,
        max_seqblock_q=max_seqblock_q,
    )
    return idx_o, o
