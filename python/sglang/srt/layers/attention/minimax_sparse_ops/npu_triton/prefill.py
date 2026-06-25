from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.attention.minimax_sparse_ops.common.index import (
    topk_index_reduce,
)
from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.flash_block_score_decode import (
    flash_decode_bnsd_with_topk_idx,
)
from sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.topk_sparse_decode import (
    flash_decode_bnsd_with_gqa_share_sparse,
)


def _cache_as_bnsd(cache: Optional[torch.Tensor], page_size: int):
    if cache is None:
        return None
    if cache.dim() == 4:
        return cache
    num_pages = cache.shape[0] // page_size
    return cache.view(num_pages, page_size, cache.shape[1], cache.shape[2])


def _build_prefill_query_meta(
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_k: int,
    page_size: int,
):
    extend_lens = torch.diff(cu_seqlens).to(torch.long)
    batch_size = extend_lens.shape[0]
    max_blocks = (max_seqlen_k + page_size - 1) // page_size
    total_q = int(cu_seqlens[-1].item())
    if total_q == 0:
        return (
            cu_seqlens.new_empty((0,), dtype=torch.int32),
            cu_seqlens.new_empty((0, max_blocks), dtype=torch.int32),
        )

    device = cu_seqlens.device
    batch_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=device, dtype=torch.long), extend_lens
    )
    q_starts = torch.repeat_interleave(cu_seqlens[:-1].to(torch.long), extend_lens)
    q_offsets = torch.arange(total_q, device=device, dtype=torch.long) - q_starts
    query_seq_lens = (
        prefix_lens.to(device=device, dtype=torch.long)[batch_ids] + q_offsets + 1
    ).to(torch.int32)

    req_idx = slot_ids.to(device=req_to_token.device, dtype=torch.long)[
        batch_ids.to(req_to_token.device)
    ]
    blk_cols = (
        torch.arange(max_blocks, device=req_to_token.device, dtype=torch.long)
        * page_size
    )
    blk_cols = blk_cols.clamp(max=req_to_token.shape[1] - 1)
    token_slots = req_to_token[req_idx][:, blk_cols]
    block_table = (token_slots // page_size).to(torch.int32)
    return query_seq_lens, block_table


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
    """Run NPU prefill with Ascend-friendly BNSD kernels.

    Each prefill token is represented as a decode query with its own effective
    KV length. This avoids the generic CUDA prefill kernels and keeps the forced
    init/local block semantics aligned with the NPU decode path.
    """
    del max_seqlen_q, block_size_q, cu_seqblocks_q, max_seqblock_q
    del all_seqblock_q, seqlens_cpu, seq_lens

    page_size = block_size_k if page_size is None else page_size
    if page_size != block_size_k:
        raise NotImplementedError(
            "MiniMax-M3 NPU Triton prefill requires page_size == block_size_k "
            f"(got page_size={page_size}, block_size_k={block_size_k})."
        )
    if sink is not None or idx_sink is not None:
        raise NotImplementedError("MiniMax-M3 NPU Triton prefill does not support sink")
    if q.shape[0] == 0:
        idx_o = None if disable_index_value else idx_q.new_empty(idx_q.shape)
        return idx_o, q.new_empty(q.shape)

    k_bnsd = _cache_as_bnsd(k_cache, page_size)
    v_bnsd = _cache_as_bnsd(v_cache, page_size)
    idx_k_bnsd = _cache_as_bnsd(idx_k_cache, page_size)
    idx_v_bnsd = None if idx_v_cache is None else _cache_as_bnsd(idx_v_cache, page_size)

    query_seq_lens, block_table = _build_prefill_query_meta(
        req_to_token,
        slot_ids,
        cu_seqlens,
        prefix_lens,
        max_seqlen_k,
        page_size,
    )

    idx_dim = idx_q.shape[-1]
    idx_o, topk_idx = flash_decode_bnsd_with_topk_idx(
        q=idx_q,
        sink=None,
        k_cache_bnsd=idx_k_bnsd,
        v_cache_bnsd=idx_v_bnsd,
        block_table=block_table,
        seq_lens=query_seq_lens,
        max_seqlen=max_seqlen_k,
        block_size=page_size,
        topk=topk,
        init_blocks=0,
        local_blocks=0,
        sm_scale=idx_sm_scale if idx_sm_scale is not None else idx_dim**-0.5,
        score_type=score_type,
        disable_index_value=disable_index_value,
    )

    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_bnsd.shape[2]
    if num_idx_heads > num_kv_heads:
        idx_group_size = num_idx_heads // num_kv_heads
        topk_idx = topk_index_reduce(
            topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
        )

    topk_idx = _merge_prefill_sparse_blocks(
        topk_idx,
        query_seq_lens,
        page_size,
        init_blocks,
        local_blocks,
    )

    head_dim = q.shape[-1]
    o = flash_decode_bnsd_with_gqa_share_sparse(
        q=q,
        sink=None,
        k_cache_bnsd=k_bnsd,
        v_cache_bnsd=v_bnsd,
        block_table=block_table,
        seq_lens=query_seq_lens,
        block_size=page_size,
        topk_idx=topk_idx,
        sm_scale=sm_scale if sm_scale is not None else head_dim**-0.5,
    )
    return idx_o, o
