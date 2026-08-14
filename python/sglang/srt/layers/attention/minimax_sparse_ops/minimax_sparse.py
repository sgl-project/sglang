# Copyright 2025 XunhaoLai. All rights reserved.

import logging
from typing import Callable, List, Optional, Tuple

import torch

from sglang.kernels.ops.attention.minimax_sparse.common.index import topk_index_reduce
from sglang.kernels.ops.attention.minimax_sparse.common.utils import get_cu_seqblocks
from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (
    flash_prefill_with_topk_index,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (
    flash_prefill_with_gqa_share_sparse,
)

logger = logging.getLogger(__name__)
_msa_fallback_warned = False


def _warn_msa_fallback(err: Exception) -> None:
    global _msa_fallback_warned
    if _msa_fallback_warned:
        return
    logger.warning(
        "MiniMax MSA backend is unavailable (%s); falling back to Triton sparse attention.",
        err,
    )
    _msa_fallback_warned = True


def minimax_sparse_prefill(
    q: torch.Tensor,  # [total_extend_tokens, num_q_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged main)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged main)
    sink: Optional[torch.Tensor],  # [num_q_heads, qk_head_dim]
    idx_q: torch.Tensor,  # [total_extend_tokens, num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim] (paged index)
    idx_v_cache: Optional[
        torch.Tensor
    ],  # [max_slots, 1, idx_head_dim] (paged index); None when disable_index_value
    idx_sink: Optional[torch.Tensor],  # [num_idx_heads, idx_head_dim]
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch_size, ]
    cu_seqlens: torch.Tensor,  # [batch_size + 1, ] (Q-side cumulative)
    seq_lens: torch.Tensor,  # [batch_size, ] total K length (prefix + chunk)
    prefix_lens: torch.Tensor,  # [batch_size, ]
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
    use_msa: bool = False,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    all_seqblock_q: Optional[int] = None,
    seqlens_cpu: Optional[List[int]] = None,
):
    """Run MiniMax-M3 sparse prefill.

    ``cu_seqblocks_q``, ``max_seqblock_q``, and ``all_seqblock_q`` are optional
    precomputed query-block metadata shared by the index and value sparse
    kernels. Supplying them avoids recomputing the same block layout twice.
    ``seqlens_cpu`` (host copy of ``torch.diff(cu_seqlens)``) is forwarded to
    ``get_cu_seqblocks`` to avoid a per-layer device sync when it recomputes.
    """
    if cu_seqblocks_q is None or max_seqblock_q is None or all_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k, seqlens_cpu
        )

    # All seqlen is less than topk, use full attention
    # Step 1: Flash attention with topk index (using index head)
    idx_o, topk_idx = flash_prefill_with_topk_index(
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
    # Step 2: Reduce topk idx if num_idx_heads > num_kv_heads
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    if idx_group_size > 1:
        topk_idx = topk_index_reduce(
            topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
        )
    # Step 3: Sparse attention using topk index (main head). The MSA path only
    # replaces this step; the indexer above is unchanged. MSA has no attn-sink
    # input, so keep the Triton path when sink is present.
    if use_msa and sink is None:
        from .msa import MSAUnavailableError, msa_sparse_prefill_main

        try:
            o = msa_sparse_prefill_main(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                topk_idx=topk_idx,
                req_to_token=req_to_token,
                slot_ids=slot_ids,
                cu_seqlens=cu_seqlens,
                seq_lens=seq_lens,
                prefix_lens=prefix_lens,
                block_size_k=block_size_k,
                sm_scale=sm_scale,
            )
        except MSAUnavailableError as err:
            _warn_msa_fallback(err)
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
                cu_seqblocks_q=cu_seqblocks_q,
                max_seqblock_q=max_seqblock_q,
            )
    else:
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
            cu_seqblocks_q=cu_seqblocks_q,
            max_seqblock_q=max_seqblock_q,
        )
    return idx_o, o


def minimax_sparse_decode(
    q: torch.Tensor,  # [batch_size, num_q_heads, qk_head_dim]
    sink: Optional[torch.Tensor],  # [num_q_heads, qk_head_dim]
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    idx_q: torch.Tensor,  # [batch_size, num_idx_heads, idx_head_dim], num_idx_heads >= num_kv_heads
    idx_sink: Optional[torch.Tensor],  # [num_idx_heads, idx_head_dim]
    idx_k_cache: torch.Tensor,  # [max_slots, 1, idx_head_dim] (paged)
    idx_v_cache: Optional[
        torch.Tensor
    ],  # [max_slots, 1, idx_head_dim] (paged); None when disable_index_value
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    slot_ids: torch.Tensor,  # [batch_size, ]
    seq_lens: torch.Tensor,  # [batch_size, ]
    max_seqlen: int,  # max of seq_lens, passed from caller to avoid sync during CUDA graph capture
    block_size_q: int,  # useless for now, will always be 1
    block_size_k: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    idx_sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    dense_main_attn_fn: Optional[Callable] = None,
    page_size: int = 1,
    use_msa: bool = False,
    msa_kv_indices: Optional[
        torch.Tensor
    ] = None,  # per-forward MSA page table (cached)
    msa_plan=None,  # per-forward MSA fmha_sm100 plan (cached)
    # ---- IndexCache (SILOTIGER-721) ----
    skip_indexer: bool = False,  # "full" reuse: do NOT launch the indexer at all
    reuse_idx_o: Optional[torch.Tensor] = None,  # "full" reuse: index-head output to reuse
    reuse_main_topk_idx: Optional[torch.Tensor] = None,  # reuse this block selection for main attn
    reuse_real_seq_lens: Optional[torch.Tensor] = None,  # reuse effective KV lengths (dense path)
    return_index_state: bool = False,  # also return (idx_o, main_topk_idx, real_seq_lens) for caching
):
    # Step 1: Flash decode with topk index (using index head). When the dense main
    # attention is used, the indexer emits the page table directly (fused
    # transform) instead of block ids, plus the per-query effective KV length.
    #
    # IndexCache: `main_topk_idx` is the selection actually consumed by the main
    # sparse attention below (post GQA reduce, or the dense page table). A cadence
    # layer computes it here and hands it back (return_index_state=True); a reuse
    # layer either skips the indexer entirely (skip_indexer, "full" mode) or
    # recomputes idx_o but substitutes the cached selection ("topk" mode).
    num_idx_heads = idx_q.shape[1]
    num_kv_heads = k_cache.shape[1]
    idx_group_size = num_idx_heads // num_kv_heads
    real_seq_lens = reuse_real_seq_lens

    if skip_indexer:
        # "full"-mode reuse layer: no indexer launch at all.
        idx_o = reuse_idx_o
        main_topk_idx = reuse_main_topk_idx
    else:
        idx_o, topk_idx, real_seq_lens = flash_decode_with_topk_idx(
            q=idx_q,
            sink=idx_sink,
            k_cache=idx_k_cache,
            v_cache=idx_v_cache,
            req_to_token=req_to_token,
            seq_lens=seq_lens,
            max_seqlen=max_seqlen,
            slot_ids=slot_ids,
            block_size=block_size_k,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            sm_scale=idx_sm_scale,
            score_type=score_type,
            disable_index_value=disable_index_value,
            use_dense_main_attn=dense_main_attn_fn is not None,
            page_size=page_size,
        )
        if dense_main_attn_fn is not None:
            # topk_idx is the page table; real_seq_lens is the per-query cache_seqlens
            assert idx_group_size == 1
            main_topk_idx = topk_idx
        else:
            # Step 2: Reduce topk idx if num_idx_heads > num_kv_heads
            if idx_group_size > 1:
                topk_idx = topk_index_reduce(
                    topk_idx.view(num_kv_heads, idx_group_size, -1, topk), dim=1
                )
            main_topk_idx = topk_idx
        # "topk"-mode reuse: keep the freshly computed idx_o but override the
        # block selection (and dense effective KV lengths) with the cached ones.
        if reuse_main_topk_idx is not None:
            main_topk_idx = reuse_main_topk_idx
            if reuse_real_seq_lens is not None:
                real_seq_lens = reuse_real_seq_lens

    # Step 3: Sparse attention using the (possibly cached) topk selection. The
    # MSA path only replaces this step; keep the Triton path when sink is present.
    if dense_main_attn_fn is not None:
        # topk_idx is the page table; real_seq_lens is the per-query cache_seqlens
        o = dense_main_attn_fn(q, main_topk_idx, real_seq_lens)
    elif use_msa and sink is None:
        from .msa import MSAUnavailableError, msa_sparse_decode_main

        try:
            o = msa_sparse_decode_main(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                topk_idx=main_topk_idx,
                req_to_token=req_to_token,
                slot_ids=slot_ids,
                seq_lens=seq_lens,
                block_size_k=block_size_k,
                sm_scale=sm_scale,
                kv_indices=msa_kv_indices,
                plan=msa_plan,
            )
        except MSAUnavailableError as err:
            _warn_msa_fallback(err)
            o = flash_decode_with_gqa_share_sparse(
                q=q,
                sink=sink,
                k_cache=k_cache,
                v_cache=v_cache,
                req_to_token=req_to_token,
                seq_lens=seq_lens,
                slot_ids=slot_ids,
                block_size=block_size_k,
                topk_idx=main_topk_idx,
                sm_scale=sm_scale,
            )
    else:
        o = flash_decode_with_gqa_share_sparse(
            q=q,
            sink=sink,
            k_cache=k_cache,
            v_cache=v_cache,
            req_to_token=req_to_token,
            seq_lens=seq_lens,
            slot_ids=slot_ids,
            block_size=block_size_k,
            topk_idx=main_topk_idx,
            sm_scale=sm_scale,
        )

    if return_index_state:
        return idx_o, o, main_topk_idx, real_seq_lens
    return idx_o, o
