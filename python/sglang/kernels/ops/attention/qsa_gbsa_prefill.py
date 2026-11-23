"""Sequence-batched native prefill with GBSA's exact automatic residual tail.

QSA supplies either complete group IDs with -1 padding or the legacy token IDs
with complete groups of four packed before an incomplete tail. Compact groups
must satisfy the same visibility and prefix-valid contract as QSA expansion.
Token tables must map aligned 128-token logical pages to physical KV pages.
"""
import logging
import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _report_native():
    logging.getLogger(__name__).info(
        "QSA prefill uses CANN GBSA: D256, block [1,4], KV page128, residual mode1"
    )


def qsa_gbsa_prefill(
    q,
    k,
    v,
    logical_indices,
    token_table,
    seq_lens,
    extend_lens,
    max_query,
    softmax_scale=None,
    *,
    indices_are_blocks=False,
    page_table=None,
    cu_seqlens_q=None,
):
    from sgl_kernel_npu.operators.attention import (
        generic_block_sparse_attention,
        generic_block_sparse_attention_metadata,
    )
    assert q.shape[1:] == (3, 256) and k.shape[1:] == (1, 256)
    assert k.shape == v.shape and k.shape[0] % 128 == 0
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert logical_indices.shape == (q.shape[0], 512 if indices_are_blocks else 2051)
    assert token_table is None or token_table.shape[1] < 2**24
    _report_native()
    if (
        indices_are_blocks
        and logical_indices.device.type == "npu"
        and logical_indices.dtype == torch.int32
        and logical_indices.is_contiguous()
        and logical_indices.shape[0] > 0
        and os.getenv("SGLANG_NPU_QSA_GBSA_SORT_COUNTS", "0") == "1"
    ):
        from .qsa_gbsa_sort_counts import sort_count_groups

        ids, counts = sort_count_groups(logical_indices)
    else:
        if indices_are_blocks:
            valid = logical_indices >= 0
            groups = valid.sum(-1, dtype=torch.int32)
            ids = torch.where(valid, logical_indices, 16777215)
        else:
            groups = (logical_indices >= 0).sum(-1, dtype=torch.int32) // 4
            ranks = torch.arange(512, device=q.device)
            block_ids = logical_indices[:, :2048:4] // 4
            ids = torch.where(ranks[None] < groups[:, None], block_ids, 16777215)
        # Keep the original order of attention's reduction. Removing this sort is
        # a separate numerical/performance experiment, not needed for compact input.
        ids = ids.float().sort(-1).values.int().unsqueeze(0)
        counts = groups[None]
    table = (
        (token_table[:, ::128] // 128).to(torch.int32).contiguous()
        if page_table is None else page_table.to(torch.int32).contiguous()
    )
    cu_q = (
        torch.cat([torch.zeros(1, dtype=torch.int64, device=q.device), extend_lens.to(torch.int64).cumsum(0)])
        if cu_seqlens_q is None else cu_seqlens_q
    )
    assert cu_q.dtype == torch.int64
    options = dict(cu_seqlens_q=cu_q, seqused_kv=seq_lens.to(torch.int32).contiguous(),
                   layout_q='TND', layout_kv='PA_BBND', layout_sparse_pattern=4,
                   mask_mode=1, quant_mode=0, softmax_precision=0,
                   residual_block_mode=1, is_consistent_topk=False)
    meta = generic_block_sparse_attention_metadata(
        ids, counts, 3, 1, 256, [1,4], max_seqlen_q=max_query,
        max_seqlen_kv=table.shape[1] * 128, **options)
    output, _ = generic_block_sparse_attention(
        q.contiguous(), k.view(-1,128,1,256), v.view(-1,128,1,256), ids,
        counts, [1,4], metadata=meta, block_table=table, softmax_scale=1/16 if softmax_scale is None else softmax_scale,
        return_softmax_lse=False, **options)
    return output
