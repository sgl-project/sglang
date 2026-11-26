"""Sequence-batched native prefill with GBSA's exact automatic residual tail.

QSA supplies either complete group IDs with -1 padding or the legacy token IDs
with complete groups of four packed before an incomplete tail. Compact groups
must satisfy the same visibility and prefix-valid contract as QSA expansion.
Token tables must map aligned 128-token logical pages to physical KV pages.
"""
import logging
import os
from contextvars import ContextVar
from functools import lru_cache, wraps

import torch


_paged_cache = ContextVar("qsa_paged_gbsa_cache", default=None)


def with_paged_gbsa_cache(forward):
    if os.getenv("SGLANG_NPU_QSA_REUSE_GBSA_META", "1") != "1":
        return forward
    flags = (
        "SGLANG_NPU_QSA_PAGED_GBSA",
        "SGLANG_NPU_QSA_LIGHTNING_INDEXER",
    )
    if not all(os.getenv(flag, "0") == "1" for flag in flags):
        return forward

    @wraps(forward)
    def run(self, input_ids, positions, forward_batch, *args, **kwargs):
        if torch.compiler.is_compiling():
            return forward(self, input_ids, positions, forward_batch, *args, **kwargs)
        mode = forward_batch.forward_mode
        active = mode.is_decode() or mode.is_target_verify()
        token = _paged_cache.set({} if active else None)
        try:
            return forward(self, input_ids, positions, forward_batch, *args, **kwargs)
        finally:
            _paged_cache.reset(token)

    return run


@lru_cache(maxsize=1)
def _report_native():
    logging.getLogger(__name__).info(
        "QSA prefill uses CANN GBSA: D256, block [1,4], KV page128, residual mode1"
    )


def _gbsa_options(cu_q, seq_lens):
    return dict(cu_seqlens_q=cu_q, seqused_kv=seq_lens.to(torch.int32).contiguous(),
                layout_q='TND', layout_kv='PA_BBND', layout_sparse_pattern=4,
                mask_mode=1, quant_mode=0, softmax_precision=0,
                residual_block_mode=1, is_consistent_topk=False)


def prepare_prefill_gbsa_metadata(row_ranges, inputs, seq_lens, extend_lens, page_count):
    """Build the native schedules for this rank's GBSA spans before the first layer."""
    from sgl_kernel_npu.operators.attention import generic_block_sparse_attention_metadata

    starts, ends = row_ranges
    if inputs is None:
        cu_q = torch.nn.functional.pad(extend_lens.to(torch.int64).cumsum(0), (1, 0))
        calls = ((0, starts.numel(), 0, seq_lens.numel(), cu_q),)
    else:
        calls = inputs.gbsa_calls
        seq_lens = inputs.sequence_lengths
    schedules = {}
    for start, end, first, last, cu_q in calls:
        counts = (ends[start:end] - starts[start:end]).clamp_max_(512).unsqueeze(0)
        # The CANN scheduler reads the index tensor's shape only. Actual block
        # IDs are selected by each layer and passed to the attention operator.
        shape_tensor = counts.new_empty((1, end - start, 512))
        schedules[start, end] = generic_block_sparse_attention_metadata(
            shape_tensor, counts, 3, 1, 256, [1, 4],
            max_seqlen_q=end - start, max_seqlen_kv=page_count * 128,
            **_gbsa_options(cu_q, seq_lens[first:last]),
        )
    return schedules


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
    sorted_block_counts=None,
    meta_cache=None,
    span=None,
    paged_step=None,
    fixed_topk_counts=False,
    prepared_metadata=None,
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
    if sorted_block_counts is not None:
        ids, counts = logical_indices.unsqueeze(0), sorted_block_counts
    elif (
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
    options = _gbsa_options(cu_q, seq_lens)
    # For native QSA, each row selects min(512, visible complete blocks).
    # This schedule is shared across layers; the block IDs below are layer-local.
    paged = paged_step is not None
    fixed_counts = paged or fixed_topk_counts
    if paged:
        meta_cache = None if torch.compiler.is_compiling() else _paged_cache.get()
    elif meta_cache is not None and (
        not indices_are_blocks
        or torch.compiler.is_compiling()
        or (q.device.type == "npu" and torch.npu.is_current_stream_capturing())
    ):
        meta_cache = None

    entry = None
    if meta_cache is not None:
        key = (
            id(paged_step) if paged else span,
            fixed_counts,
            q.device,
            ids.shape,
            ids.dtype,
            max_query,
            table.shape[1],
        )
        entry = meta_cache.get(key)
    if prepared_metadata is not None:
        meta = prepared_metadata
    elif entry is not None and (fixed_counts or torch.equal(counts, entry[0])):
        meta = entry[1]
    else:
        meta = generic_block_sparse_attention_metadata(
            ids, counts, 3, 1, 256, [1,4], max_seqlen_q=max_query,
            max_seqlen_kv=table.shape[1] * 128, **options)
        if meta_cache is not None:
            meta_cache[key] = (None if fixed_counts else counts, meta)
    output, _ = generic_block_sparse_attention(
        q.contiguous(), k.view(-1,128,1,256), v.view(-1,128,1,256), ids,
        counts, [1,4], metadata=meta, block_table=table, softmax_scale=1/16 if softmax_scale is None else softmax_scale,
        return_softmax_lse=False, **options)
    return output
