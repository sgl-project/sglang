"""Sparse KV offload row mappings and target KV layout on NPU."""

import torch


def prepare_sparse_kv_offload(
    k,
    k_rope,
    request_ids,
    seq_lens,
    positions,
    out_cache_loc,
    *,
    request_rows,
    context_len,
    host_capacity,
    query_width=1,
    extend_seq_lens=None,
    extend_prefix_lens=None,
    compact_prefill=False,
):
    """Prepare offload rows and, for target decode/verify, concatenate KV."""
    from sgl_kernel_npu.operators.mem_cache.prepare_sparse_kv_offload_rows import (
        prepare_sparse_kv_offload_rows,
    )

    rows = prepare_sparse_kv_offload_rows(
        k.shape[0],
        k.device,
        request_ids,
        seq_lens,
        positions,
        out_cache_loc,
        request_rows=request_rows,
        context_len=context_len,
        host_capacity=host_capacity,
        query_width=query_width,
        extend_seq_lens=extend_seq_lens,
        extend_prefix_lens=extend_prefix_lens,
        compact_prefill=compact_prefill,
    )
    current = (
        (k, k_rope) if extend_seq_lens is not None else torch.cat((k, k_rope), dim=-1)
    )
    return (*rows, current)
