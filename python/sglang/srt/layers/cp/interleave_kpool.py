"""KPool compression and query metadata for interleave context parallelism."""

import torch

from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.layers.cp.interleave import (
    interleave_rows_per_request,
    is_interleave_extend,
)
from sglang.srt.runtime_context import get_parallel


def local_query_inputs(forward_batch, metadata):
    """Compression covers full requests; logits cover only this rank's queries."""
    if not is_interleave_extend(forward_batch):
        return {}
    parallel = get_parallel()
    counts = interleave_rows_per_request(
        forward_batch.extend_seq_lens_cpu, parallel.attn_cp_rank, parallel.attn_cp_size
    )
    indices = torch.tensor(
        [i for i, count in enumerate(counts) if count],
        dtype=torch.long,
        device=forward_batch.req_pool_indices.device,
    )
    return dict(
        local_real_page_table=metadata.real_page_table,
        local_seqlens_expanded=metadata.dsa_seqlens_expanded,
        local_extend_seq_lens_cpu=metadata.dsa_extend_seq_lens_list,
        local_seq_lens_cpu=metadata.indexer_seq_lens_cpu.tolist(),
        local_req_pool_indices=forward_batch.req_pool_indices.index_select(0, indices),
    )


def materialize_compression_inputs(key, score, positions, forward_batch):
    """Replicate ordered keys/scores before pooling across adjacent tokens.

    Queries and head gates stay local. Every rank writes its replicated index
    cache and tail state, so subsequent decode needs no layer-owner protocol.
    """
    if not is_interleave_extend(forward_batch):
        return key, score, positions
    width = key.shape[-1]
    packed = torch.cat((key, score), dim=-1)
    packed = get_cp_strategy().gather_kv_cache(packed, forward_batch)
    return (
        packed[..., :width].contiguous(),
        packed[..., width:].contiguous(),
        forward_batch.positions,
    )
