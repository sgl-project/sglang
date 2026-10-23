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


def run_indexer_with_kv_gather(
    indexer, x, q_lora, positions, forward_batch, layer_id, kv_local
):
    """Overlap MLA KV replication with already-enqueued indexer kernels.

    Record readiness immediately before MQA logits, then enqueue the KV gather
    after submitting the indexer. Waiting on that event (rather than the
    current stream tail) starts communication alongside logits and top-k.
    """
    from sglang.srt.runtime_context import get_stream

    current_stream = torch.cuda.current_stream()
    stream = get_stream("dsa_cp_kv")
    ready = torch.cuda.Event()
    marked_ready = False

    def mark_ready():
        nonlocal marked_ready
        assert not marked_ready, "The indexer overlap boundary must be recorded once"
        ready.record(current_stream)
        marked_ready = True

    topk = indexer._forward_cuda_impl(
        x=x,
        q_lora=q_lora,
        positions=positions,
        forward_batch=forward_batch,
        layer_id=layer_id,
        on_logits_start=mark_ready,
    )
    if not marked_ready:
        # Short/empty or unplanned prefills can skip MQA logits. Preserve
        # the usual gather after their indexer (and index-key collective).
        mark_ready()

    def gather_kv():
        stream.wait_event(ready)
        with torch.cuda.stream(stream):
            return get_cp_strategy().gather_kv_cache(kv_local, forward_batch)

    try:
        kv_full = gather_kv()
    finally:
        current_stream.wait_stream(stream)
    # The result was allocated on the side stream and outlives this function.
    # Keep its storage until attention (or the BCG bridge copy) finishes.
    kv_full.record_stream(current_stream)
    return topk, kv_full
