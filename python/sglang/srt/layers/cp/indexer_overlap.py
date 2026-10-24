"""Overlap interleave CP attention KV replication with DSA scoring.

The indexer owns its compute boundary; this module owns streams and lifetimes.
Index-key collectives must finish before on_logits_start is invoked.
"""

import torch

from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)


def indexer_with_cp_kv(
    indexer, x, q_lora, positions, forward_batch, layer_id, kv_local
):
    if is_in_breakable_cuda_graph():
        from sglang.srt.layers.cp.bcg.indexer_overlap import bcg_indexer_with_kv

        output = torch.empty(
            (x.shape[0], indexer.topk_output_width), dtype=torch.int32, device=x.device
        )
        kv_output = kv_local.new_empty(
            (forward_batch.extend_num_tokens, kv_local.shape[-1])
        )
        bcg_indexer_with_kv(
            indexer, x, q_lora, positions, output, layer_id, kv_local, kv_output
        )
        return output, kv_output
    return run_indexer_with_kv_gather(
        indexer, x, q_lora, positions, forward_batch, layer_id, kv_local
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

    topk = indexer.forward_cuda(
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
            # Match the serial TRT-LLM FP8 path: communicate and reorder raw
            # bytes, since not all collectives/indexing kernels support FP8.
            if kv_local.dtype == torch.float8_e4m3fn:
                return (
                    get_cp_strategy()
                    .gather_kv_cache(kv_local.view(torch.uint8), forward_batch)
                    .view(kv_local.dtype)
                )
            return get_cp_strategy().gather_kv_cache(kv_local, forward_batch)

    try:
        kv_full = gather_kv()
    finally:
        current_stream.wait_stream(stream)
    # The result was allocated on the side stream and outlives this function.
    # Keep its storage until attention (or the BCG bridge copy) finishes.
    kv_full.record_stream(current_stream)
    return topk, kv_full
