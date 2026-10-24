"""Overlap interleave CP attention KV replication with DSA scoring.

This module owns stream ordering and lifetimes. Attention KV is independent
of the indexer and is submitted before any index-key collective on every rank.
"""

import torch

from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)


class DSAInterleaveCPIndexerOverlap:
    """Per-forward MLA orchestration for attention KV/indexer overlap.

    No-RoPE models can launch the indexer at its original boundary. RoPE
    models defer it until the final local KV is available. All state belongs
    to this forward call, including while tracing a breakable CUDA graph.
    """

    def __init__(self, attention, forward_batch):
        self.attention = attention
        self.forward_batch = forward_batch
        self._kv = None
        self._deferred = None

    @property
    def can_fuse_bmm(self):
        from sglang.srt.layers.cp.interleave import is_interleave_extend

        # Deferred RoPE indexers need materialized absorbed Q before attention.
        return not (
            self.attention.rotary_emb is not None
            and is_interleave_extend(self.forward_batch)
        )

    def _can_overlap(self):
        from sglang.srt.layers.cp.interleave import is_interleave_extend
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang.srt.utils import is_cuda

        attention = self.attention
        if not (
            is_cuda()
            and is_interleave_extend(self.forward_batch)
            and hasattr(attention.indexer, "topk_output_width")
        ):
            return False
        if not attention._fuse_rope_for_trtllm_mla(self.forward_batch):
            return True
        backend = get_attn_backend()
        return (
            getattr(backend, "dsa_prefill_impl", None) == "trtllm"
            and not backend.use_mha
        )

    def run_indexer(self, *, k_nope, latent_cache, **indexer_args):
        attention = self.attention
        if not self._can_overlap():
            return attention.indexer(**indexer_args)
        if attention.rotary_emb is not None:
            self._deferred = indexer_args
            return None
        kv_local = torch.cat(
            (k_nope.squeeze(1), latent_cache[..., attention.kv_lora_rank :]),
            dim=-1,
        )
        topk, self._kv = indexer_with_cp_kv(
            attention.indexer, kv_local=kv_local, **indexer_args
        )
        return topk

    def prepare_qkv(self, q_nope, q_pe, k_nope, k_pe, topk, *, defer_materialization):
        from sglang.srt.layers.attention.dsa_backend import prepare_kv_for_attention

        attention = self.attention
        if self._deferred is not None:
            if defer_materialization:
                from sglang.kernels.ops.attention.utils import (
                    mla_quantize_and_rope_for_fp8,
                )

                # Preserve the backend's fused numerics, with local Q/K rows
                # and local positions, before the KV all-gather.
                q_nope, k_nope, k_pe = mla_quantize_and_rope_for_fp8(
                    q_nope,
                    q_pe,
                    k_nope.squeeze(1),
                    k_pe.squeeze(1),
                    self._deferred["positions"],
                    attention.rotary_emb.cos_sin_cache,
                    attention.rotary_emb.is_neox_style,
                    attention.kv_lora_rank,
                    attention.qk_rope_head_dim,
                )
                # The backend must consume prepared Q/K without a second RoPE.
                q_pe = None
            kv_local = torch.cat(
                (
                    k_nope.reshape(-1, attention.kv_lora_rank),
                    k_pe.reshape(-1, attention.qk_rope_head_dim),
                ),
                dim=-1,
            )
            topk, self._kv = indexer_with_cp_kv(
                attention.indexer, kv_local=kv_local, **self._deferred
            )

        if self._kv is not None:
            k_nope = self._kv[..., : attention.kv_lora_rank].unsqueeze(1)
            k_pe = self._kv[..., attention.kv_lora_rank :].unsqueeze(1)
        else:
            k_nope, k_pe = prepare_kv_for_attention(
                attention,
                self.forward_batch,
                k_nope,
                k_pe,
                defer_materialization=defer_materialization,
            )
        return q_nope, q_pe, k_nope, k_pe, topk


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
    """Start KV replication before indexer projection, compression and scoring.

    The producer event covers only local KV preparation. All ranks submit the
    attention KV collective before the index-key collective, including short
    prefills that skip scoring. Join before returning to attention or the BCG
    bridge so both eager and graph consumers observe completed KV writes.
    """
    from sglang.srt.runtime_context import get_stream

    current_stream = torch.cuda.current_stream()
    stream = get_stream("dsa_cp_kv")
    stream.wait_stream(current_stream)

    def gather_kv():
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
        topk = indexer.forward_cuda(
            x=x,
            q_lora=q_lora,
            positions=positions,
            forward_batch=forward_batch,
            layer_id=layer_id,
            # This region already owns the eager break and stable BCG buffers.
            use_cuda_graph=False,
        )
    finally:
        current_stream.wait_stream(stream)
    # The result was allocated on the side stream and outlives this function.
    # Keep its storage until attention (or the BCG bridge copy) finishes.
    kv_full.record_stream(current_stream)
    return topk, kv_full
