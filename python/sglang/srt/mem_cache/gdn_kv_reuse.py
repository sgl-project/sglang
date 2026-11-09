"""Borrow cached Full-Attention chunks while replaying GDN state."""

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.runtime_context import (
    attention_backends,
    get_disagg,
    get_exec,
    get_lora,
    get_memory,
    get_parallel,
    get_schedule,
    mamba_checkpoint_grid,
)

logger = logging.getLogger(__name__)


def try_borrow_full_kv(batch):
    """Return per-request borrowed slots, or None if no request can reuse KV.

    Admission still charges the ordinary extend budget. Each reusable request
    must have a page/checkpoint-aligned, entirely cached prefill chunk.
    """
    if not envs.SGLANG_ENABLE_GDN_KV_REUSE.get():
        return None
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    cache = batch.tree_cache
    config = batch.model_config
    parallel = get_parallel()
    if (
        not isinstance(cache, UnifiedRadixCache)
        or cache.disable
        or not cache.supports_mamba()
        or len(cache.components) != 2
        or not cache.enable_mamba_extra_buffer
        or get_memory().enable_hierarchical_cache
        or get_memory().enable_session_radix_cache
        or get_disagg().disaggregation_mode != "null"
        or not (
            (
                config.hf_text_config.model_type == "qwen3_5_text"
                and getattr(config.hf_text_config, "attn_output_gate", False)
            )
            or (
                config.hf_text_config.model_type == "olmo_hybrid"
                and getattr(config.hf_text_config, "sglang_olmo_hybrid_adapter", False)
            )
        )
        or config.quantization is not None
        or config.dtype != torch.bfloat16
        or not batch.spec_algorithm.is_none()
        or get_schedule().enable_mixed_chunk
        or get_lora().enable_lora
        or parallel.tp_size != 1
        or parallel.pp_size != 1
        or parallel.attn_cp_size != 1
        or parallel.attn_dp_size != 1
        or attention_backends()[0] != "flashinfer"
        or get_exec().graph.cuda_graph_backend_prefill != "disabled"
    ):
        return None
    borrowed = []
    try:
        for req in batch.reqs:
            borrowed.append(_try_borrow_request(cache, req))
    except Exception:
        for req, indices in zip(batch.reqs, borrowed):
            if indices is not None:
                req.kv.protect_prefix(req.prefix_len)
                release_full_kv_pin(cache, req)
        raise
    if all(indices is None for indices in borrowed):
        return None
    batch.full_kv_replay = True
    # Compact K/V are produced only for fresh tokens in a mixed batch. The
    # backend uses these same token indices to select physical write locations.
    fresh_tokens = []
    offset = 0
    for req, indices in zip(batch.reqs, borrowed):
        length = req.extend_end - req.prefix_len
        if indices is None:
            fresh_tokens.extend(range(offset, offset + length))
        offset += length
    # This per-forward tensor is never mutated in place. The overlap scheduler
    # retains every ScheduleBatch field until the forward stream is done.
    batch.full_kv_fresh_token_indices = (
        torch.tensor(fresh_tokens, dtype=torch.int64, device=batch.device)
        if fresh_tokens
        else None
    )
    logger.debug(
        "GDN_KV_REUSE_BATCH size=%s reused=%s fresh_tokens=%s",
        len(batch.reqs),
        sum(indices is not None for indices in borrowed),
        len(fresh_tokens),
    )
    return borrowed


def _try_borrow_request(cache, req):
    start, end = req.prefix_len, req.extend_end
    grid = mamba_checkpoint_grid(cache.page_size)
    if (
        start == 0
        or end <= start
        or start % cache.page_size
        or end % cache.page_size
        or (end - start) % grid
        or req.kv.cache_protected_len != start
        or req.full_kv_replay_node is not None
        or req.skip_radix_cache_insert
        or req.multimodal_inputs is not None
        or req.input_embeds is not None
        or req.positional_embed_overrides is not None
        or req.session_id is not None
        or (
            req.mamba_branching_seqlen is not None
            and start < req.mamba_branching_seqlen < end
        )
    ):
        return None
    key = RadixKey(
        req.full_untruncated_fill_ids,
        req.extra_key,
        limit=end,
        cache_salt=req.cache_salt,
    )
    hit, node, _ = cache.tree_core.match_full_device_prefix(key)
    if hit < end:
        return None
    cache.tree_core.inc_full_pin(node)
    try:
        indices = cache.path_device_indices(node)[start:end]
        assert len(indices) == end - start
    except Exception:
        cache.tree_core.dec_full_pin(node)
        raise
    req.full_kv_replay_node = node
    req.kv.protect_prefix(end)
    logger.debug(
        "GDN_KV_REUSE rid=%s C=%s H=%s tokens=%s", req.rid, start, end, end - start
    )
    return indices


def release_full_kv_pin(cache, req):
    """Release after a normal tree lock takes over, or on request teardown."""
    if req.full_kv_replay_node is not None:
        cache.tree_core.dec_full_pin(req.full_kv_replay_node)
        req.full_kv_replay_node = None
