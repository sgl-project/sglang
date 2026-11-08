"""Borrow an entirely cached Full-Attention chunk while replaying GDN state."""

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
    """Return existing extend slots, or None without changing the request.

    Admission still charges the ordinary extend budget. This first path only
    handles a single, page/checkpoint-aligned, entirely cached prefill chunk.
    In particular it never splits a chunk or lets a mixed batch write borrowed KV.
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
        or config.hf_text_config.model_type != "qwen3_5_text"
        or not getattr(config.hf_text_config, "attn_output_gate", False)
        or config.quantization is not None
        or config.dtype != torch.bfloat16
        or len(batch.reqs) != 1
        or batch.enable_overlap
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
    req = batch.reqs[0]
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
    batch.full_kv_replay = True
    logger.debug(
        "GDN_KV_REUSE rid=%s C=%s H=%s tokens=%s", req.rid, start, end, end - start
    )
    return indices


def release_full_kv_pin(cache, req):
    """Release after a normal tree lock takes over, or on request teardown."""
    if req.full_kv_replay_node is not None:
        cache.tree_core.dec_full_pin(req.full_kv_replay_node)
        req.full_kv_replay_node = None
