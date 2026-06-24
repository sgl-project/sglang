"""DeepSeek V4 wrappers around the LayerSplit pool API."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.attention.dsa.utils import (
    dsa_cp_round_robin_split_data,
    dsa_use_prefill_cp,
)
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.mem_cache.cp_kv_layer_split.deepseek_v4_pool import (
    CpKvLayerSplitDeepSeekV4TokenToKVPool,
)
from sglang.srt.model_executor.forward_context import get_attn_backend


def is_cp_kv_layer_split_deepseek_v4_pool(pool) -> bool:
    return isinstance(pool, CpKvLayerSplitDeepSeekV4TokenToKVPool)


def _should_sync_cp_kv_layer_split(pool, forward_batch) -> bool:
    if not is_cp_kv_layer_split_deepseek_v4_pool(pool):
        return False
    # LayerSplit collectives are defined only for prefill CP extend. Decode,
    # warmup, and CUDA graph capture must not enter CP barriers or broadcasts.
    return forward_batch is not None and dsa_use_prefill_cp(forward_batch)


def cp_kv_layer_split_pre_compressor_skip(
    pool, layer_id: int, forward_batch, *, is_indexer: bool
) -> bool:
    """Return whether this rank should skip the owner-only compressor write."""
    if not is_cp_kv_layer_split_deepseek_v4_pool(pool):
        return False
    if not dsa_use_prefill_cp(forward_batch):
        pool.clear_staging_remap_for_read()
    if is_indexer:
        return pool.should_skip_indexer_compressor_write(layer_id)
    return pool.should_skip_core_compressor_write(layer_id)


def maybe_start_cp_kv_swa_for_read(pool, layer_id: int, forward_batch=None) -> None:
    if _should_sync_cp_kv_layer_split(pool, forward_batch):
        core_metadata = _get_core_attn_metadata()
        pool.start_swa_layer_for_read(layer_id, core_metadata.swa_page_indices)


def maybe_finish_cp_kv_swa_for_read(pool, layer_id: int, forward_batch=None) -> None:
    if _should_sync_cp_kv_layer_split(pool, forward_batch):
        pool.finish_swa_layer_for_read(layer_id)


def maybe_sync_cp_kv_extra_for_read(pool, layer_id: int, forward_batch=None) -> None:
    if _should_sync_cp_kv_layer_split(pool, forward_batch):
        if pool.should_use_c4_extra_broadcast_overlap(layer_id):
            core_metadata = _get_core_attn_metadata()
            pool.start_page_table_extra_key_layer_for_read(
                layer_id, core_metadata.page_table
            )
            return
        core_metadata = _get_core_attn_metadata()
        pool.sync_extra_key_layer_for_read(layer_id, core_metadata)


def maybe_sync_cp_kv_indexer_for_read(pool, layer_id: int, forward_batch=None) -> None:
    if _should_sync_cp_kv_layer_split(pool, forward_batch):
        attn_backend = get_attn_backend()
        if hasattr(attn_backend, "_maybe_upgrade_forward_metadata"):
            attn_backend._maybe_upgrade_forward_metadata()
        metadata = getattr(attn_backend, "forward_metadata", None)
        indexer_metadata = getattr(metadata, "indexer_metadata", None)
        if indexer_metadata is None:
            raise RuntimeError("CP KV LayerSplit requires DSV4 indexer metadata")
        pool.sync_index_k_layer_for_read(layer_id, indexer_metadata.page_table)


def _get_core_attn_metadata():
    attn_backend = get_attn_backend()
    if hasattr(attn_backend, "_maybe_upgrade_forward_metadata"):
        attn_backend._maybe_upgrade_forward_metadata()
    metadata = getattr(attn_backend, "forward_metadata", None)
    core_metadata = getattr(metadata, "core_attn_metadata", None)
    if core_metadata is None:
        raise RuntimeError("CP KV LayerSplit requires DSV4 core attention metadata")
    return core_metadata


def cp_kv_layer_split_resolve_store_swa_loc(
    pool,
    layer_id: int,
    forward_batch,
    out_cache_loc: torch.Tensor,
    swa_k_shape0: int,
) -> Optional[torch.Tensor]:
    """Resolve SWA write locations, or ``None`` when this rank skips the write."""
    raw_loc = maybe_all_gather_cp_kv_layer_split_raw_loc(
        pool, out_cache_loc, forward_batch, swa_k_shape0
    )
    if pool.should_skip_swa_write(layer_id):
        return None
    return pool.translate_loc_from_full_to_swa(raw_loc).to(torch.int32)


def maybe_all_gather_cp_kv_layer_split_raw_loc(
    pool, raw_loc: torch.Tensor, forward_batch, kv_num_tokens: int
) -> torch.Tensor:
    """Align SWA cache locations with the full-KV shape under DSA prefill CP."""
    if not _should_sync_cp_kv_layer_split(pool, forward_batch):
        return raw_loc

    if raw_loc.shape[0] == kv_num_tokens:
        return raw_loc

    if raw_loc.shape[0] == kv_num_tokens * pool.cp_size:
        return dsa_cp_round_robin_split_data(raw_loc)

    if raw_loc.shape[0] * pool.cp_size != kv_num_tokens:
        raise RuntimeError(
            "CP KV LayerSplit cannot align SWA cache locations with KV tensor: "
            f"raw_loc={raw_loc.shape[0]}, kv_num_tokens={kv_num_tokens}, "
            f"cp_size={pool.cp_size}"
        )

    return cp_all_gather_rerange_output(
        raw_loc.contiguous(),
        pool.cp_size,
        forward_batch,
        torch.cuda.current_stream(),
    )
