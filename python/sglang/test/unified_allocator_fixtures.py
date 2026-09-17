"""Small real-pool builders shared by unified allocator regression tests.

These helpers contain no TestCase classes or reclaim-specific assertions.
"""

from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_memory_pool import (
    init_unified_mamba_swa_pools,
    init_unified_swa_pools,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs


def setup_allocator_context():
    reset_context()
    publish(ServerArgs(model_path="Qwen/Qwen3-0.6B", device="cpu"), role="tokenizer")


def build_swa_pool(
    *,
    occupancy=96,
    lazy=False,
    page_size=1,
    ratio=(1, 1),
    total_bytes=None,
    sessions=False,
    window=None,
):
    full_layers, swa_layers = ratio
    bundle = init_unified_swa_pools(
        device="cpu",
        kv_cache_dtype=torch.float16,
        head_num=1,
        head_dim=8,
        v_head_dim=8,
        swa_head_num=1,
        swa_head_dim=8,
        swa_v_head_dim=8,
        page_size=page_size,
        start_layer=0,
        end_layer=full_layers + swa_layers,
        swa_attention_layer_ids=list(range(full_layers, full_layers + swa_layers)),
        full_attention_layer_ids=list(range(full_layers)),
        full_max_total_num_tokens=100 * page_size,
        swa_max_total_num_tokens=100 * page_size,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=lazy,
        total_bytes=total_bytes,
    )
    allocator = bundle.token_to_kv_pool_allocator
    req_pool = ReqToTokenPool(
        size=4,
        max_context_len=256 * page_size,
        device="cpu",
        enable_memory_saver=False,
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            enable_session_radix_cache=sessions,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            sliding_window_size=128 * page_size if window is None else window,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
        )
    )
    slots = allocator.alloc(occupancy * page_size)
    assert slots is not None, "Fixture allocation must fit"
    for i in range(occupancy):
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(i * 1000, i * 1000 + page_size))),
                value=slots[i * page_size : (i + 1) * page_size],
            )
        )
    return allocator, cache


def build_tri_pool(
    *,
    lazy=True,
    temporal=(0, 0, 0),
    page_size=1,
    ratio=(1, 1),
    sessions=False,
    state_cache=True,
):
    full_layers, swa_layers = ratio
    config = SimpleNamespace(
        shape=SimpleNamespace(conv=[(3, 8)], temporal=temporal),
        dtype=SimpleNamespace(conv=torch.bfloat16, temporal=torch.float32),
        layers=[0],
    )
    bundle = init_unified_mamba_swa_pools(
        device="cpu",
        kv_cache_dtype=torch.float16,
        head_num=1,
        head_dim=8,
        v_head_dim=8,
        swa_head_num=1,
        swa_head_dim=8,
        swa_v_head_dim=8,
        page_size=page_size,
        start_layer=0,
        end_layer=full_layers + swa_layers,
        swa_attention_layer_ids=list(range(full_layers, full_layers + swa_layers)),
        full_attention_layer_ids=list(range(full_layers)),
        full_max_total_num_tokens=100 * page_size,
        swa_max_total_num_tokens=100 * page_size,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=lazy,
        mamba_layer_ids=[0],
        mamba2_cache_params=config,
        max_mamba_cache_size=8,
        model_context_len=256 * page_size,
        extra_max_context_len=1,
        max_num_reqs=4,
        enable_mamba_extra_buffer=False,
        disable_overlap_schedule=True,
        sliding_window_size=32,
    )
    a = bundle.token_to_kv_pool_allocator
    c = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=bundle.req_to_token_pool,
            token_to_kv_pool_allocator=a,
            page_size=page_size,
            sliding_window_size=128 * page_size,
            tree_components=(
                (ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA)
                if state_cache
                else (ComponentType.FULL, ComponentType.SWA)
            ),
            enable_session_radix_cache=sessions,
        )
    )
    return bundle, a, c
