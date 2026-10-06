"""Small unified-pool builders and fakes shared by allocator regression tests."""

from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    UnifiedKVPool,
    init_unified_mamba_swa_pools,
    init_unified_swa_pools,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs


def setup_allocator_context():
    reset_context()
    publish(ServerArgs(model_path="dummy", device="cpu"), role="tokenizer")


def build_swa_pool(*, page_size=1):
    """Lazy full + SWA composite over real CPU pools, 100 pages per side."""
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
        end_layer=2,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        full_max_total_num_tokens=100 * page_size,
        swa_max_total_num_tokens=100 * page_size,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=True,
    )
    return bundle.token_to_kv_pool_allocator


def build_tri_pool(*, page_size=1):
    """Lazy full + SWA + mamba composite over real CPU pools; returns the pool
    bundle and its allocator."""
    config = SimpleNamespace(
        shape=SimpleNamespace(conv=[(3, 8)], temporal=(1, 4, 8)),
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
        end_layer=2,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        full_max_total_num_tokens=100 * page_size,
        swa_max_total_num_tokens=100 * page_size,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=True,
        mamba_layer_ids=[0],
        mamba2_cache_params=config,
        max_mamba_cache_size=8,
        model_context_len=256 * page_size,
        extra_max_context_len=1,
        max_num_reqs=4,
        enable_mamba_extra_buffer=False,
        enable_mamba_extra_buffer_lazy=False,
        disable_overlap_schedule=True,
        sliding_window_size=32,
    )
    return bundle, bundle.token_to_kv_pool_allocator


class FakeKVCache:
    """buf[p] == virtual id stored at physical slot p (-1 free); moves copy it."""

    def __init__(self, max_slots: int):
        self.buf = torch.full((max_slots,), -1, dtype=torch.int64)

    def move_kv_cache(self, dst_loc: torch.Tensor, src_loc: torch.Tensor):
        self.buf[dst_loc] = self.buf[src_loc].clone()


class FakeUnifiedSWAKVPool:
    class _SubKV(FakeKVCache):
        def __init__(self, max_slots):
            super().__init__(max_slots)
            self.allocator = None

        def attach_allocator(self, allocator):
            self.allocator = allocator

    def __init__(self, shared_pool: UnifiedKVPool):
        self.full_kv_pool = self._SubKV(shared_pool.max_slots("full"))
        self.swa_kv_pool = self._SubKV(shared_pool.max_slots("swa"))
        self._full_allocator = None
        self._swa_allocator = None

    def attach_allocators(self, *, full_allocator, swa_allocator):
        self._full_allocator = full_allocator
        self._swa_allocator = swa_allocator


def tri_sub_pool_specs(
    full_layer_num=4, swa_layer_num=2, state_layer_num=2, head_num=2, head_dim=4
):
    full = MHASubPoolSpec(
        name="full",
        layer_num=full_layer_num,
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=torch.float16,
        grow_direction="down",
    )
    swa = MHASubPoolSpec(
        name="swa",
        layer_num=swa_layer_num,
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=torch.float16,
        grow_direction="float",
    )
    mamba = MambaSubPoolSpec(
        name="mamba",
        layer_num=state_layer_num,
        conv_state_shapes=((3, 8),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(0, 0, 0),  # Inkling: conv-only, no SSM state
        temporal_dtype=torch.float32,
        grow_direction="up",
    )
    return full, swa, mamba
