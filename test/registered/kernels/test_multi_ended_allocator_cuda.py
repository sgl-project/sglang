import torch
from sglang.srt.mem_cache.multi_ended_allocator import MultiEndedAllocator
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    UnifiedKVPool,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class _MarkerKVCache:
    def __init__(self, max_slots: int):
        self.markers = torch.full((max_slots,), -1, dtype=torch.int64, device="cuda")

    def move_kv_cache(self, dst_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        self.markers[dst_loc] = self.markers[src_loc]


def test_large_lazy_compaction_device_remap_preserves_data():
    full = MHASubPoolSpec(
        name="full",
        layer_num=1,
        head_num=1,
        head_dim=8,
        store_dtype=torch.float16,
        grow_direction="up",
    )
    mamba = MambaSubPoolSpec(
        name="mamba",
        layer_num=1,
        conv_state_shapes=((4, 3),),
        conv_dtype=torch.float32,
        temporal_state_shape=(2, 2, 2),
        temporal_dtype=torch.float32,
        grow_direction="down",
    )
    total_bytes = full.entry_bytes() * 256 + mamba.entry_bytes() * 64
    pool = UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full, mamba],
        device="cuda",
        enable_memory_saver=False,
    )
    full_cache = _MarkerKVCache(pool.max_slots("full"))
    mamba_cache = _MarkerKVCache(pool.max_slots("mamba"))
    full_alloc = MultiEndedAllocator(
        kvcache=full_cache,
        unified_buffer=pool,
        sub_pool_name="full",
        device="cuda",
        is_id_owner=True,
        lazy_compaction=True,
    )
    mamba_alloc = MultiEndedAllocator(
        kvcache=mamba_cache,
        unified_buffer=pool,
        sub_pool_name="mamba",
        device="cuda",
        is_id_owner=True,
        lazy_compaction=True,
    )
    full_alloc.bind_peer(mamba_alloc)
    mamba_alloc.bind_peer(full_alloc)

    virtual = full_alloc.alloc(64)
    assert virtual is not None
    physical = full_alloc.virtual_to_physical[virtual]
    full_cache.markers[physical] = virtual

    freed = virtual[:32:2]
    full_alloc.free(freed)
    moves = full_alloc._flush(urgent=True)
    torch.cuda.synchronize()

    assert moves == 16
    live = virtual[~torch.isin(virtual, freed)]
    live_physical = full_alloc.virtual_to_physical[live]
    assert torch.equal(full_cache.markers[live_physical], live)
    assert torch.equal(full_alloc.physical_to_virtual[live_physical], live)
