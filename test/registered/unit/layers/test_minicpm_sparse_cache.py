from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.minicpm.cache import MiniCPMReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class RecordingAllocator:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.next_slot = 1
        self.live: set[int] = set()

    def alloc(self, size: int):
        if size > self.available_size():
            return None
        slots = torch.arange(self.next_slot, self.next_slot + size, dtype=torch.int32)
        self.next_slot += size
        self.live.update(slots.tolist())
        return slots

    def free(self, slots: torch.Tensor):
        self.live.difference_update(slots.tolist())

    def available_size(self):
        return self.capacity - len(self.live)


class ChunkCacheStub:
    def __init__(self, allocator: RecordingAllocator):
        self.token_to_kv_pool_allocator = allocator

    def is_chunk_cache(self):
        return True

    def available_and_evictable_str(self):
        return f"available={self.token_to_kv_pool_allocator.available_size()}"

    def pretty_print(self):
        return ""


def make_pool_and_req(capacity: int = 64):
    pool = MiniCPMReqToTokenPool(
        size=2,
        max_context_len=64,
        device="cpu",
        enable_memory_saver=False,
        kernel_size=4,
        kernel_stride=2,
    )
    req = SimpleNamespace(
        req_pool_idx=None,
        inflight_middle_chunks=0,
        kv_committed_len=0,
    )
    req_pool_idx = pool.alloc([req])[0]
    allocator = RecordingAllocator(capacity)
    return pool, req, req_pool_idx, allocator, ChunkCacheStub(allocator)


def alloc_extend(pool, tree_cache, req_pool_idx: int, seq_len: int):
    pool.alloc_aux_for_extend(
        tree_cache=tree_cache,
        req_pool_indices_cpu=torch.tensor([req_pool_idx], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
    )


def test_extend_allocates_at_sparse_boundaries():
    pool, _, req_pool_idx, allocator, tree_cache = make_pool_and_req()

    alloc_extend(pool, tree_cache, req_pool_idx, seq_len=3)
    assert allocator.available_size() == 64

    alloc_extend(pool, tree_cache, req_pool_idx, seq_len=4)
    assert allocator.available_size() == 63

    alloc_extend(pool, tree_cache, req_pool_idx, seq_len=16)
    assert allocator.available_size() == 56


def test_chunk_reuse_only_allocates_new_sparse_slots():
    pool, _, req_pool_idx, allocator, tree_cache = make_pool_and_req()

    alloc_extend(pool, tree_cache, req_pool_idx, seq_len=8)
    assert allocator.available_size() == 61

    alloc_extend(pool, tree_cache, req_pool_idx, seq_len=12)
    assert allocator.available_size() == 59

    alloc_extend(pool, tree_cache, req_pool_idx, seq_len=12)
    assert allocator.available_size() == 59


def test_decode_does_not_duplicate_sparse_slots():
    pool, _, req_pool_idx, allocator, tree_cache = make_pool_and_req()
    alloc_extend(pool, tree_cache, req_pool_idx, seq_len=15)

    pool.alloc_aux_for_decode(
        tree_cache=tree_cache,
        req_pool_indices_cpu=torch.tensor([req_pool_idx], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([15], dtype=torch.int64),
        token_per_req=1,
    )
    available_after_first_decode = allocator.available_size()

    pool.alloc_aux_for_decode(
        tree_cache=tree_cache,
        req_pool_indices_cpu=torch.tensor([req_pool_idx], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([15], dtype=torch.int64),
        token_per_req=1,
    )
    assert available_after_first_decode == 56
    assert allocator.available_size() == available_after_first_decode


def test_partial_failure_rolls_back_and_free_releases_every_slot():
    pool, req, req_pool_idx, allocator, tree_cache = make_pool_and_req(capacity=7)

    with pytest.raises(RuntimeError, match="Out of memory"):
        alloc_extend(pool, tree_cache, req_pool_idx, seq_len=16)

    assert allocator.available_size() == 7

    allocator.capacity = 8
    alloc_extend(pool, tree_cache, req_pool_idx, seq_len=16)
    assert allocator.available_size() == 0

    pool.free(req)
    assert req.req_pool_idx is None
    assert allocator.available_size() == 8
