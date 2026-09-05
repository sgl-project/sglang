import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.minicpm.cache import (
    attach_compressed_cache,
)
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class RecordingAllocator(BaseTokenToKVPoolAllocator):
    """Single-pool double. Subclassing the base routes free_full / free_segment /
    free_segments into free(), so a new free API cannot slip past the recorder."""

    def __init__(self, capacity: int):
        super().__init__(
            size=capacity,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.capacity = capacity
        self.next_slot = 1
        self.live: set[int] = set()

    def alloc(self, size: int):
        if size > self.available_size():
            return None
        slots = torch.arange(self.next_slot, self.next_slot + size, dtype=torch.int64)
        self.next_slot += size
        self.live.update(slots.tolist())
        return slots

    def free(self, slots: torch.Tensor):
        self.live.difference_update(slots.tolist())

    def available_size(self):
        return self.capacity - len(self.live)

    def get_all_free_pages(self):
        return self.free_pages

    def clear(self):
        self.next_slot = 1
        self.live.clear()


def make_pool_and_req(capacity: int = 64):
    allocator = RecordingAllocator(capacity)
    pool = ReqToTokenPool(
        size=2,
        max_context_len=64,
        device="cpu",
        enable_memory_saver=False,
    )
    attach_compressed_cache(
        pool,
        allocator,
        kernel_size=4,
        kernel_stride=2,
        enable_memory_saver=False,
    )
    req = SimpleNamespace(
        inflight_middle_chunks=0,
        kv=ReqKvInfo(),
    )
    req_pool_idx = pool.alloc([req])[0]
    return pool, req, req_pool_idx, allocator


def alloc_extend(pool, req_pool_idx: int, seq_len: int):
    pool.alloc_aux_to_lengths(
        req_pool_indices_cpu=torch.tensor([req_pool_idx], dtype=torch.int64),
        target_seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
    )


def test_extend_allocates_at_sparse_boundaries():
    pool, _, req_pool_idx, allocator = make_pool_and_req()
    cache = pool._aux_cache

    alloc_extend(pool, req_pool_idx, seq_len=3)
    assert allocator.available_size() == 39
    assert len(cache.free_slots) == 25

    alloc_extend(pool, req_pool_idx, seq_len=4)
    assert allocator.available_size() == 39
    assert len(cache.free_slots) == 24

    alloc_extend(pool, req_pool_idx, seq_len=16)
    assert allocator.available_size() == 39
    assert len(cache.free_slots) == 17


def test_chunk_reuse_only_allocates_new_sparse_slots():
    pool, _, req_pool_idx, _ = make_pool_and_req()
    cache = pool._aux_cache

    alloc_extend(pool, req_pool_idx, seq_len=8)
    assert len(cache.free_slots) == 22

    alloc_extend(pool, req_pool_idx, seq_len=12)
    assert len(cache.free_slots) == 20

    alloc_extend(pool, req_pool_idx, seq_len=12)
    assert len(cache.free_slots) == 20


def test_decode_does_not_duplicate_sparse_slots():
    """Retrying the same decode position must not allocate duplicate cache slots."""
    pool, _, req_pool_idx, _ = make_pool_and_req()
    cache = pool._aux_cache
    alloc_extend(pool, req_pool_idx, seq_len=15)

    pool.alloc_aux_to_lengths(
        req_pool_indices_cpu=torch.tensor([req_pool_idx], dtype=torch.int64),
        target_seq_lens_cpu=torch.tensor([16], dtype=torch.int64),
    )
    available_after_first_decode = len(cache.free_slots)

    pool.alloc_aux_to_lengths(
        req_pool_indices_cpu=torch.tensor([req_pool_idx], dtype=torch.int64),
        target_seq_lens_cpu=torch.tensor([16], dtype=torch.int64),
    )
    assert available_after_first_decode == 17
    assert len(cache.free_slots) == available_after_first_decode


def test_reserve_leaves_only_dense_capacity_visible():
    allocator = RecordingAllocator(capacity=69)
    pool = ReqToTokenPool(
        size=2,
        max_context_len=64,
        device="cpu",
        enable_memory_saver=False,
    )
    attach_compressed_cache(
        pool,
        allocator,
        kernel_size=32,
        kernel_stride=16,
        enable_memory_saver=False,
    )

    assert allocator.available_size() == 64
    assert len(pool._aux_cache.reserved_slots) == 5
    assert pool.schedulable_token_capacity(69) == 64


def test_reserved_slots_are_excluded_from_full_pool_invariant():
    pool, _, _, allocator = make_pool_and_req(capacity=69)
    checker = SchedulerInvariantChecker(
        is_hybrid_swa=False,
        is_hybrid_ssm=True,
        disaggregation_mode=None,
        page_size=1,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=64,
        tree_cache=SimpleNamespace(
            supports_mamba=lambda: False,
            protected_size=lambda: 0,
        ),
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=pool,
        pool_stats_observer=SimpleNamespace(session_held_tokens=lambda: 0),
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
        scheduler_stage_metrics=None,
    )

    leak, message = checker._check_full_pool(
        SimpleNamespace(
            full_available_size=allocator.available_size(), full_evictable_size=0
        )
    )

    assert not leak, message


def test_hybrid_pool_stats_exclude_reserved_slots():
    pool, _, _, allocator = make_pool_and_req(capacity=69)
    pool.mamba_allocator = SimpleNamespace(available_size=lambda: 1)
    pool.mamba_pool = SimpleNamespace(size=1)
    observer = SchedulerPoolStatsObserver(
        tree_cache=SimpleNamespace(supports_mamba=lambda: False),
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=pool,
        session_controller=None,
        hisparse_coordinator=None,
        is_hybrid_swa=False,
        is_hybrid_ssm=True,
        enable_hisparse=False,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=42,
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
    )

    stats = observer._get_mamba_token_info()

    assert stats.full_num_used == 0
    assert stats.full_token_usage == 0


def test_streaming_session_release_frees_compressed_slots():
    pool, _, req_pool_idx, allocator = make_pool_and_req()
    alloc_extend(pool, req_pool_idx, seq_len=16)
    dense_slots = allocator.alloc(16)
    pool.req_to_token[req_pool_idx, :16] = dense_slots.to(torch.int32)
    compressed_cache = pool._aux_cache
    assert len(compressed_cache.free_slots) < len(compressed_cache.reserved_slots)

    session = StreamingSession(
        SimpleNamespace(
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        )
    )
    session.slots["session-a"] = SessionSlot(
        kv=ReqKvInfo(req_pool_idx=req_pool_idx, kv_allocated_len=16),
    )

    session.release_session("session-a")

    assert req_pool_idx in pool.free_slots
    assert len(compressed_cache.free_slots) == len(compressed_cache.reserved_slots)


def test_mamba_leak_diagnostic_does_not_report_reserved_slots():
    pool, _, _, allocator = make_pool_and_req(capacity=69)
    allocator.free_pages = torch.arange(6, 70, dtype=torch.int64)
    pool.mamba_pool = SimpleNamespace(size=1)
    pool.mamba_allocator = SimpleNamespace(
        size=1,
        free_slots=torch.empty(0, dtype=torch.int64),
    )
    checker = SchedulerInvariantChecker(
        is_hybrid_swa=False,
        is_hybrid_ssm=True,
        disaggregation_mode=None,
        page_size=1,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=64,
        tree_cache=SimpleNamespace(
            mamba_protected_size=lambda: 0,
            all_values_flatten=lambda: torch.empty(0, dtype=torch.int64),
            all_mamba_values_flatten=lambda: torch.empty(0, dtype=torch.int64),
        ),
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=pool,
        pool_stats_observer=SimpleNamespace(
            session_held_mamba_slots=lambda: 0,
        ),
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
        scheduler_stage_metrics=None,
    )

    leak, message = checker._check_mamba_pool(
        SimpleNamespace(mamba_available_size=0, mamba_evictable_size=0)
    )

    assert leak
    assert "leaked_full_pages" not in message
    assert "leaked_mamba_pages={1}" in message


def test_partial_failure_rolls_back_and_free_releases_every_slot():
    """A failed cache-level allocation must release slots allocated for other levels."""
    pool, req, req_pool_idx, allocator = make_pool_and_req(capacity=18)
    cache = pool._aux_cache

    with pytest.raises(RuntimeError, match="out of reserved slots"):
        alloc_extend(pool, req_pool_idx, seq_len=16)

    assert allocator.available_size() == 11
    assert len(cache.free_slots) == 7

    allocator.capacity = 20
    allocator.clear()
    pool.reset_aux_cache_allocator()
    alloc_extend(pool, req_pool_idx, seq_len=16)
    assert allocator.available_size() == 12
    assert len(cache.free_slots) == 0

    pool.free(req)
    assert req.kv.req_pool_idx is None
    assert allocator.available_size() == 12
    assert len(cache.free_slots) == 8


def test_allocator_reset_rebuilds_reserve():
    pool, _, _, allocator = make_pool_and_req()

    allocator.clear()
    assert allocator.available_size() == 64

    pool.reset_aux_cache_allocator()
    assert allocator.available_size() == 39
    assert len(pool._aux_cache.free_slots) == 25


def test_attach_compressed_cache_is_idempotent():
    allocator = RecordingAllocator(capacity=64)
    pool = ReqToTokenPool(
        size=2,
        max_context_len=64,
        device="cpu",
        enable_memory_saver=False,
    )
    attach_compressed_cache(
        pool,
        allocator,
        kernel_size=4,
        kernel_stride=2,
        enable_memory_saver=False,
    )
    cache = pool._aux_cache
    k1_table = pool.req_to_sparse_k1_token

    attach_compressed_cache(
        pool,
        allocator,
        kernel_size=4,
        kernel_stride=2,
        enable_memory_saver=False,
    )

    assert pool._aux_cache is cache
    assert pool.req_to_sparse_k1_token is k1_table


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
