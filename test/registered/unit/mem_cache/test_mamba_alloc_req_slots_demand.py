"""CPU-only tests for the mamba eviction demand of ``alloc_req_slots``.

Admitting a prefill batch must evict only what ``HybridReqToTokenPool.alloc``
will actually take. A chunked continuation already holds its active slot and
ping-pong buffer, and a COW match already holds the active slot; charging
every request three slots evicted up to three cached checkpoints per chunk
under a tight pool and starved prefix reuse.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocation import alloc_req_slots, mamba_slots_needed
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _pool(*, extra_buffer=True, lazy=False, overlap=True):
    return SimpleNamespace(
        enable_mamba_extra_buffer=extra_buffer,
        enable_mamba_extra_buffer_lazy=lazy,
        mamba_ping_pong_track_buffer_size=2 if overlap else 1,
    )


def _req(*, holds_mamba: bool, has_ping_pong: bool):
    return SimpleNamespace(
        kv=SimpleNamespace(
            holds_mamba=holds_mamba,
            mamba_ping_pong_track_buffer=(
                torch.tensor([1, 2]) if has_ping_pong else None
            ),
        )
    )


NEW = _req(holds_mamba=False, has_ping_pong=False)
COW = _req(holds_mamba=True, has_ping_pong=False)
CHUNKED = _req(holds_mamba=True, has_ping_pong=True)


class TestMambaSlotsNeeded(unittest.TestCase):
    def test_demand_counts_only_what_alloc_will_take(self):
        pool = _pool()
        self.assertEqual(mamba_slots_needed(req_to_token_pool=pool, reqs=[NEW]), 3)
        self.assertEqual(mamba_slots_needed(req_to_token_pool=pool, reqs=[COW]), 2)
        self.assertEqual(mamba_slots_needed(req_to_token_pool=pool, reqs=[CHUNKED]), 0)


class _RecordingCache:
    def __init__(self):
        self.evict_params = []

    def supports_mamba(self):
        return True

    def evict_for_alloc(self, params: EvictParams):
        self.evict_params.append(params)


def _hybrid_pool(available: int):
    """A HybridReqToTokenPool shell: only the fields alloc_req_slots reads."""
    pool = object.__new__(HybridReqToTokenPool)
    pool.enable_mamba_extra_buffer = True
    pool.enable_mamba_extra_buffer_lazy = False
    pool.mamba_ping_pong_track_buffer_size = 2
    pool.mamba_allocator = SimpleNamespace(schedulable_available_size=lambda: available)
    pool.alloc = lambda reqs: list(range(1, len(reqs) + 1))
    return pool


class TestAllocReqSlotsEviction(unittest.TestCase):
    def test_evicts_only_the_shortfall(self):
        cache = _RecordingCache()
        # a chunked continuation holds everything already: nothing to evict
        alloc_req_slots(_hybrid_pool(available=0), [CHUNKED], cache)
        self.assertEqual(cache.evict_params, [])
        # a COW-matched request needs its two ping-pong slots; one is free
        alloc_req_slots(_hybrid_pool(available=1), [COW], cache)
        self.assertEqual(cache.evict_params, [EvictParams(num_tokens=0, mamba_num=1)])


if __name__ == "__main__":
    unittest.main()
