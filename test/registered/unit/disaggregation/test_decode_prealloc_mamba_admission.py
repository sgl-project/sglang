"""Decode prealloc must evict cached mamba checkpoints before taking a slot."""

import unittest
from unittest.mock import Mock

from sglang.srt.mem_cache.allocation import alloc_req_slots
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _hybrid_pool(mamba_available: int) -> HybridReqToTokenPool:
    # Bypass __init__: the pool's constructor allocates device tensors.
    pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
    pool.enable_mamba_extra_buffer_lazy = False
    pool.mamba_allocator = Mock(
        schedulable_available_size=Mock(return_value=mamba_available)
    )
    pool.alloc = Mock(return_value=[0])
    return pool


class TestDecodePreallocMambaAdmission(CustomTestCase):
    def test_exhausted_mamba_pool_evicts_checkpoints_first(self):
        # Decode-side radix cache on an SSM model: cached checkpoints hold
        # every mamba slot, so a fresh request must evict before alloc.
        pool = _hybrid_pool(mamba_available=0)
        tree_cache = Mock(supports_mamba=Mock(return_value=True))

        indices = alloc_req_slots(pool, [Mock()], tree_cache)

        self.assertEqual(indices, [0])
        (params,), _ = tree_cache.evict_for_alloc.call_args
        self.assertIsInstance(params, EvictParams)
        self.assertEqual(params.num_tokens, 0)
        self.assertGreater(params.mamba_num, 0)
        pool.alloc.assert_called_once()

    def test_free_mamba_slots_skip_eviction(self):
        pool = _hybrid_pool(mamba_available=64)
        tree_cache = Mock(supports_mamba=Mock(return_value=True))

        alloc_req_slots(pool, [Mock()], tree_cache)

        tree_cache.evict_for_alloc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
