"""Guard the MLX auxiliary-state pool's CUDA-style interface parity.

The sglang scheduler reaches into ``req_to_token_pool.mamba_allocator``
in two places that the MLX path did not originally satisfy:

* ``pool_stats_observer._get_mamba_token_info`` calls
  ``mamba_allocator.available_size()``.
* ``scheduler._get_new_batch_prefill_raw`` calls
  ``mamba_allocator.alloc_group_begin(num_reqs)`` /
  ``alloc_group_end()`` (CUDA-style match-prefix pre-allocation).

The MLX pool must expose these names and behave the same as the CUDA
``MambaSlotAllocator`` (release any unused group slots on end).
"""

from __future__ import annotations

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxAuxiliaryStatePoolAllocGroup(unittest.TestCase):
    """``MlxAuxiliaryStatePool.alloc_group_begin/end`` contract."""

    def _make_pool(self, size: int = 16):
        from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
            MlxAuxiliaryStatePool,
        )

        return MlxAuxiliaryStatePool(size=size, device="cpu")

    def test_alloc_group_begin_preallocates_and_iter_serves_singles(self):
        pool = self._make_pool(size=8)
        before = pool.available_size()

        pool.alloc_group_begin(3)
        # Group pre-allocation consumes a contiguous block of 3 slots.
        self.assertEqual(pool.available_size(), before - 3)
        # _alloc_iter yields single-slot tensors that satisfy the per-req
        # alloc path used by the scheduler.
        self.assertIsNotNone(pool._alloc_iter)
        slot0 = next(pool._alloc_iter)
        self.assertEqual(slot0.numel(), 1)

    def test_alloc_group_end_releases_unused_slots(self):
        pool = self._make_pool(size=8)
        pool.alloc_group_begin(4)
        # Consume 1 slot, leave 3 in the iterator.
        next(pool._alloc_iter)
        mid = pool.available_size()

        pool.alloc_group_end()

        # All four pre-allocated slots should now be free again.
        self.assertEqual(pool.available_size(), mid + 3)
        self.assertIsNone(pool._alloc_iter)

    def test_alloc_group_end_is_safe_when_no_group_active(self):
        pool = self._make_pool(size=4)
        before = pool.available_size()

        # Calling alloc_group_end without begin must be a no-op.
        pool.alloc_group_end()

        self.assertEqual(pool.available_size(), before)
        self.assertIsNone(pool._alloc_iter)

    def test_alloc_group_begin_with_zero_reqs_is_noop(self):
        pool = self._make_pool(size=4)
        before = pool.available_size()

        pool.alloc_group_begin(0)
        pool.alloc_group_end()

        self.assertEqual(pool.available_size(), before)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxReqToTokenPoolMambaAllocatorAlias(unittest.TestCase):
    """``req_to_token_pool.mamba_allocator`` must point to the MLX pool."""

    def test_mamba_allocator_is_same_object_as_mamba_pool(self):
        from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
            MlxAuxiliaryStateReqToTokenPool,
        )

        pool = MlxAuxiliaryStateReqToTokenPool(
            size=4,
            max_context_len=16,
            device="cpu",
            enable_memory_saver=False,
            auxiliary_state_size=8,
        )

        self.assertIs(
            pool.mamba_allocator,
            pool.mamba_pool,
            msg=(
                "MlxAuxiliaryStateReqToTokenPool.mamba_allocator must alias "
                "mamba_pool so CUDA-style pool_stats_observer can call "
                "mamba_allocator.available_size()."
            ),
        )


if __name__ == "__main__":
    unittest.main()
