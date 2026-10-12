"""A weight update that flushes the cache must drop L3 too.

`flush_cache` resets the tree and the host tier but never touches the storage
backend, and a storage key carries no weight version, so a prefetch after the
update would serve KV computed with the previous weights. Pure CPU test.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _TreeCacheStub:
    def __init__(self, cleared_ok=True):
        self.clear_calls = 0
        self._cleared_ok = cleared_ok

    def clear_storage_backend(self) -> bool:
        self.clear_calls += 1
        return self._cleared_ok


def _manager(*, hierarchical=True, lmcache=False, cleared_ok=True, scheduler=True):
    tree_cache = _TreeCacheStub(cleared_ok)
    manager = SchedulerWeightUpdaterManager.__new__(SchedulerWeightUpdaterManager)
    object.__setattr__(manager, "flush_cache", lambda **kwargs: True)
    object.__setattr__(
        manager,
        "scheduler",
        (
            SimpleNamespace(
                enable_hierarchical_cache=hierarchical,
                enable_lmcache=lmcache,
                tree_cache=tree_cache,
            )
            if scheduler
            else None
        ),
    )
    return manager, tree_cache


def _req(flush_cache=True):
    return SimpleNamespace(flush_cache=flush_cache, torch_empty_cache=False)


class TestWeightUpdateClearsStorage(CustomTestCase):
    def test_flush_also_clears_the_storage_backend(self):
        manager, tree_cache = _manager()
        manager.flush_cache_after_weight_update(_req())
        self.assertEqual(
            tree_cache.clear_calls,
            1,
            "L3 survives flush_cache and its keys carry no weight version, so "
            "the stale pages must be dropped explicitly",
        )

    def test_lmcache_is_cleared_too(self):
        manager, tree_cache = _manager(hierarchical=False, lmcache=True)
        manager.flush_cache_after_weight_update(_req())
        self.assertEqual(tree_cache.clear_calls, 1)

    def test_no_flush_requested_leaves_storage_alone(self):
        # The caller owns the decision: without a flush the device and host
        # tiers keep their stale pages too.
        manager, tree_cache = _manager()
        manager.flush_cache_after_weight_update(_req(flush_cache=False))
        self.assertEqual(tree_cache.clear_calls, 0)

    def test_no_storage_tier_is_a_no_op(self):
        manager, tree_cache = _manager(hierarchical=False, lmcache=False)
        manager.flush_cache_after_weight_update(_req())
        self.assertEqual(tree_cache.clear_calls, 0)

    def test_a_failed_clear_does_not_raise(self):
        # The backend may be detached or unreachable; the update still completes.
        manager, tree_cache = _manager(cleared_ok=False)
        manager.flush_cache_after_weight_update(_req())
        self.assertEqual(tree_cache.clear_calls, 1)

    def test_no_scheduler_is_a_no_op(self):
        manager, _ = _manager(scheduler=False)
        manager.flush_cache_after_weight_update(_req())


if __name__ == "__main__":
    unittest.main()
