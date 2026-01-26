"""
"Test force attach/detach setup rollback.

Usage:
    python3 -m pytest test/srt/hicache/test_force_setup_rollback.py -v
"""

import unittest

from sglang.srt.managers.io_struct import (
    AttachHiCacheStorageReqInput,
    DetachHiCacheStorageReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=200, suite="stage-b-test-small-1-gpu")


class _FakeTreeCache:
    def __init__(self, blocked: bool):
        self._blocked = blocked
        self.block_calls = []
        self.attach_called = False
        self.detach_called = False

    def is_storage_io_blocked(self) -> bool:
        return self._blocked

    def set_storage_io_blocked(self, blocked: bool, reason: str = ""):
        self._blocked = blocked
        self.block_calls.append((blocked, reason))
        return True, ""

    def wait_storage_ops_idle(self):
        raise RuntimeError("boom in wait_storage_ops_idle")

    def attach_storage_backend(self, *args, **kwargs):
        self.attach_called = True
        return True, ""

    def detach_storage_backend(self):
        self.detach_called = True
        return True, ""


class TestHiCacheForceSetupRollback(unittest.TestCase):
    def _build_scheduler(self, tree_cache: _FakeTreeCache) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hierarchical_cache = True
        scheduler.tree_cache = tree_cache
        scheduler._is_idle_for_hicache_storage_op = lambda: True
        return scheduler

    def test_force_attach_setup_exception_rolls_back_block(self):
        tree_cache = _FakeTreeCache(blocked=False)
        scheduler = self._build_scheduler(tree_cache)

        req = AttachHiCacheStorageReqInput(
            hicache_storage_backend="file",
            force=True,
        )
        out = scheduler.attach_hicache_storage_wrapped(req)

        self.assertFalse(out.success)
        self.assertFalse(tree_cache.is_storage_io_blocked())
        self.assertEqual(tree_cache.block_calls, [(True, "force_attach"), (False, "")])
        self.assertFalse(tree_cache.attach_called)

    def test_force_detach_setup_exception_rolls_back_block(self):
        tree_cache = _FakeTreeCache(blocked=True)
        scheduler = self._build_scheduler(tree_cache)

        req = DetachHiCacheStorageReqInput(force=True)
        out = scheduler.detach_hicache_storage_wrapped(req)

        self.assertFalse(out.success)
        self.assertTrue(tree_cache.is_storage_io_blocked())
        self.assertEqual(tree_cache.block_calls, [(True, "force_detach"), (True, "")])
        self.assertFalse(tree_cache.detach_called)


if __name__ == "__main__":
    unittest.main()
