import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.io_struct import ClearHiCacheReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHybridCacheControllerClearStorage(unittest.TestCase):
    def test_only_explicit_false_is_failure(self):
        for clear_result, expected in ((False, False), (True, True), (None, True)):
            with self.subTest(clear_result=clear_result):
                controller = HybridCacheController.__new__(HybridCacheController)
                controller.enable_storage = True
                controller.storage_backend = SimpleNamespace(
                    clear=MagicMock(return_value=clear_result)
                )

                result = controller.clear_storage_backend()

                self.assertIs(result, expected)
                controller.storage_backend.clear.assert_called_once_with()


class TestSchedulerClearHiCacheStorage(unittest.TestCase):
    def test_only_explicit_false_is_failure(self):
        for clear_result, expected in ((False, False), (True, True), (None, True)):
            with self.subTest(clear_result=clear_result):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.enable_hierarchical_cache = True
                scheduler.enable_lmcache = False
                scheduler.tree_cache = SimpleNamespace(
                    clear_storage_backend=MagicMock(return_value=clear_result)
                )

                result = scheduler.clear_hicache_storage_wrapped(ClearHiCacheReqInput())

                self.assertIs(result.success, expected)
                scheduler.tree_cache.clear_storage_backend.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
