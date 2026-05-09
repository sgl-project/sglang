import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestSchedulerCache(unittest.TestCase):
    def _new_idle_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.is_fully_idle = MagicMock(return_value=True)
        scheduler.cur_batch = MagicMock()
        scheduler.last_batch = MagicMock()
        scheduler.tree_cache = MagicMock()
        scheduler.req_to_token_pool = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.grammar_manager = MagicMock()
        scheduler.reset_metrics = MagicMock()
        scheduler.draft_worker = None
        scheduler.device_module = MagicMock()
        return scheduler

    def test_flush_cache_empties_device_allocator_by_default(self):
        scheduler = self._new_idle_scheduler()

        with patch("sglang.srt.managers.scheduler.empty_device_cache") as mock_empty:
            self.assertTrue(scheduler.flush_cache())

        mock_empty.assert_called_once_with(scheduler.device_module)

    def test_flush_cache_can_skip_device_allocator_empty(self):
        scheduler = self._new_idle_scheduler()

        with patch("sglang.srt.managers.scheduler.empty_device_cache") as mock_empty:
            self.assertTrue(scheduler.flush_cache(empty_cache=False))

        mock_empty.assert_not_called()


if __name__ == "__main__":
    unittest.main()
