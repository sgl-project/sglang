import unittest
from collections import deque
from unittest.mock import MagicMock, patch

from sglang.srt.managers.io_struct import FlushCacheReqInput, FlushCacheReqOutput
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")


class TestSchedulerFlushCache(unittest.TestCase):
    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._pending_flush = deque()
        scheduler.send_to_tokenizer = MagicMock()
        scheduler.flush_cache = MagicMock(return_value=True)
        scheduler.is_fully_idle = MagicMock(return_value=False)
        return scheduler

    def test_flush_cache_wrapped_non_positive_timeout_immediate(self):
        scheduler = self._new_scheduler()
        scheduler.flush_cache.return_value = False

        output = Scheduler.flush_cache_wrapped(
            scheduler, FlushCacheReqInput(timeout_s=None)
        )

        self.assertIsInstance(output, FlushCacheReqOutput)
        self.assertFalse(output.success)
        scheduler.flush_cache.assert_called_once()
        self.assertEqual(len(scheduler._pending_flush), 0)

    def test_flush_cache_wrapped_positive_timeout_idle_immediate(self):
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True

        output = Scheduler.flush_cache_wrapped(
            scheduler, FlushCacheReqInput(timeout_s=5.0)
        )

        self.assertIsInstance(output, FlushCacheReqOutput)
        self.assertTrue(output.success)
        scheduler.flush_cache.assert_called_once()
        self.assertEqual(len(scheduler._pending_flush), 0)

    def test_flush_cache_wrapped_positive_timeout_busy_enqueues(self):
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=3.0)

        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=10.0):
            output = Scheduler.flush_cache_wrapped(scheduler, req)

        self.assertIsNone(output)
        self.assertEqual(len(scheduler._pending_flush), 1)
        pending_req, deadline = scheduler._pending_flush[0]
        self.assertIs(pending_req, req)
        self.assertEqual(deadline, 13.0)
        scheduler.flush_cache.assert_not_called()

    def test_check_pending_flush_idle_flushes_all(self):
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True

        req1 = FlushCacheReqInput(timeout_s=1.0)
        req2 = FlushCacheReqInput(timeout_s=2.0)
        scheduler._pending_flush = deque([(req1, 111.0), (req2, 222.0)])

        Scheduler._check_pending_flush(scheduler)

        scheduler.flush_cache.assert_called_once()
        self.assertEqual(len(scheduler._pending_flush), 0)
        calls = scheduler.send_to_tokenizer.send_output.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0].args[0].success)
        self.assertIs(calls[0].args[1], req1)
        self.assertTrue(calls[1].args[0].success)
        self.assertIs(calls[1].args[1], req2)

    def test_check_pending_flush_busy_expires_only_timed_out(self):
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = False

        expired_req = FlushCacheReqInput(timeout_s=1.0)
        alive_req = FlushCacheReqInput(timeout_s=5.0)
        scheduler._pending_flush = deque([(expired_req, 99.0), (alive_req, 101.0)])

        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            Scheduler._check_pending_flush(scheduler)

        scheduler.flush_cache.assert_not_called()
        self.assertEqual(len(scheduler._pending_flush), 1)
        pending_req, deadline = scheduler._pending_flush[0]
        self.assertIs(pending_req, alive_req)
        self.assertEqual(deadline, 101.0)

        calls = scheduler.send_to_tokenizer.send_output.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0].args[0].success)
        self.assertIs(calls[0].args[1], expired_req)


if __name__ == "__main__":
    unittest.main()
