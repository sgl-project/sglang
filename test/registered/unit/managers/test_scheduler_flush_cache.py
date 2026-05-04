import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import FlushCacheReqInput
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=13, suite="stage-a-test-cpu")


class TestSchedulerFlushCache(unittest.TestCase):
    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._pending_flush = None
        scheduler.send_to_tokenizer = MagicMock()
        scheduler.flush_cache = MagicMock(return_value=True)
        scheduler.is_fully_idle = MagicMock(return_value=False)
        return scheduler

    def test_immediate_flush_no_timeout(self):
        """No timeout → flush immediately regardless of idle state."""
        scheduler = self._new_scheduler()
        scheduler.flush_cache.return_value = False

        output = Scheduler.flush_cache_wrapped(
            scheduler, FlushCacheReqInput(timeout_s=None)
        )

        self.assertFalse(output.success)
        scheduler.flush_cache.assert_called_once()

    def test_immediate_flush_when_idle(self):
        """Positive timeout but already idle → flush immediately."""
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True

        output = Scheduler.flush_cache_wrapped(
            scheduler, FlushCacheReqInput(timeout_s=5.0)
        )

        self.assertTrue(output.success)
        scheduler.flush_cache.assert_called_once()

    def test_defers_when_busy(self):
        """Positive timeout + busy → defers, returns None."""
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=3.0)

        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=10.0):
            output = Scheduler.flush_cache_wrapped(scheduler, req)

        self.assertIsNone(output)
        pending_req, deadline = scheduler._pending_flush
        self.assertIs(pending_req, req)
        self.assertEqual(deadline, 13.0)

    def test_rejects_when_already_pending(self):
        """Any new request is rejected while another is pending."""
        scheduler = self._new_scheduler()
        scheduler._pending_flush = (FlushCacheReqInput(timeout_s=10.0), 999.0)

        for timeout in [None, 5.0]:
            output = Scheduler.flush_cache_wrapped(
                scheduler, FlushCacheReqInput(timeout_s=timeout)
            )
            self.assertFalse(output.success)
            self.assertIn("already in progress", output.message)

        scheduler.flush_cache.assert_not_called()

    def test_pending_flush_completes_on_idle(self):
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True
        req = FlushCacheReqInput(timeout_s=1.0)
        scheduler._pending_flush = (req, 111.0)

        Scheduler._check_pending_flush(scheduler)

        self.assertIsNone(scheduler._pending_flush)
        scheduler.flush_cache.assert_called_once()
        out = scheduler.send_to_tokenizer.send_output.call_args.args[0]
        self.assertTrue(out.success)

    def test_pending_flush_expires_on_timeout(self):
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=1.0)
        scheduler._pending_flush = (req, 99.0)

        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            Scheduler._check_pending_flush(scheduler)

        self.assertIsNone(scheduler._pending_flush)
        scheduler.flush_cache.assert_not_called()
        out = scheduler.send_to_tokenizer.send_output.call_args.args[0]
        self.assertFalse(out.success)

    def test_pending_flush_survives_before_deadline(self):
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=5.0)
        scheduler._pending_flush = (req, 101.0)

        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            Scheduler._check_pending_flush(scheduler)

        self.assertIsNotNone(scheduler._pending_flush)
        scheduler.send_to_tokenizer.send_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
