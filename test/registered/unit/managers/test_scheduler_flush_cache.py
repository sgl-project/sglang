import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import FlushCacheReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.flush_wrapper import (
    SchedulerFlushWrapper,
)

register_cpu_ci(est_time=14, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="stage-b-test-cpu-intel")


class TestSchedulerFlushCache(unittest.TestCase):
    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ipc_channels = MagicMock()
        scheduler.flush_cache = MagicMock(return_value=True)
        scheduler.is_fully_idle = MagicMock(return_value=False)
        scheduler.flush_wrapper = SchedulerFlushWrapper(
            flush_cache=scheduler.flush_cache,
            is_fully_idle=scheduler.is_fully_idle,
            ipc_channels=scheduler.ipc_channels,
        )
        return scheduler

    def _new_flushable_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.is_fully_idle = MagicMock(return_value=True)
        scheduler.cur_batch_for_debug = object()
        scheduler.last_batch = object()
        scheduler.tree_cache = MagicMock()
        scheduler.req_to_token_pool = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.grammar_manager = MagicMock()
        scheduler.metrics_reporter = MagicMock()
        scheduler.metrics_reporter.is_stats_logging_rank = False
        scheduler.draft_worker = None
        scheduler.waiting_queue = []
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.reqs = []
        return scheduler

    def test_immediate_flush_no_timeout(self):
        """No timeout → flush immediately regardless of idle state."""
        scheduler = self._new_scheduler()
        scheduler.flush_cache.return_value = False

        output = scheduler.flush_wrapper.handle(FlushCacheReqInput(timeout_s=None))

        self.assertFalse(output.success)
        scheduler.flush_cache.assert_called_once()

    def test_immediate_flush_when_idle(self):
        """Positive timeout but already idle → flush immediately."""
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True

        output = scheduler.flush_wrapper.handle(FlushCacheReqInput(timeout_s=5.0))

        self.assertTrue(output.success)
        scheduler.flush_cache.assert_called_once()

    def test_defers_when_busy(self):
        """Positive timeout + busy → defers, returns None."""
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=3.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=10.0,
        ):
            output = scheduler.flush_wrapper.handle(req)

        self.assertIsNone(output)
        pending_req, deadline = scheduler.flush_wrapper._pending
        self.assertIs(pending_req, req)
        self.assertEqual(deadline, 13.0)

    def test_rejects_when_already_pending(self):
        """Any new request is rejected while another is pending."""
        scheduler = self._new_scheduler()
        scheduler.flush_wrapper._pending = (FlushCacheReqInput(timeout_s=10.0), 999.0)

        for timeout in [None, 5.0]:
            output = scheduler.flush_wrapper.handle(
                FlushCacheReqInput(timeout_s=timeout)
            )
            self.assertFalse(output.success)
            self.assertIn("already in progress", output.message)

        scheduler.flush_cache.assert_not_called()

    def test_pending_flush_completes_on_idle(self):
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True
        req = FlushCacheReqInput(timeout_s=1.0)
        scheduler.flush_wrapper._pending = (req, 111.0)

        scheduler.flush_wrapper.check_pending()

        self.assertIsNone(scheduler.flush_wrapper._pending)
        scheduler.flush_cache.assert_called_once()
        out = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        self.assertTrue(out.success)

    def test_pending_flush_expires_on_timeout(self):
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=1.0)
        scheduler.flush_wrapper._pending = (req, 99.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=100.0,
        ):
            scheduler.flush_wrapper.check_pending()

        self.assertIsNone(scheduler.flush_wrapper._pending)
        scheduler.flush_cache.assert_not_called()
        out = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        self.assertFalse(out.success)

    def test_pending_flush_survives_before_deadline(self):
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=5.0)
        scheduler.flush_wrapper._pending = (req, 101.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=100.0,
        ):
            scheduler.flush_wrapper.check_pending()

        self.assertIsNotNone(scheduler.flush_wrapper._pending)
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_successful_flush_clears_mm_embedding_cache_before_allocator(self):
        scheduler = self._new_flushable_scheduler()
        embedding_cache = MagicMock()
        call_order = []
        embedding_cache.clear.side_effect = lambda: call_order.append("mm_cache")

        with (
            patch(
                "sglang.srt.managers.scheduler.mm_schedule.embedding_cache",
                embedding_cache,
            ),
            patch(
                "sglang.srt.managers.scheduler.current_platform.empty_cache",
                side_effect=lambda: call_order.append("allocator"),
            ),
        ):
            success = scheduler.flush_cache()

        self.assertTrue(success)
        self.assertEqual(call_order, ["mm_cache", "allocator"])
        embedding_cache.clear.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
