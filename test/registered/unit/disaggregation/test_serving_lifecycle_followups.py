"""CPU coverage for retained lifecycle paths not yet present on main."""

import asyncio
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.encoder.receiver import MMReceiverBase
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSchedulerLifecycle(unittest.TestCase):
    def test_sub_page_chunk_budget_stops_admission(self):
        adder = SimpleNamespace(page_size=64, rem_chunk_tokens=None)
        for remaining, exhausted in (
            (None, False),
            (0, True),
            (63, True),
            (64, False),
            (65, False),
        ):
            with self.subTest(remaining=remaining):
                adder.rem_chunk_tokens = remaining
                self.assertEqual(PrefillAdder.chunk_budget_exhausted(adder), exhausted)

    def test_all_pending_health_signals_are_returned(self):
        sender = MagicMock()
        scheduler = SimpleNamespace(
            return_health_check_ipcs=deque(["worker-0", "worker-1", "worker-2"]),
            ipc_channels=SimpleNamespace(send_to_tokenizer=sender),
        )
        Scheduler.maybe_send_health_check_signal(scheduler)
        self.assertFalse(scheduler.return_health_check_ipcs)
        self.assertEqual(
            [
                call.args[0].http_worker_ipc
                for call in sender.send_output.call_args_list
            ],
            ["worker-0", "worker-1", "worker-2"],
        )


class TestEncoderCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_joins_task_then_releases_buffer(self):
        started = asyncio.Event()
        finished = asyncio.Event()

        async def encode():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        task = asyncio.create_task(encode())
        await started.wait()
        cleanup = MagicMock(side_effect=lambda _: self.assertTrue(finished.is_set()))
        receiver = SimpleNamespace(_cleanup_mooncake_buffer=cleanup)
        await MMReceiverBase._abort_encode_and_cleanup(receiver, task, "request-1")
        self.assertTrue(task.cancelled())
        cleanup.assert_called_once_with("request-1")

    async def test_missing_task_still_releases_request_buffer(self):
        receiver = SimpleNamespace(_cleanup_mooncake_buffer=MagicMock())
        await MMReceiverBase._abort_encode_and_cleanup(receiver, None, "request-2")
        receiver._cleanup_mooncake_buffer.assert_called_once_with("request-2")
        receiver._cleanup_mooncake_buffer.reset_mock()
        await MMReceiverBase._abort_encode_and_cleanup(receiver, None, None)
        receiver._cleanup_mooncake_buffer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
