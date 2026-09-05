"""CPU coverage for retained lifecycle paths not yet present on main."""

import asyncio
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.encoder.receiver import MMReceiverHTTP
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
    def make_receiver(self):
        receiver = MMReceiverHTTP.__new__(MMReceiverHTTP)
        receiver.encode_urls = ["http://encoder"]
        receiver.context = object()
        receiver.host = "127.0.0.1"
        receiver.recv_timeout = 60
        receiver._extract_url_data = MagicMock(return_value=[{"modality": "image"}])
        started = [asyncio.Event(), asyncio.Event()]
        finished = [asyncio.Event(), asyncio.Event()]

        async def pending(index):
            started[index].set()
            try:
                await asyncio.Event().wait()
            finally:
                finished[index].set()

        receiver.encode = lambda *args, **kwargs: pending(0)
        receiver._recv_mm_data = lambda *args, **kwargs: pending(1)
        socket = MagicMock()
        socket.close.side_effect = lambda **kwargs: self.assertTrue(
            all(event.is_set() for event in finished)
        )
        return receiver, socket, started, finished

    async def test_timeout_joins_both_tasks_before_closing_socket(self):
        receiver, socket, _, finished = self.make_receiver()
        receiver.recv_timeout = 0.01
        with patch(
            "sglang.srt.disaggregation.encoder.receiver.get_zmq_socket_on_host",
            return_value=(12345, socket),
        ):
            result = await receiver.recv_mm_data(object(), object(), "prompt")
        self.assertIsNone(result)
        self.assertTrue(all(event.is_set() for event in finished))
        socket.close.assert_called_once_with(linger=0)

    async def test_cancellation_propagates_after_join_and_socket_close(self):
        receiver, socket, started, finished = self.make_receiver()
        with patch(
            "sglang.srt.disaggregation.encoder.receiver.get_zmq_socket_on_host",
            return_value=(12345, socket),
        ):
            task = asyncio.create_task(
                receiver.recv_mm_data(object(), object(), "prompt")
            )
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in started)), timeout=1
            )
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(task.cancelled())
        self.assertTrue(all(event.is_set() for event in finished))
        socket.close.assert_called_once_with(linger=0)


if __name__ == "__main__":
    unittest.main()
