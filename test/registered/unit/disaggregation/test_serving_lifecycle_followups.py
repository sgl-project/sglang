"""CPU coverage for retained lifecycle paths not yet present on main."""

import asyncio
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.encoder.receiver import MMReceiverHTTP
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSchedulerLifecycle(unittest.TestCase):
    def test_empty_prefill_only_batch_clears_flags_before_next_admission(self):
        scheduler = MagicMock(spec=Scheduler)
        scheduler.scheduler_stage_metrics = None
        scheduler.enable_fpm = False
        scheduler.dllm_config = None
        scheduler.chunked_req = None
        scheduler.enable_hisparse = False
        scheduler.require_mlp_sync = False
        scheduler._should_defer_prefill.return_value = True
        scheduler.dp_attn_adapter = MagicMock()
        scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = (
            lambda batch, **kwargs: batch
        )
        scheduler.dp_attn_adapter.maybe_convert_decode_to_extend.side_effect = (
            lambda batch: batch
        )
        scheduler.ngram_embedding_manager = MagicMock()
        scheduler.ngram_embedding_manager.prepare_for_forward.side_effect = (
            lambda batch, **kwargs: batch
        )
        finished_req = MagicMock()
        finished_req.finished.return_value = True
        running = ScheduleBatch(
            reqs=[finished_req], batch_is_full=True, is_prefill_only=True
        )

        plan = Scheduler.get_next_batch_to_run(scheduler, running, None)

        self.assertTrue(plan.running_batch.is_empty())
        self.assertFalse(plan.running_batch.batch_is_full)
        self.assertFalse(plan.running_batch.is_prefill_only)
        self.assertIsNone(plan.batch_to_run)
        scheduler.update_running_batch.assert_not_called()

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
