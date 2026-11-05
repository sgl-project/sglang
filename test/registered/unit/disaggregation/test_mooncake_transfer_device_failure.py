"""Worker initialization failures must reach the thread polling the sender."""

import threading
import unittest
from concurrent.futures import BrokenExecutor
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.mooncake import conn
from sglang.srt.platforms import current_platform
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class RoomStates(dict):
    def collect(self, room):
        return [self[room], self[room]]


class TestMooncakeTransferDeviceFailure(unittest.TestCase):
    @contextmanager
    def manager(self, set_device):
        # Keep the real threads, lazy executor initialization and polling. The
        # transfer loop stops on its normal sentinel; no native engine is used.
        def init_common(manager, args, mode, *unused):
            manager.kv_args = args
            manager.disaggregation_mode = mode
            manager.request_status = RoomStates({7: conn.KVPoll.Transferring})

        worker_errors = []
        with ExitStack() as stack:
            for name, value in {
                "SGLANG_DISAGG_STAGING_BUFFER": False,
                "SGLANG_ENABLE_FAILED_SESSION_PROBE": False,
                "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE": 1,
                "SGLANG_DISAGGREGATION_QUEUE_SIZE": 1,
            }.items():
                stack.enter_context(
                    patch.object(getattr(conn.envs, name), "get", return_value=value)
                )
            stack.enter_context(
                patch.object(conn.CommonKVManager, "__init__", init_common)
            )
            for name in (
                "init_engine",
                "register_buffer_to_engine",
                "start_prefill_thread",
            ):
                stack.enter_context(patch.object(conn.MooncakeKVManager, name))
            stack.enter_context(
                patch.object(
                    conn,
                    "check_mooncake_custom_mem_pool_enabled",
                    return_value=(False, None),
                )
            )
            stack.enter_context(
                patch.object(
                    conn,
                    "get_observability",
                    return_value=SimpleNamespace(enable_trace=False),
                )
            )
            stack.enter_context(
                patch.object(
                    current_platform, "get_device", side_effect=lambda rank: rank
                )
            )
            stack.enter_context(
                patch.object(current_platform, "set_device", set_device)
            )
            stack.enter_context(
                patch.object(
                    threading,
                    "excepthook",
                    lambda args: worker_errors.append(args.exc_value),
                )
            )
            manager = conn.MooncakeKVManager(
                SimpleNamespace(gpu_id=3), conn.DisaggregationMode.PREFILL, None
            )
            try:
                manager.transfer_queues[0].put(None)
                manager._worker_threads[0].join(timeout=5)
                self.assertFalse(manager._worker_threads[0].is_alive())
                yield manager, worker_errors
            finally:
                manager.executors[0].shutdown(wait=True, cancel_futures=True)

    def check_sender(self, manager, error=None):
        sender = conn.MooncakeKVSender.__new__(conn.MooncakeKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = 7
        sender.conclude_state = None
        for poll in (sender.poll, sender.poll_pp_consensus):
            if error is None:
                self.assertEqual(poll(), conn.KVPoll.Transferring)
            else:
                with self.assertRaisesRegex(
                    RuntimeError, "could not bind device 3"
                ) as caught:
                    poll()
                self.assertIs(caught.exception.__cause__, error)

    def test_worker_failure_reaches_both_sender_polls_and_preserves_first_cause(self):
        first = RuntimeError("worker binding failed")
        bind = Mock(side_effect=first)
        with self.manager(bind) as (manager, worker_errors):
            self.assertEqual(worker_errors, [first])
            bind.assert_called_once_with(3)
            self.check_sender(manager, first)
            # A second failure breaks the lazy pool without replacing the first cause.
            bind.side_effect = RuntimeError("executor binding failed")
            payload = Mock()
            with (
                self.assertLogs("concurrent.futures", level="CRITICAL"),
                self.assertRaises(BrokenExecutor),
            ):
                manager.executors[0].submit(payload).result(timeout=5)
            payload.assert_not_called()
            self.check_sender(manager, first)

    def test_pool_only_failure_reaches_both_sender_polls(self):
        bind = Mock()
        with self.manager(bind) as (manager, worker_errors):
            self.assertFalse(worker_errors)
            self.check_sender(manager)
            error = RuntimeError("executor binding failed")
            bind.side_effect = error
            payload = Mock()
            with (
                self.assertLogs("concurrent.futures", level="CRITICAL"),
                self.assertRaises(BrokenExecutor),
            ):
                manager.executors[0].submit(payload).result(timeout=5)
            payload.assert_not_called()
            self.check_sender(manager, error)


if __name__ == "__main__":
    unittest.main()
