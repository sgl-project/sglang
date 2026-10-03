import asyncio
import enum
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.srt.managers.tokenizer_manager import (
    ServerStatus,
    SignalHandler,
    TokenizerManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _ChunkStatus(enum.Enum):
    Ready = 1
    Pending = 2
    Closed = 3


class _RecordingCallback:
    def __init__(self):
        self.calls = []

    def __call__(self, payload, *, finished=False, error=None):
        self.calls.append((payload, finished, error))
        return _ChunkStatus.Ready


class _FakeTokenizerManager:
    def __init__(self, responses):
        self.responses = responses

    def generate_request(self, obj, request=None):
        async def generate():
            for response in self.responses:
                yield response

        return generate()


def _make_runtime_handle(responses):
    handle = RuntimeHandle.__new__(RuntimeHandle)
    handle.tokenizer_manager = _FakeTokenizerManager(responses)
    return handle


class TestNativeGrpcParallelResponses(CustomTestCase):
    def test_non_streaming_returns_every_choice_before_finishing(self):
        callback = _RecordingCallback()
        responses = [
            [
                {"output_ids": [1], "meta_info": {"id": "choice-0"}},
                {"output_ids": [2], "meta_info": {"id": "choice-1"}},
            ]
        ]
        handle = _make_runtime_handle(responses)
        obj = SimpleNamespace(rid="logical", batch_size=1, parallel_sample_num=2)

        asyncio.run(
            handle._run_generate(
                obj,
                callback,
                stream=False,
                request=None,
            )
        )

        self.assertEqual([call[0]["output_ids"] for call in callback.calls], [[1], [2]])
        self.assertEqual([call[1] for call in callback.calls], [False, True])

    def test_streaming_first_finished_choice_is_not_batch_terminal(self):
        callback = _RecordingCallback()
        responses = [
            {
                "index": 0,
                "output_ids": [1],
                "meta_info": {"id": "choice-0", "finish_reason": None},
            },
            {
                "index": 0,
                "output_ids": [2],
                "meta_info": {
                    "id": "choice-0",
                    "finish_reason": {"type": "stop"},
                },
            },
            {
                "index": 1,
                "output_ids": [3],
                "meta_info": {
                    "id": "choice-1",
                    "finish_reason": {"type": "stop"},
                },
            },
        ]
        handle = _make_runtime_handle(responses)
        obj = SimpleNamespace(rid="logical", sampling_params={"n": 2})

        asyncio.run(
            handle._run_generate(
                obj,
                callback,
                stream=True,
                request=None,
            )
        )

        self.assertEqual(
            [call[0]["output_ids"] for call in callback.calls],
            [[1], [2], [3]],
        )
        self.assertEqual([call[1] for call in callback.calls], [False, False, True])


class TestEngineStateNotifications(CustomTestCase):
    def setUp(self):
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager._engine_state_changed_callback = None
        self.manager._server_status = ServerStatus.Starting
        self.manager._gracefully_exit = False
        self.manager._is_pause = False
        self.notifications = 0
        self.manager.set_engine_state_changed_callback(self._notify)

    def _notify(self):
        self.notifications += 1

    def test_observable_state_notifies_only_on_changes(self):
        self.manager.is_pause = False
        self.manager.server_status = ServerStatus.Starting
        self.manager.gracefully_exit = False
        self.assertEqual(self.notifications, 0)

        self.manager.is_pause = True
        self.manager.server_status = ServerStatus.Up
        self.manager.gracefully_exit = True
        self.assertEqual(self.notifications, 3)

    def test_graceful_exit_notifies_and_changes_computed_health(self):
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle.tokenizer_manager = self.manager
        self.manager.server_status = ServerStatus.Up
        self.notifications = 0

        self.assertTrue(handle.health_check())
        self.manager.gracefully_exit = True

        self.assertEqual(self.notifications, 1)
        self.assertFalse(handle.health_check())


class TestGrpcShutdown(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_from_rpc_thread_uses_sigterm_handler(self):
        active_requests = {"in-flight": object()}
        manager = SimpleNamespace(
            gracefully_exit=False,
            signal_handler_class=SignalHandler,
            rid_to_state=active_requests,
        )
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle.tokenizer_manager = manager
        handle._event_loop = asyncio.get_running_loop()

        # Rust calls the bridge from a different thread. Repeated calls must
        # leave draining and cleanup to the existing shutdown watchdog.
        await asyncio.to_thread(handle.shutdown)
        await asyncio.to_thread(handle.shutdown)
        await asyncio.sleep(0)

        self.assertTrue(manager.gracefully_exit)
        self.assertIs(manager.rid_to_state, active_requests)
        self.assertIn("in-flight", active_requests)

    async def test_shutdown_reports_closed_event_loop(self):
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle.tokenizer_manager = SimpleNamespace(signal_handler_class=SignalHandler)
        handle._event_loop = asyncio.new_event_loop()
        handle._event_loop.close()

        with self.assertRaises(RuntimeError):
            handle.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
