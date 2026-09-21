import asyncio
import enum
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


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

    def test_runtime_handle_registers_callback_with_manager(self):
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle.tokenizer_manager = self.manager
        callback = object()

        handle.set_engine_state_changed_callback(callback)

        self.assertIs(self.manager._engine_state_changed_callback, callback)

    def test_graceful_exit_notifies_and_changes_computed_health(self):
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle.tokenizer_manager = self.manager
        self.manager.server_status = ServerStatus.Up
        self.notifications = 0

        self.assertTrue(handle.health_check())
        self.manager.gracefully_exit = True

        self.assertEqual(self.notifications, 1)
        self.assertFalse(handle.health_check())


if __name__ == "__main__":
    unittest.main(verbosity=2)
