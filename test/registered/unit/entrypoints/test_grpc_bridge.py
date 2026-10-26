import asyncio
import enum
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

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


class TestNativeGrpcOpenAIResponses(CustomTestCase):
    def test_rerank_list_of_models_is_serialized_as_over_http(self):
        from sglang.srt.entrypoints.openai.protocol import RerankResponse

        class _Rerank:
            async def handle_request(self, request, raw_request):
                return [RerankResponse(score=0.5, document="d", index=0)]

        calls = []
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle._get_openai_serving = lambda: {"rerank": _Rerank()}
        body = b'{"query": "q", "documents": ["d"]}'
        callback = lambda payload, **kwargs: calls.append((payload, kwargs))

        asyncio.run(
            handle._run_openai_request("rerank", body, callback, streaming=False)
        )

        payload, kwargs = calls[-1]
        self.assertEqual(kwargs["status_code"], 200)
        self.assertEqual(json.loads(payload)[0]["score"], 0.5)


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
    async def test_shutdown_uses_configured_sigterm_handler_on_owner_loop(self):
        loop = asyncio.get_running_loop()
        active_requests = {"in-flight": object()}
        manager = SimpleNamespace(gracefully_exit=False, rid_to_state=active_requests)

        def sigterm():
            self.assertIs(asyncio.get_running_loop(), loop)
            SignalHandler(manager).sigterm_handler()

        callback = Mock(side_effect=sigterm)
        manager.signal_handler_class = Mock(
            return_value=SimpleNamespace(sigterm_handler=callback)
        )
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle.tokenizer_manager = manager
        handle._event_loop = loop
        results = await asyncio.gather(
            *(asyncio.to_thread(handle.shutdown) for _ in range(8))
        )
        await asyncio.sleep(0)
        self.assertEqual(results, [None] * 8)
        self.assertEqual(callback.call_count, 8)
        manager.signal_handler_class.assert_called_with(manager)
        self.assertTrue(manager.gracefully_exit)
        self.assertIs(manager.rid_to_state, active_requests)

    async def test_shutdown_reports_unavailable_event_loop(self):
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle._event_loop = asyncio.new_event_loop()
        try:
            with self.assertRaises(RuntimeError):
                handle.shutdown()
            handle._event_loop.close()
            with self.assertRaises(RuntimeError):
                handle.shutdown()
        finally:
            handle._event_loop.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
