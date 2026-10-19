import asyncio
import enum
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException

from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.srt.managers.tokenizer_manager import (
    ServerStatus,
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

    def __call__(self, payload, *, finished=False, error=None, status_code=None):
        self.calls.append((payload, finished, error))
        return _ChunkStatus.Ready


class _FakeTokenizerManager:
    def __init__(self, responses):
        self.responses = responses
        self.gracefully_exit = False

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
        handle.tokenizer_manager = SimpleNamespace(gracefully_exit=False)
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
    def setUp(self):
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager._gracefully_exit = False
        self.manager._server_status = ServerStatus.Up
        self.manager._is_pause = False
        self.notify = Mock()
        self.manager.set_engine_state_changed_callback(self.notify)
        self.manager.rid_to_state = {"in-flight": object()}
        self.handle = RuntimeHandle.__new__(RuntimeHandle)
        self.handle.tokenizer_manager = self.manager

    async def test_acceptance_is_applied_before_return_and_is_idempotent(self):
        self.handle._event_loop = asyncio.get_running_loop()
        active_requests = self.manager.rid_to_state.copy()
        results = await asyncio.wait_for(
            asyncio.gather(
                *(asyncio.to_thread(self.handle.shutdown) for _ in range(8))
            ),
            timeout=2,
        )
        self.assertEqual(results, [None] * 8)
        self.assertTrue(self.manager.gracefully_exit)
        self.assertFalse(self.manager.is_ready())
        self.assertFalse(self.handle.health_check())
        self.notify.assert_called_once_with()
        self.assertEqual(self.manager.rid_to_state, active_requests)

    async def test_shutdown_on_owner_loop_does_not_deadlock(self):
        self.handle._event_loop = asyncio.get_running_loop()
        self.handle.shutdown()
        self.assertTrue(self.manager.gracefully_exit)

    async def test_shutdown_reports_unavailable_event_loop(self):
        loop = asyncio.new_event_loop()
        self.handle._event_loop = loop
        try:
            with self.assertRaises(RuntimeError):
                self.handle.shutdown()
            loop.close()
            with self.assertRaises(RuntimeError):
                self.handle.shutdown()
            self.assertFalse(self.manager.gracefully_exit)
        finally:
            loop.close()

    async def test_timed_out_acceptance_can_still_finish(self):
        # Model a busy owner loop deterministically: retain its callback and
        # execute it after the caller's bounded wait has expired.
        callbacks = []
        self.handle._event_loop = SimpleNamespace(
            is_running=lambda: True,
            is_closed=lambda: False,
            call_soon_threadsafe=callbacks.append,
        )
        self.handle._SHUTDOWN_TIMEOUT_S = 0
        with self.assertRaises(TimeoutError):
            await asyncio.to_thread(self.handle.shutdown)
        self.assertFalse(self.manager.gracefully_exit)
        callbacks.pop()()
        self.assertTrue(self.manager.gracefully_exit)
        self.notify.assert_called_once_with()

    async def test_new_inference_is_rejected_before_admission(self):
        from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput

        self.handle._event_loop = asyncio.get_running_loop()
        self.handle.shutdown()
        self.manager.auto_create_handle_loop = Mock()
        self.manager._init_req_state = Mock()
        for obj in (GenerateReqInput(input_ids=[1]), EmbeddingReqInput(input_ids=[1])):
            with self.subTest(request_type=type(obj).__name__):
                with self.assertRaises(HTTPException) as caught:
                    await self.manager.generate_request(obj).__anext__()
                self.assertEqual(caught.exception.status_code, 503)
        self.manager._init_req_state.assert_not_called()

        with self.assertRaises(HTTPException) as caught:
            self.handle.submit_request(
                req_type="generate", req_dict={}, chunk_callback=None
            )
        self.assertEqual(caught.exception.status_code, 503)
        with self.assertRaises(HTTPException):
            self.handle.submit_openai_chat(json_body=b"{}", chunk_callback=None)

    async def test_openai_queued_before_shutdown_is_rejected_on_owner_loop(self):
        self.manager.gracefully_exit = True
        callback = Mock()
        await self.handle._run_openai_request("chat", b"{}", callback, streaming=True)
        self.assertEqual(callback.call_args.kwargs["status_code"], 503)
        self.assertTrue(callback.call_args.kwargs["finished"])

    async def test_watchdog_dispatches_scheduler_shutdown_after_drain(self):
        from sglang.srt.managers.io_struct import ShutdownReq

        self.handle._event_loop = asyncio.get_running_loop()
        self.handle.shutdown()
        self.manager._subprocess_watchdog = Mock()
        self.manager._server_stop_hook = Mock()
        self.manager._dispatch_to_scheduler = Mock()
        self.manager.asyncio_tasks = set()

        async def finish_active_request(_):
            self.manager._dispatch_to_scheduler.assert_not_called()
            self.manager.rid_to_state.clear()

        module = "sglang.srt.managers.tokenizer_manager"
        with (
            patch(f"{module}.asyncio.sleep", side_effect=finish_active_request),
            patch(f"{module}.get_bool_env_var", return_value=False),
            patch(f"{module}.collect_scheduler_processes", return_value=[]),
            patch(f"{module}.kill_process_tree") as cleanup,
        ):
            await self.manager.sigterm_watchdog()
        self.manager._subprocess_watchdog.stop.assert_called_once()
        self.manager._dispatch_to_scheduler.assert_called_once()
        self.assertIsInstance(
            self.manager._dispatch_to_scheduler.call_args.args[0], ShutdownReq
        )
        cleanup.assert_called_once()
        self.manager._server_stop_hook.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
