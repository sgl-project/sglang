import asyncio
import enum
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.srt.entrypoints.openai.protocol import ErrorResponse
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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


class _KwargsRecordingCallback:
    def __init__(self):
        self.calls = []

    def __call__(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        return _ChunkStatus.Ready


class _ErrorResponseServing:
    """OpenAI serving whose handle_request rejects the request, as it does for
    a stream aborted before its first chunk (e.g. queue full)."""

    def __init__(self, enveloped=False):
        self.enveloped = enveloped

    async def handle_request(self, request, raw_request):
        error = ErrorResponse(
            object="error",
            message="The request queue is full.",
            type="503",
            code=503,
        ).model_dump()
        if self.enveloped:
            error = {"error": error}
        return ORJSONResponse(content=error, status_code=503)


def _make_openai_runtime_handle(enveloped=False):
    handle = RuntimeHandle.__new__(RuntimeHandle)
    handle._openai_serving_classes = {"chat": _ErrorResponseServing(enveloped)}
    handle._get_openai_request_class = lambda serving_key: (
        lambda **fields: SimpleNamespace(**fields)
    )
    return handle


class TestOpenAIGrpcErrorResponses(CustomTestCase):
    def _run(self, stream, enveloped=False):
        callback = _KwargsRecordingCallback()
        handle = _make_openai_runtime_handle(enveloped)
        asyncio.run(
            handle._run_openai_request(
                "chat",
                json.dumps({"stream": stream}).encode("utf-8"),
                callback,
                # submit_openai_chat always runs the streaming RPC; the
                # request body's "stream" flag picks the response shape.
                streaming=True,
            )
        )
        return callback.calls

    def test_streaming_error_response_is_an_in_band_error_chunk(self):
        """The streaming RPC drops status_code, so an error response must arrive
        as an {"error": ...} data chunk, not as a normal final chunk."""
        calls = self._run(stream=True)

        self.assertEqual(len(calls), 2)
        payload, kwargs = calls[0]
        self.assertFalse(kwargs["finished"])
        error = json.loads(payload)["error"]
        self.assertEqual(error["message"], "The request queue is full.")
        self.assertEqual(error["code"], 503)
        self.assertEqual(calls[1], (b"", {"finished": True}))

    def test_streaming_error_response_already_enveloped_is_not_rewrapped(self):
        calls = self._run(stream=True, enveloped=True)

        error = json.loads(calls[0][0])["error"]
        self.assertEqual(error["message"], "The request queue is full.")

    def test_non_streaming_error_response_keeps_status_code(self):
        """stream:false on the streaming RPC still gets one final chunk with the
        error body and status_code."""
        calls = self._run(stream=False)

        self.assertEqual(len(calls), 1)
        payload, kwargs = calls[0]
        self.assertEqual(kwargs, {"finished": True, "status_code": 503})
        self.assertEqual(json.loads(payload)["code"], 503)


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


class TestServerInfoSchedulerFields(CustomTestCase):
    def test_scheduler_info_fields_reach_json_info(self):
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle.tokenizer_manager = SimpleNamespace(
            server_args=ServerArgs(model_path="dummy"), mm_processor=None
        )
        handle.scheduler_info = {
            "max_req_input_len": 1024,
            "hicache_object_layout": {
                "enabled": True,
                "backend": "mooncake",
                "page_size": 256,
                "hicache_object_layout": {
                    "pools": [],
                    "key_prefix": "deepseek-ai-DeepSeek-V4",
                },
            },
        }

        payload = json.loads(handle.get_server_info())

        self.assertEqual(payload["max_req_input_len"], 1024)
        self.assertEqual(
            payload["hicache_object_layout"]["hicache_object_layout"]["key_prefix"],
            "deepseek-ai-DeepSeek-V4",
        )


class TestServerInfo(CustomTestCase):
    def test_advertises_each_media_kind_the_processor_places(self):
        vision = SimpleNamespace(
            mm_tokens=SimpleNamespace(
                image_token="<image>", video_token="<video>", audio_token=None
            )
        )
        for mm_processor, features in (
            (vision, ["image_inputs", "video_inputs"]),
            (SimpleNamespace(mm_tokens=None), []),
            (None, []),
        ):
            with self.subTest(features=features, mm_processor=mm_processor):
                handle = RuntimeHandle.__new__(RuntimeHandle)
                handle.tokenizer_manager = SimpleNamespace(
                    server_args=None, mm_processor=mm_processor
                )
                handle.scheduler_info = {}

                with patch(
                    "sglang.srt.entrypoints.grpc_bridge.build_server_info",
                    return_value={"model_path": "model"},
                ):
                    info = json.loads(handle.get_server_info())

                self.assertEqual(info["model_path"], "model")
                self.assertEqual(info["grpc_features"], features)


if __name__ == "__main__":
    unittest.main(verbosity=2)
