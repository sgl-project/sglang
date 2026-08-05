import asyncio
import enum
import threading
import unittest
from concurrent.futures import Future
from types import SimpleNamespace

from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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
        obj = SimpleNamespace(rid="logical", batch_size=1, parallel_sample_num=2)

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


class TestNativeGrpcRequestLifecycle(CustomTestCase):
    def test_reused_request_id_keeps_current_generation_future(self):
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle._active_generation_futures = {}
        handle._generation_futures_lock = threading.Lock()
        old_future = Future()
        current_future = Future()

        handle._track_generation_future("reused", old_future)
        handle._track_generation_future("reused", current_future)
        old_future.set_result(None)

        self.assertIs(handle._active_generation_futures["reused"], current_future)
        handle._cancel_generation_futures("reused", abort_all=False)
        self.assertTrue(current_future.cancelled())


if __name__ == "__main__":
    unittest.main(verbosity=2)
