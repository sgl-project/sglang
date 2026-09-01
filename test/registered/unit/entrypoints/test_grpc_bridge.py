import asyncio
import enum
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.srt.managers.tokenizer_manager import TokenizerManager
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


class TestNativeGrpcPauseStatus(CustomTestCase):
    def test_status_tracks_pause_and_continue(self):
        async def run_test():
            tokenizer_manager = TokenizerManager.__new__(TokenizerManager)
            tokenizer_manager.is_pause = False
            tokenizer_manager.is_pause_cond = asyncio.Condition()
            dispatch_started = asyncio.Event()
            finish_dispatch = asyncio.Event()

            async def dispatch_to_scheduler(_obj):
                dispatch_started.set()
                await finish_dispatch.wait()

            tokenizer_manager._async_dispatch_to_scheduler = dispatch_to_scheduler
            handle = RuntimeHandle.__new__(RuntimeHandle)
            handle.tokenizer_manager = tokenizer_manager

            self.assertFalse(handle.get_pause_status())

            pause_task = asyncio.create_task(
                tokenizer_manager.pause_generation(SimpleNamespace(mode="in_place"))
            )
            await dispatch_started.wait()

            self.assertTrue(handle.get_pause_status())

            finish_dispatch.set()
            await pause_task

            dispatch_started.clear()
            finish_dispatch.clear()
            continue_task = asyncio.create_task(
                tokenizer_manager.continue_generation(SimpleNamespace())
            )
            await dispatch_started.wait()

            self.assertTrue(handle.get_pause_status())

            finish_dispatch.set()
            await continue_task

            self.assertFalse(handle.get_pause_status())

        asyncio.run(run_test())


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
