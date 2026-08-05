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
    def __init__(self, responses, *, incremental=False):
        self.responses = responses
        self.server_args = SimpleNamespace(incremental_streaming_output=incremental)

    def generate_request(self, obj, request=None, **kwargs):
        async def generate():
            for response in self.responses:
                if isinstance(response, BaseException):
                    raise response
                yield response

        return generate()


def _make_runtime_handle(responses, *, incremental=False):
    handle = RuntimeHandle.__new__(RuntimeHandle)
    handle.tokenizer_manager = _FakeTokenizerManager(responses, incremental=incremental)
    return handle


class TestNativeGrpcParallelResponses(CustomTestCase):
    def test_non_streaming_returns_every_choice_before_finishing(self):
        callback = _RecordingCallback()
        responses = [
            [
                {
                    "output_ids": [1],
                    "meta_info": {
                        "id": "choice-0",
                        "finish_reason": {"type": "stop"},
                    },
                },
                {
                    "output_ids": [2],
                    "meta_info": {
                        "id": "choice-1",
                        "finish_reason": {"type": "length"},
                    },
                },
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
                choice_aware=True,
            )
        )

        self.assertEqual([call[0]["output_ids"] for call in callback.calls], [[1], [2]])
        self.assertEqual(
            [call[0]["delta_output_ids"] for call in callback.calls], [[1], [2]]
        )
        self.assertEqual([call[0]["index"] for call in callback.calls], [0, 1])
        self.assertEqual([call[1] for call in callback.calls], [False, True])

    def test_streaming_preserves_interleaved_choice_identity_and_deltas(self):
        callback = _RecordingCallback()
        responses = [
            {
                "index": 0,
                "output_ids": [1],
                "meta_info": {"id": "choice-0", "finish_reason": None},
            },
            {
                "index": 1,
                "output_ids": [3],
                "meta_info": {"id": "choice-1", "finish_reason": None},
            },
            {
                "index": 0,
                "output_ids": [1, 2],
                "meta_info": {
                    "id": "choice-0",
                    "finish_reason": {"type": "stop"},
                },
            },
            {
                "index": 1,
                "output_ids": [3, 4],
                "meta_info": {
                    "id": "choice-1",
                    "finish_reason": {"type": "length"},
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
                choice_aware=True,
            )
        )

        self.assertEqual([call[0]["index"] for call in callback.calls], [0, 1, 0, 1])
        self.assertEqual(
            [call[0]["output_ids"] for call in callback.calls],
            [[1], [3], [1, 2], [3, 4]],
        )
        self.assertEqual(
            [call[0]["delta_output_ids"] for call in callback.calls],
            [[1], [3], [2], [4]],
        )
        self.assertEqual(
            [call[1] for call in callback.calls], [False, False, False, True]
        )

    def test_generation_error_terminates_each_unfinished_choice(self):
        callback = _RecordingCallback()
        responses = [
            {
                "index": 0,
                "output_ids": [1],
                "meta_info": {
                    "id": "choice-0",
                    "finish_reason": {"type": "stop"},
                },
            },
            {
                "index": 1,
                "output_ids": [7],
                "meta_info": {"id": "choice-1", "finish_reason": None},
            },
            RuntimeError("scheduler failed"),
        ]
        handle = _make_runtime_handle(responses)
        obj = SimpleNamespace(rid="logical", batch_size=1, parallel_sample_num=2)

        asyncio.run(
            handle._run_generate(
                obj,
                callback,
                stream=True,
                request=None,
                choice_aware=True,
            )
        )

        self.assertEqual([call[0]["index"] for call in callback.calls], [0, 1, 1])
        self.assertEqual(
            callback.calls[2][0]["meta_info"]["finish_reason"]["type"], "error"
        )
        self.assertEqual(callback.calls[2][0]["output_ids"], [7])
        self.assertEqual([call[1] for call in callback.calls], [False, False, True])

    def test_incremental_streaming_preserves_legacy_segments(self):
        callback = _RecordingCallback()
        responses = [
            {
                "index": 0,
                "output_ids": [1],
                "meta_info": {"finish_reason": None},
            },
            {
                "index": 0,
                "output_ids": [2],
                "meta_info": {"finish_reason": {"type": "stop"}},
            },
        ]
        handle = _make_runtime_handle(responses, incremental=True)
        obj = SimpleNamespace(rid="logical", batch_size=1, parallel_sample_num=1)

        asyncio.run(
            handle._run_generate(
                obj,
                callback,
                stream=True,
                request=None,
                choice_aware=True,
            )
        )

        self.assertEqual([call[0]["output_ids"] for call in callback.calls], [[1], [2]])
        self.assertEqual(
            [call[0]["delta_output_ids"] for call in callback.calls], [[1], [2]]
        )


class TestNativeGrpcRequestLifecycle(CustomTestCase):
    def test_stale_lifecycle_cannot_abort_reused_request_id(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.rid_to_state = {
            "reused": SimpleNamespace(lifecycle_id=2, abort_requested=False)
        }
        manager.child_rid_to_logical_rid = {}
        manager.logical_rid_to_child_rids = {}

        aborted = manager.abort_request("reused", lifecycle_id=1)

        self.assertFalse(aborted)
        self.assertFalse(manager.rid_to_state["reused"].abort_requested)


if __name__ == "__main__":
    unittest.main(verbosity=2)
