import asyncio
import enum
import unittest
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


class TestNativeGrpcOutputShape(CustomTestCase):
    @staticmethod
    def _submit_generate(*, tokenizer_manager=None, **kwargs):
        handle = RuntimeHandle.__new__(RuntimeHandle)
        handle.tokenizer_manager = tokenizer_manager or SimpleNamespace(
            request_logger=SimpleNamespace(log_requests=False, log_requests_level=0),
            dump_requests_folder="",
            crash_dump_folder="",
            request_metrics_exporter_manager=SimpleNamespace(
                exporter_enabled=lambda: False
            ),
        )
        captured = []

        async def capture(obj, chunk_callback, stream, request):
            captured.append(obj)

        handle._run_generate = capture
        handle._submit_on_tm_loop = asyncio.run
        handle.submit_request(
            req_type="generate",
            req_dict={"input_ids": [1], "sampling_params": {}, "stream": True},
            chunk_callback=_RecordingCallback(),
            **kwargs,
        )
        return captured[0]

    def test_submit_request_marks_token_id_generate_no_text(self):
        obj = self._submit_generate(output_text_required=False)

        self.assertFalse(obj._output_text_required)

    def test_submit_request_defaults_to_text_required(self):
        obj = self._submit_generate()

        self.assertTrue(obj._output_text_required)

    def test_internal_output_consumers_keep_text(self):
        cases = {
            "request logging": {
                "request_logger": SimpleNamespace(
                    log_requests=True, log_requests_level=2
                )
            },
            "request dump": {"dump_requests_folder": "/dump"},
            "crash dump": {"crash_dump_folder": "/crash"},
            "metrics exporter": {
                "request_metrics_exporter_manager": SimpleNamespace(
                    exporter_enabled=lambda: True
                )
            },
        }
        for name, overrides in cases.items():
            with self.subTest(name=name):
                manager_config = {
                    "request_logger": SimpleNamespace(
                        log_requests=False, log_requests_level=0
                    ),
                    "dump_requests_folder": "",
                    "crash_dump_folder": "",
                    "request_metrics_exporter_manager": SimpleNamespace(
                        exporter_enabled=lambda: False
                    ),
                }
                manager_config.update(overrides)
                manager = SimpleNamespace(**manager_config)
                obj = self._submit_generate(
                    tokenizer_manager=manager, output_text_required=False
                )

                self.assertTrue(obj._output_text_required)


if __name__ == "__main__":
    unittest.main(verbosity=2)
