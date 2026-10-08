"""Dense tensor embeddings retain dtype through batching and IPC routing."""

import asyncio
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.multi_tokenizer_mixin import _handle_output_by_index
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _convert(embeddings, formats, retracted=None):
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(
                encoding_format=fmt,
                is_retracted=False if retracted is None else retracted[i],
            )
            for i, fmt in enumerate(formats)
        ]
    )
    return SchedulerBatchResultProcessor._convert_embeddings(
        None, result=SimpleNamespace(embeddings=embeddings), batch=batch
    )


def _stream(embeddings):
    streamer = SimpleNamespace(
        send_to_detokenizer=Mock(),
        get_cached_tokens_details=lambda req: None,
    )
    reqs = [
        SimpleNamespace(
            rid=f"r{i}",
            http_worker_ipc=f"worker{i}",
            finished=lambda: True,
            finished_reason=SimpleNamespace(to_json=lambda: {"type": "length"}),
            embedding=embedding,
            origin_input_ids=[1, 2],
            cached_tokens=0,
            time_stats=None,
            retraction_count=0,
            pooled_hidden_state=None,
        )
        for i, embedding in enumerate(embeddings)
    ]
    SchedulerOutputStreamer._stream_output_embedding(streamer, reqs)
    return streamer.send_to_detokenizer.send_output.call_args.args[0]


class TestEmbeddingEncodingFormat(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    def setUp(self):
        sparse = patch.object(
            envs.SGLANG_EMBEDDINGS_SPARSE_HEAD, "is_set", return_value=False
        )
        sparse.start()
        self.addCleanup(sparse.stop)
        codec = patch("sglang.srt.managers.io_struct._USE_PICKLE_IPC", False)
        codec.start()
        self.addCleanup(codec.stop)

    def test_normalization_and_subrequests(self):
        for fmt in (None, "float", "tensor", "TENSOR"):
            for cross_encoder in (False, True):
                with self.subTest(fmt=fmt, cross_encoder=cross_encoder):
                    req = EmbeddingReqInput(
                        text=[["q", "d"], ["q", "e"]] if cross_encoder else ["a", "b"],
                        encoding_format=fmt,
                        is_cross_encoder_request=cross_encoder,
                    )
                    req.normalize_batch_and_arguments()
                    expected = fmt.lower() if fmt is not None else None
                    self.assertEqual(req.encoding_format, expected)
                    for i in range(2):
                        self.assertEqual(req[i].encoding_format, expected)

    def test_invalid_format_rejected(self):
        for fmt in ("base64", "", "unknown", 123):
            with (
                self.subTest(fmt=fmt),
                self.assertRaisesRegex(ValueError, "encoding_format"),
            ):
                EmbeddingReqInput(
                    text="a", encoding_format=fmt
                ).normalize_batch_and_arguments()

    def test_media_only_batches_preserve_encoding_format(self):
        """Audio/video-only requests retain their encoding format when split."""
        for media_field in ("audio_data", "video_data"):
            for fmt in (None, "float", "TENSOR"):
                with self.subTest(media_field=media_field, fmt=fmt):
                    req = EmbeddingReqInput(
                        **{media_field: [["first"], ["second"]]},
                        encoding_format=fmt,
                    )
                    req.normalize_batch_and_arguments()
                    self.assertFalse(req.is_single)
                    self.assertEqual(req.batch_size, 2)
                    expected = fmt.lower() if fmt is not None else None
                    for i in range(2):
                        self.assertEqual(req[i].encoding_format, expected)
                        req[i].normalize_batch_and_arguments()

    def test_default_and_mixed_dense_batches(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            values = torch.arange(12, dtype=dtype).reshape(3, 4).requires_grad_()
            for formats in (
                (None, "float", None),
                ("tensor", "float", None),
                ("tensor",) * 3,
            ):
                with self.subTest(dtype=dtype, formats=formats):
                    outputs = _convert(values, formats)
                    for i, fmt in enumerate(formats):
                        if fmt == "tensor":
                            self.assertIsInstance(outputs[i], torch.Tensor)
                            self.assertEqual(outputs[i].device.type, "cpu")
                            self.assertEqual(outputs[i].dtype, dtype)
                            self.assertFalse(outputs[i].requires_grad)
                            torch.testing.assert_close(
                                outputs[i], values[i], rtol=0, atol=0
                            )
                        else:
                            self.assertEqual(outputs[i], values[i].tolist())

    def test_variable_shape_outputs_preserve_nesting(self):
        values = [
            torch.ones(2, 4, dtype=torch.float16),
            torch.zeros(3, 4, dtype=torch.float16),
        ]
        outputs = _convert(values, ("tensor", "float"))
        torch.testing.assert_close(outputs[0], values[0], rtol=0, atol=0)
        self.assertEqual(outputs[1], values[1].tolist())

    def test_retracted_tensor_request_does_not_change_default_output(self):
        values = torch.ones(2, 4)
        outputs = _convert(values, ("tensor", None), retracted=(True, False))
        self.assertEqual(outputs, values.tolist())

    def test_sparse_default_is_unchanged(self):
        values = torch.sparse_coo_tensor(
            [[0, 1], [2, 3]], [0.5, 1.0], (2, 4)
        ).coalesce()
        with patch.object(
            envs.SGLANG_EMBEDDINGS_SPARSE_HEAD, "is_set", return_value=True
        ):
            self.assertEqual(_convert(values, (None, "float")), [{2: 0.5}, {3: 1.0}])

    async def test_http_tensor_request_rejected_before_dispatch(self):
        manager = SimpleNamespace(auto_create_handle_loop=Mock())
        request = EmbeddingReqInput(text="a", encoding_format="tensor")
        with self.assertRaisesRegex(ValueError, "not HTTP endpoints"):
            await TokenizerManager.generate_request(
                manager, request, object()
            ).__anext__()

    async def test_sparse_tensor_request_rejected_before_dispatch(self):
        manager = SimpleNamespace(auto_create_handle_loop=Mock())
        request = EmbeddingReqInput(text="a", encoding_format="tensor")
        with patch.object(
            envs.SGLANG_EMBEDDINGS_SPARSE_HEAD, "is_set", return_value=True
        ):
            with self.assertRaisesRegex(ValueError, "requires dense embeddings"):
                await TokenizerManager.generate_request(manager, request).__anext__()

    async def test_mixed_outputs_survive_streaming_routing_and_receiving(self):
        tensor = torch.arange(8, dtype=torch.float16).reshape(2, 4)
        output = _stream([[1.0, 2.0], tensor])
        self.assertEqual(output.embeddings, [[1.0, 2.0], []])
        self.assertIsNone(output.tensor_embeddings[0])
        output = msgpack_decode(msgpack_encode(output))

        for index, expected in enumerate(([1.0, 2.0], tensor)):
            routed = msgpack_decode(
                msgpack_encode(_handle_output_by_index(output, index))
            )
            self.assertEqual(routed.rids, [f"r{index}"])
            self.assertEqual(routed.retraction_counts, [0])
            state = ReqState(
                out_list=[],
                finished=False,
                event=asyncio.Event(),
                obj=EmbeddingReqInput(text="a"),
                time_stats=Mock(
                    first_token_time=1.0,
                    trace_ctx=SimpleNamespace(tracing_enable=False),
                    get_e2e_latency=Mock(return_value=0.0),
                ),
            )
            manager = SimpleNamespace(
                rid_to_state={f"r{index}": state},
                config_value=lambda name: "test",
                enable_metrics=False,
                dump_requests_folder=None,
                crash_dump_folder=None,
                _release_lora_once=Mock(),
            )
            with (
                patch(
                    "sglang.srt.managers.tokenizer_manager.get_serving",
                    return_value=SimpleNamespace(batch_notify_size=1),
                ),
                patch(
                    "sglang.srt.managers.tokenizer_manager.get_spec",
                    return_value=SimpleNamespace(speculative_algorithm=None),
                ),
            ):
                await TokenizerManager._handle_batch_output(manager, routed)
            actual = state.out_list[0]["embedding"]
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.dtype, torch.float16)
            else:
                self.assertEqual(actual, expected)
            self.assertTrue(state.event.is_set())

    def test_float_only_stream_omits_tensor_payload(self):
        output = msgpack_decode(msgpack_encode(_stream([[1.0, 2.0]])))
        self.assertIsNone(output.tensor_embeddings)
        self.assertEqual(_handle_output_by_index(output, 0).embeddings, [[1.0, 2.0]])

    def test_tensor_output_also_supports_pickle_ipc(self):
        tensor = torch.ones(4, dtype=torch.bfloat16)
        with patch("sglang.srt.managers.io_struct._USE_PICKLE_IPC", True):
            output = pickle.loads(pickle.dumps(_stream([tensor])))
            routed = _handle_output_by_index(output, 0)
        torch.testing.assert_close(routed.tensor_embeddings[0], tensor, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
