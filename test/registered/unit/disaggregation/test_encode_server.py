import asyncio
import pickle
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
from sglang.srt.disaggregation.encoder.receiver import EmbeddingData
from sglang.srt.disaggregation.encoder.runtime import execute_encode_pipeline
from sglang.srt.disaggregation.encoder.server import (
    BadRequestError,
    EncodeContext,
    EncoderDelivery,
    InternalError,
    MMEncoder,
    MooncakeDelivery,
    ReqState,
    SendDestination,
    ZmqDelivery,
    meta_registry,
    rid_to_cond,
    rid_to_receive_count,
    rid_to_receive_endpoint,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.utils.common import safe_pickle_loads
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEncoderPreprocessorKimiGrid(CustomTestCase):
    @staticmethod
    def _make_preprocessor(model_type="kimi_vl"):
        preprocessor = EncoderPreprocessor.__new__(EncoderPreprocessor)
        preprocessor.model_type = model_type
        preprocessor.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                vision_config=SimpleNamespace(merge_kernel_size=(2, 2))
            )
        )
        preprocessor.image_processor = SimpleNamespace(merge_size=2)
        preprocessor._model_preprocessor = None
        return preprocessor

    @staticmethod
    def _make_encoder():
        return MMEncoder.__new__(MMEncoder)

    def test_kimi_vl_prefers_and_normalizes_hw_grid(self):
        mm_inputs = {
            "image_grid_hws": np.array([[40, 60]], dtype=np.int64),
            "image_grid_thw": torch.tensor([[1, 20, 30]]),
            "grid_thws": torch.tensor([[1, 10, 15]]),
        }

        grid = self._make_preprocessor()._get_mm_grid_dim(mm_inputs, Modality.IMAGE)

        self.assertIsInstance(grid, torch.Tensor)
        torch.testing.assert_close(grid, torch.tensor([[40, 60]]))

    def test_kimi_k25_keeps_thw_grid_preference(self):
        mm_inputs = {
            "image_grid_hws": np.array([[40, 60]], dtype=np.int64),
            "grid_thws": np.array([[1, 10, 15]], dtype=np.int64),
        }

        grid = self._make_preprocessor("kimi_k25")._get_mm_grid_dim(
            mm_inputs, Modality.IMAGE
        )

        torch.testing.assert_close(grid, torch.tensor([[1, 10, 15]]))

    def test_kimi_vl_2d_grid_counting_and_slicing(self):
        preprocessor = self._make_preprocessor()
        encoder = self._make_encoder()
        grids = torch.tensor([[40, 60], [20, 40]])
        embedding = torch.arange(800 * 2).reshape(800, 2)

        self.assertEqual(
            preprocessor.get_num_patches(grids[0], Modality.IMAGE),
            2400,
        )
        self.assertEqual(
            preprocessor.get_num_tokens(grids[0], Modality.IMAGE),
            600,
        )

        slices = encoder.slice_embedding(embedding, [600, 200])

        self.assertEqual([item.shape for item in slices], [(600, 2), (200, 2)])
        torch.testing.assert_close(slices[0], embedding[:600])
        torch.testing.assert_close(slices[1], embedding[600:])

    def test_kimi_3d_grid_remains_supported(self):
        preprocessor = self._make_preprocessor()
        grid = torch.tensor([1, 40, 60])

        self.assertEqual(preprocessor.get_num_patches(grid, Modality.IMAGE), 2400)
        self.assertEqual(preprocessor.get_num_tokens(grid, Modality.IMAGE), 600)

    def test_kimi_k25_3d_patch_counting_is_unchanged(self):
        preprocessor = self._make_preprocessor("kimi_k25")
        grid = torch.tensor([2, 12, 16])

        self.assertEqual(preprocessor.get_num_patches(grid, Modality.IMAGE), 384)
        self.assertEqual(preprocessor.get_num_tokens(grid, Modality.IMAGE), 48)

    def test_grid_metadata_is_safe_to_deserialize(self):
        grid = self._make_preprocessor()._get_mm_grid_dim(
            {"image_grid_hws": np.array([[40, 60]], dtype=np.int64)},
            Modality.IMAGE,
        )
        embedding_data = EmbeddingData(
            req_id="test-request",
            num_parts=1,
            part_idx=0,
            grid_dim=grid,
            modality=Modality.IMAGE,
            embedding=torch.zeros((600, 4)),
        )

        restored = safe_pickle_loads(
            pickle.dumps(embedding_data.copy_without_embedding())
        )

        torch.testing.assert_close(restored.grid_dim, torch.tensor([[40, 60]]))


class TestEncoderDelivery(CustomTestCase):
    def test_contract_has_two_direct_implementations(self):
        self.assertEqual(EncoderDelivery.__abstractmethods__, {"send", "release"})
        self.assertEqual(
            set(EncoderDelivery.__subclasses__()),
            {
                MooncakeDelivery,
                ZmqDelivery,
            },
        )

    def test_zmq_delivery_cleanup_is_configurable(self):
        async def run():
            req_id = "test-zmq-delivery-cleanup"
            rid_to_receive_endpoint[req_id] = {"127.0.0.1:1"}
            rid_to_receive_count[req_id] = 1
            rid_to_cond[req_id] = asyncio.Condition()
            state = ReqState(req_id)
            encoder = SimpleNamespace()

            await ZmqDelivery(encoder, cleanup_receive_state=False).release(state)
            self.assertIn(req_id, rid_to_receive_endpoint)
            self.assertIn(req_id, rid_to_receive_count)
            self.assertIn(req_id, rid_to_cond)

            await ZmqDelivery(encoder, cleanup_receive_state=True).release(state)
            self.assertNotIn(req_id, rid_to_receive_endpoint)
            self.assertNotIn(req_id, rid_to_receive_count)
            self.assertNotIn(req_id, rid_to_cond)

        asyncio.run(run())

    def test_preprocess_metadata_precedes_embedding(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 0
            encoder.req_states = {}
            encoder._embedding_dims = {Modality.IMAGE: 8}
            encoder._embedding_dtype = torch.float16
            encoder._element_size = 2
            first_state = encoder._acquire_encode_ref("req-0")
            second_state = encoder._acquire_encode_ref("req-1")
            ctx = SimpleNamespace(
                req_id="req-0",
                modality=Modality.IMAGE,
                items_per_req=[1, 2],
                preprocess_result=SimpleNamespace(
                    token_counts=[2, 3, 4],
                    grid_thw=[[1, 2, 3], [1, 4, 5], [1, 6, 7]],
                ),
            )
            requests = [
                {"req_id": "req-0", "num_parts": 2, "part_idx": 0},
                {"req_id": "req-1", "num_parts": 2, "part_idx": 1},
            ]

            publish = AsyncMock()
            with patch.object(meta_registry, "publish", publish):
                await encoder._publish_preprocess_metadata(ctx, requests)

            self.assertIs(encoder.req_states["req-0"], first_state)
            self.assertIs(encoder.req_states["req-1"], second_state)
            self.assertEqual(first_state.embedding_data.shape, [2, 8])
            self.assertEqual(second_state.embedding_data.shape, [7, 8])
            self.assertEqual(first_state.embedding_data.grid_dim, [[1, 2, 3]])
            self.assertEqual(
                second_state.embedding_data.grid_dim,
                [[1, 4, 5], [1, 6, 7]],
            )
            self.assertEqual(first_state.embedding_data.dtype, torch.float16)
            self.assertEqual(second_state.embedding_data.dtype, torch.float16)
            self.assertFalse(first_state.embedding_ready.is_set())
            self.assertFalse(second_state.embedding_ready.is_set())
            self.assertEqual(
                publish.await_args_list,
                [
                    unittest.mock.call("req-0", 32, 2, 8),
                    unittest.mock.call("req-1", 112, 7, 8),
                ],
            )
            await encoder._release_encode_ref(first_state)
            await encoder._release_encode_ref(second_state)

        asyncio.run(run())

    @staticmethod
    def _encode_context():
        return EncodeContext(
            req_id="req",
            modality=Modality.IMAGE,
            preprocess_result=SimpleNamespace(
                token_counts=[2],
                grid_thw=torch.tensor([[1, 2, 4]]),
            ),
            get_feature_fn=None,
            mm_feature=torch.zeros((8, 3)),
            num_items=1,
            items_per_req=[1],
            aux_data={},
            str_mm_hashes=None,
            use_global_cache=False,
            is_health_check=False,
        )

    @staticmethod
    def _load_grpc_server():
        try:
            import grpc
            from smg_grpc_proto import sglang_encoder_pb2
        except ImportError as e:
            raise unittest.SkipTest(f"gRPC test dependencies unavailable: {e}") from e
        except Exception as e:
            # Generated protobuf modules raise VersionError when the runner's
            # protobuf runtime is older than the code generator.
            if type(e).__name__ != "VersionError":
                raise
            raise unittest.SkipTest(f"gRPC test dependencies unavailable: {e}") from e

        # Import SGLang outside the dependency guard. Product-code import
        # failures are regressions and must fail the test instead of skipping.
        from sglang.srt.disaggregation.encoder.grpc_server import (
            SGLangEncoderServer,
        )

        return grpc, sglang_encoder_pb2, SGLangEncoderServer

    def test_remote_preprocess_failure_stops_all_tp_ranks_before_forward(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder._prepare_encode_context = AsyncMock(
                return_value=self._encode_context()
            )
            encoder._publish_preprocess_metadata = AsyncMock()

            class TPGroup:
                world_size = 2
                cpu_group = object()

                @staticmethod
                def all_gather_object(local_error):
                    return [local_error, "bad image"]

            def all_gather(statuses, local_status, group):
                self.assertIs(group, TPGroup.cpu_group)
                statuses[0].copy_(local_status)
                statuses[1].copy_(torch.tensor([400, 1, 0, 0]))

            with (
                patch(
                    "sglang.srt.disaggregation.encoder.server.get_tp_group",
                    return_value=TPGroup(),
                ),
                patch(
                    "sglang.srt.disaggregation.encoder.server.torch.distributed.all_gather",
                    side_effect=all_gather,
                ),
            ):
                with self.assertRaisesRegex(
                    BadRequestError,
                    "failed on TP rank 1: bad image",
                ):
                    await encoder._prepare_encode_context_on_all_ranks(
                        [{"req_id": "req"}],
                        Modality.IMAGE,
                        use_global_cache=False,
                    )

        asyncio.run(run())

    def test_tp_preprocess_layout_mismatch_fails_before_forward(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder._prepare_encode_context = AsyncMock(
                return_value=self._encode_context()
            )
            encoder._publish_preprocess_metadata = AsyncMock()

            class TPGroup:
                world_size = 2
                cpu_group = object()

            def all_gather(statuses, local_status, group):
                self.assertIs(group, TPGroup.cpu_group)
                statuses[0].copy_(local_status)
                statuses[1].copy_(local_status)
                statuses[1][2] += 1

            with (
                patch(
                    "sglang.srt.disaggregation.encoder.server.get_tp_group",
                    return_value=TPGroup(),
                ),
                patch(
                    "sglang.srt.disaggregation.encoder.server.torch.distributed.all_gather",
                    side_effect=all_gather,
                ),
            ):
                with self.assertRaisesRegex(
                    InternalError,
                    "inconsistent layouts across TP ranks 0 and 1",
                ):
                    await encoder._prepare_encode_context_on_all_ranks(
                        [{"req_id": "req"}],
                        Modality.IMAGE,
                        use_global_cache=False,
                    )

        asyncio.run(run())

    def test_remote_metadata_failure_stops_tp_peer_before_forward(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder._prepare_encode_context = AsyncMock(
                return_value=self._encode_context()
            )
            encoder._publish_preprocess_metadata = AsyncMock()

            class TPGroup:
                world_size = 2
                cpu_group = object()

                @staticmethod
                def all_gather_object(local_error):
                    self.assertIsNone(local_error)
                    return ["registry down", None]

            def all_gather(statuses, local_status, group):
                self.assertIs(group, TPGroup.cpu_group)
                statuses[0].copy_(torch.tensor([500, 2, 0, 0]))
                statuses[1].copy_(local_status)

            with (
                patch(
                    "sglang.srt.disaggregation.encoder.server.get_tp_group",
                    return_value=TPGroup(),
                ),
                patch(
                    "sglang.srt.disaggregation.encoder.server.torch.distributed.all_gather",
                    side_effect=all_gather,
                ),
            ):
                with self.assertRaisesRegex(
                    InternalError,
                    "metadata publication failed on TP rank 0: registry down",
                ):
                    await encoder._prepare_encode_context_on_all_ranks(
                        [{"req_id": "req"}],
                        Modality.IMAGE,
                        use_global_cache=False,
                    )

        asyncio.run(run())

    def test_unexpected_preprocess_failure_is_internal(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.preprocessor = SimpleNamespace(
                process_batch_mm_items=AsyncMock(side_effect=RuntimeError("boom"))
            )
            with self.assertRaisesRegex(InternalError, "boom"):
                await encoder._prepare_encode_context(
                    [{"req_id": "req"}],
                    Modality.IMAGE,
                    use_global_cache=False,
                )

        asyncio.run(run())

    def test_grpc_rejects_invalid_request_before_tp_dispatch(self):
        async def run():
            grpc, sglang_encoder_pb2, SGLangEncoderServer = self._load_grpc_server()

            context = SimpleNamespace(
                set_code=unittest.mock.Mock(),
                set_details=unittest.mock.Mock(),
            )
            server = SGLangEncoderServer(
                encoder=SimpleNamespace(),
                send_sockets=[object()],
                server_args=SimpleNamespace(),
            )
            request = sglang_encoder_pb2.EncodeRequest(
                mm_items=["image"],
                req_id="invalid",
                part_idx=0,
            )

            with patch(
                "sglang.srt.disaggregation.encoder.grpc_server.async_sock_send",
                new_callable=AsyncMock,
            ) as send:
                await server.Encode(request, context)

            send.assert_not_awaited()
            context.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
            self.assertIn("num_parts", context.set_details.call_args.args[0])

        asyncio.run(run())

    def test_grpc_maps_processor_bad_request_to_invalid_argument(self):
        async def run():
            grpc, sglang_encoder_pb2, SGLangEncoderServer = self._load_grpc_server()

            encoder = SimpleNamespace(
                encode_dispatch_lock=asyncio.Lock(),
                encode_request=AsyncMock(
                    return_value=(
                        0,
                        0,
                        0,
                        "invalid image",
                        HTTPStatus.BAD_REQUEST,
                    )
                ),
                release_request=AsyncMock(),
            )
            context = SimpleNamespace(
                set_code=unittest.mock.Mock(),
                set_details=unittest.mock.Mock(),
            )
            server = SGLangEncoderServer(
                encoder=encoder,
                send_sockets=[],
                server_args=SimpleNamespace(),
            )
            request = sglang_encoder_pb2.EncodeRequest(
                mm_items=["bad-image"],
                req_id="bad-image",
                num_parts=1,
                part_idx=0,
            )

            await server.Encode(request, context)

            context.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details.assert_called_once_with("invalid image")
            encoder.release_request.assert_awaited_once_with("bad-image")

        asyncio.run(run())

    def test_grpc_serializes_tp_dispatch_with_rank_zero_encode(self):
        async def run():
            _, sglang_encoder_pb2, SGLangEncoderServer = self._load_grpc_server()
            from sglang.srt.managers.io_struct import unwrap_from_pickle

            first_started = asyncio.Event()
            release_first = asyncio.Event()
            events = []

            class Encoder:
                def __init__(self):
                    self.encode_dispatch_lock = asyncio.Lock()

                async def encode_request(self, request, _modality):
                    req_id = request["req_id"]
                    events.append(("encode-start", req_id))
                    if req_id == "first":
                        first_started.set()
                        await release_first.wait()
                    events.append(("encode-end", req_id))
                    return 8, 1, 8, None, None

            async def send(_socket, payload):
                request = unwrap_from_pickle(payload)
                events.append(("send", request["req_id"]))

            server = SGLangEncoderServer(
                encoder=Encoder(),
                send_sockets=[object()],
                server_args=SimpleNamespace(),
            )
            requests = [
                sglang_encoder_pb2.EncodeRequest(
                    mm_items=["image"],
                    req_id=req_id,
                    num_parts=1,
                    part_idx=0,
                )
                for req_id in ("first", "second")
            ]
            contexts = [
                SimpleNamespace(
                    set_code=unittest.mock.Mock(),
                    set_details=unittest.mock.Mock(),
                )
                for _ in requests
            ]

            with (
                patch(
                    "sglang.srt.disaggregation.encoder.grpc_server.async_sock_send",
                    side_effect=send,
                ),
                patch(
                    "sglang.srt.disaggregation.encoder.grpc_server.get_disagg",
                    return_value=SimpleNamespace(encoder_transfer_backend="mooncake"),
                ),
            ):
                first = asyncio.create_task(server.Encode(requests[0], contexts[0]))
                await first_started.wait()
                second = asyncio.create_task(server.Encode(requests[1], contexts[1]))
                await asyncio.sleep(0)
                self.assertEqual(
                    events,
                    [("send", "first"), ("encode-start", "first")],
                )
                release_first.set()
                await asyncio.gather(first, second)

            self.assertEqual(
                events,
                [
                    ("send", "first"),
                    ("encode-start", "first"),
                    ("encode-end", "first"),
                    ("send", "second"),
                    ("encode-start", "second"),
                    ("encode-end", "second"),
                ],
            )

        asyncio.run(run())

    def test_grpc_encode_cancellation_releases_request(self):
        async def run():
            _, sglang_encoder_pb2, SGLangEncoderServer = self._load_grpc_server()

            encode_started = asyncio.Event()
            finish_encode = asyncio.Event()

            async def encode_request(*_args):
                encode_started.set()
                await finish_encode.wait()
                return 8, 1, 8, None, None

            encoder = SimpleNamespace(
                encode_dispatch_lock=asyncio.Lock(),
                encode_request=AsyncMock(side_effect=encode_request),
                release_request=AsyncMock(),
            )
            server = SGLangEncoderServer(
                encoder=encoder,
                send_sockets=[],
                server_args=SimpleNamespace(),
            )
            request = sglang_encoder_pb2.EncodeRequest(
                mm_items=["image"],
                req_id="cancelled-encode",
                num_parts=1,
                part_idx=0,
            )
            context = SimpleNamespace(
                set_code=unittest.mock.Mock(),
                set_details=unittest.mock.Mock(),
            )

            with patch(
                "sglang.srt.disaggregation.encoder.grpc_server.get_disagg",
                return_value=SimpleNamespace(encoder_transfer_backend="mooncake"),
            ):
                task = asyncio.create_task(server.Encode(request, context))
                await encode_started.wait()
                task.cancel()
                await asyncio.sleep(0)

                # Cancellation cannot interrupt an in-flight TP collective.
                self.assertFalse(task.done())
                encoder.release_request.assert_not_awaited()

                finish_encode.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            encoder.release_request.assert_awaited_once_with("cancelled-encode")

        asyncio.run(run())

    def test_grpc_send_cancellation_releases_request(self):
        async def run():
            _, sglang_encoder_pb2, SGLangEncoderServer = self._load_grpc_server()

            send_started = asyncio.Event()

            async def send(*_args, **_kwargs):
                send_started.set()
                await asyncio.Event().wait()

            encoder = SimpleNamespace(
                send=AsyncMock(side_effect=send),
                release_request=AsyncMock(),
            )
            server = SGLangEncoderServer(
                encoder=encoder,
                send_sockets=[],
                server_args=SimpleNamespace(),
            )
            request = sglang_encoder_pb2.SendRequest(
                req_id="cancelled-send",
                prefill_host="127.0.0.1",
                embedding_port=30001,
            )
            context = SimpleNamespace()

            task = asyncio.create_task(server.Send(request, context))
            await send_started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

            encoder.release_request.assert_awaited_once_with("cancelled-send")

        asyncio.run(run())

    def test_mooncake_embedding_is_ready_only_after_cuda_sync(self):
        class FakeCudaEmbedding:
            shape = (2, 4)
            dtype = torch.float16
            nbytes = 16
            is_cuda = True
            device = "cuda:0"

            def __getitem__(self, key):
                return self

        encoder = MMEncoder.__new__(MMEncoder)
        encoder.rank = 0
        events = []
        encoder._stage_embedding = lambda mm_data: events.append("ready")
        ctx = SimpleNamespace(
            req_id="req",
            modality=Modality.IMAGE,
            items_per_req=[1],
            preprocess_result=SimpleNamespace(
                token_counts=[2],
                grid_thw=[[1, 2, 3]],
            ),
            aux_data={},
            use_global_cache=True,
        )
        stream = SimpleNamespace(synchronize=lambda: events.append("sync"))

        with patch.object(torch.cuda, "current_stream", return_value=stream):
            encoder._stage_embeddings(
                ctx,
                [{"req_id": "req", "num_parts": 1, "part_idx": 0}],
                FakeCudaEmbedding(),
                keep_on_gpu=True,
            )

        self.assertEqual(events, ["sync", "ready"])

    def test_stage_embedding_does_not_resurrect_missing_state(self):
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.req_states = {}

        with self.assertRaisesRegex(
            InternalError, "No request state exists while encoding request: req"
        ):
            encoder._stage_embedding(
                EmbeddingData(
                    "req",
                    1,
                    0,
                    None,
                    Modality.IMAGE,
                    embedding=torch.ones((1, 1)),
                )
            )

        self.assertNotIn("req", encoder.req_states)

    def test_stage_embedding_requires_active_encode(self):
        encoder = MMEncoder.__new__(MMEncoder)
        state = ReqState("req")
        encoder.req_states = {"req": state}

        with self.assertRaisesRegex(
            InternalError, "Request state has no active encode work: req"
        ):
            encoder._stage_embedding(
                EmbeddingData(
                    "req",
                    1,
                    0,
                    None,
                    Modality.IMAGE,
                    embedding=torch.ones((1, 1)),
                )
            )

        self.assertIs(encoder.req_states["req"], state)
        self.assertIsNone(state.embedding_data)
        self.assertFalse(state.embedding_ready.is_set())

    def test_release_during_encode_is_deferred_without_resurrecting_state(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 0
            encoder.req_states = {}
            encoder.delivery = SimpleNamespace(release=AsyncMock())

            state = encoder._acquire_encode_ref("req")
            state.embedding_data = EmbeddingData(
                "req",
                1,
                0,
                None,
                Modality.IMAGE,
                embedding_shape=[1, 1],
                dtype=torch.float32,
            )

            discard = AsyncMock()
            with patch.object(meta_registry, "discard", discard):
                await encoder.release_request("req")
                self.assertTrue(state.release_requested)
                self.assertIn("req", encoder.req_states)
                encoder.delivery.release.assert_not_awaited()

                embedding = torch.ones((1, 1))
                encoder._stage_embedding(
                    EmbeddingData(
                        "req",
                        1,
                        0,
                        None,
                        Modality.IMAGE,
                        embedding=embedding,
                    )
                )
                await encoder._release_encode_ref(state)

            encoder.delivery.release.assert_awaited_once_with(state)
            discard.assert_awaited_once_with("req")
            self.assertIsNone(state.embedding_data)
            self.assertNotIn("req", encoder.req_states)

        asyncio.run(run())

    def test_error_metadata_survives_buffer_release_for_waiter(self):
        async def run():
            req_id = "test-error-metadata-waiter"
            await meta_registry.discard(req_id)

            encoder = MMEncoder.__new__(MMEncoder)
            encoder.req_states = {}
            encoder.delivery = SimpleNamespace(release=AsyncMock())
            state = ReqState(
                req_id,
                EmbeddingData(
                    req_id,
                    1,
                    0,
                    None,
                    Modality.IMAGE,
                    error_msg="encode failed",
                ),
            )
            state.embedding_ready.set()
            encoder.req_states[req_id] = state

            waiter = asyncio.create_task(meta_registry.wait(req_id))
            await asyncio.sleep(0)
            try:
                await meta_registry.publish(req_id, 0, 0, 0, error="encode failed")
                await encoder.release_request(req_id, preserve_metadata=True)
                meta = await asyncio.wait_for(waiter, timeout=1)
                self.assertEqual(meta, {"error": "encode failed"})
                self.assertNotIn(req_id, encoder.req_states)
            finally:
                if not waiter.done():
                    waiter.cancel()
                await meta_registry.discard(req_id)

        asyncio.run(run())

    def test_zmq_pipeline_sends_only_after_encode_completes(self):
        async def run():
            events = []
            finish_encode = asyncio.Event()
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.transfer_backend = "zmq_to_tokenizer"

            async def encode(**kwargs):
                events.append("metadata_published")
                await finish_encode.wait()
                events.append("encode_completed")
                return 16, 2, 4, None, None

            async def send(**kwargs):
                events.append("send")
                return True

            async def release_request(req_id, **kwargs):
                events.append("release")

            encoder.encode = AsyncMock(side_effect=encode)
            encoder.send = AsyncMock(side_effect=send)
            encoder.release_request = AsyncMock(side_effect=release_request)

            publish = AsyncMock()
            request = {
                "req_id": "req",
                "mm_items": ["item"],
                "modality": "image",
                "num_parts": 1,
                "part_idx": 0,
                "prefill_host": "127.0.0.1",
                "embedding_port": 1234,
            }
            with patch.object(meta_registry, "publish", publish):
                task = asyncio.create_task(
                    execute_encode_pipeline(encoder, None, request)
                )
                await asyncio.sleep(0)
                self.assertEqual(events, ["metadata_published"])
                encoder.send.assert_not_awaited()

                finish_encode.set()
                self.assertIsNone(await task)

            self.assertEqual(
                events,
                ["metadata_published", "encode_completed", "send", "release"],
            )
            publish.assert_awaited_once_with("req", 16, 2, 4)

        asyncio.run(run())

    def test_send_waits_for_embedding_published_by_encode(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 0
            encoder.req_states = {}
            state = encoder._acquire_encode_ref("req")
            state.embedding_data = EmbeddingData(
                "req",
                1,
                0,
                None,
                Modality.IMAGE,
                embedding_shape=[1, 1],
                dtype=torch.float32,
            )

            delivered = []

            async def send(current_state, destination):
                delivered.append(await encoder._wait_for_embedding(current_state))

            encoder.delivery = SimpleNamespace(
                send=AsyncMock(side_effect=send),
                release=AsyncMock(),
            )
            send_task = asyncio.create_task(
                encoder.send_to_destination(state, SendDestination("127.0.0.1:1"))
            )
            await asyncio.sleep(0)
            self.assertFalse(send_task.done())

            embedding = torch.ones((1, 1))
            encoder._stage_embedding(
                EmbeddingData(
                    "req",
                    1,
                    0,
                    None,
                    Modality.IMAGE,
                    embedding=embedding,
                )
            )
            await send_task
            await encoder._release_encode_ref(state)

            with patch.object(meta_registry, "discard", AsyncMock()):
                await encoder.release_request("req")

            self.assertEqual(len(delivered), 1)
            self.assertIs(delivered[0].embedding, embedding)

        asyncio.run(run())

    def test_release_waits_for_send_then_clears_embedding(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.req_states = {}

            send_started = asyncio.Event()
            finish_send = asyncio.Event()

            async def send(state, destination):
                send_started.set()
                await finish_send.wait()

            embedding_seen_by_release = []

            async def release(state):
                embedding_seen_by_release.append(state.embedding_data.embedding)

            encoder.delivery = SimpleNamespace(
                send=AsyncMock(side_effect=send),
                release=AsyncMock(side_effect=release),
            )
            embedding = torch.ones((1, 1))
            state = ReqState(
                "req",
                EmbeddingData("req", 1, 0, None, Modality.IMAGE, embedding=embedding),
            )
            state.embedding_ready.set()
            encoder.req_states[state.req_id] = state

            send_task = asyncio.create_task(
                encoder.send_to_destination(state, SendDestination("127.0.0.1:1"))
            )
            await send_started.wait()
            release_task = asyncio.create_task(encoder.release_request("req"))
            await asyncio.sleep(0)

            encoder.delivery.release.assert_not_awaited()
            self.assertIs(state.embedding_data.embedding, embedding)

            finish_send.set()
            await send_task
            await release_task

            encoder.delivery.release.assert_awaited_once_with(state)
            self.assertEqual(len(embedding_seen_by_release), 1)
            self.assertIs(embedding_seen_by_release[0], embedding)
            self.assertIsNone(state.embedding_data)
            self.assertNotIn("req", encoder.req_states)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
