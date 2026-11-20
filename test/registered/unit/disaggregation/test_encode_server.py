import asyncio
import pickle
import socket
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch
import zmq
import zmq.asyncio

import sglang.srt.disaggregation.encoder.server as encoder_server
from sglang.srt.disaggregation.encoder import http_server
from sglang.srt.disaggregation.encoder import runtime as encoder_runtime
from sglang.srt.disaggregation.encoder.preprocessor import (
    EncoderPreprocessor,
    EncoderPreprocessResult,
    _convert,
)
from sglang.srt.disaggregation.encoder.receiver import EmbeddingData
from sglang.srt.disaggregation.encoder.runtime import (
    _DP_RELEASE_AFTER_ENCODE,
    DPDispatcher,
    _retire_abandoned_encode,
    validate_encode_request,
)
from sglang.srt.disaggregation.encoder.server import (
    EncodeContext,
    EncoderMetaRegistry,
    MMEncoder,
    MMError,
    meta_registry,
)
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.managers.io_struct import unwrap_from_pickle
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.multimodal.encoder_preprocessing import EncoderMediaProcessorConfig
from sglang.srt.multimodal.modality import Modality
from sglang.srt.utils.serialization import safe_pickle_loads
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestEncoderDPErrorHandling(CustomTestCase):
    @staticmethod
    async def _run_registration_error(error):
        encoder = SimpleNamespace(
            register_embedding_destinations=AsyncMock(side_effect=error)
        )
        send = AsyncMock()
        request = {
            "req_id": "req",
            "receive_count": 1,
            "receive_url": "tcp://127.0.0.1:1",
        }
        with patch.object(encoder_runtime, "async_sock_send", send):
            await encoder_runtime._dp_worker_handle_request(
                encoder,
                None,
                object(),
                asyncio.Lock(),
                0,
                request,
                "register_destinations",
            )
        return unwrap_from_pickle(send.await_args.args[1])

    def test_worker_reports_third_party_exception_with_callable_code(self):
        class RpcLikeError(Exception):
            def code(self):
                return "INTERNAL"

        envelope = asyncio.run(
            self._run_registration_error(RpcLikeError("registration failed"))
        )
        self.assertEqual(envelope["_error"], "registration failed")
        self.assertEqual(envelope["_error_code"], 500)

    def test_worker_preserves_mm_error_status(self):
        envelope = asyncio.run(
            self._run_registration_error(MMError("bad destination", code=400))
        )
        self.assertEqual(envelope["_error_code"], 400)

    def test_dispatcher_drops_malformed_result_without_stopping_listener(self):
        async def run():
            dispatcher = encoder_runtime.DPDispatcher(
                dp_size=1,
                dispatch_sockets=[object()],
                release_sockets=[object()],
                result_socket=object(),
                worker_processes=[],
            )
            future = asyncio.get_running_loop().create_future()
            dispatcher.pending_futures[0]["req"] = future
            dispatcher.req_id_to_rank["req"] = 0
            valid = {"req_id": "req", "_dp_type": "encode", "content": None}
            recv = AsyncMock(
                side_effect=[
                    ["not", "an", "envelope"],
                    valid,
                    asyncio.CancelledError(),
                ]
            )

            with patch.object(encoder_runtime, "async_sock_recv", recv):
                listener = asyncio.create_task(dispatcher._result_listener())
                await asyncio.wait_for(future, timeout=1)
                listener.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await listener

            self.assertEqual(future.result(), valid)

        asyncio.run(run())


class TestEncodeRequestInputEdges(CustomTestCase):
    def test_scalar_hashes_are_rejected_before_dispatch(self):
        request = {
            "req_id": "r",
            "modality": "image",
            "mm_items": ["x"],
            "num_parts": 1,
            "part_idx": 0,
            "hashes": 5,
        }
        self.assertIsNotNone(validate_encode_request(request))

    def test_convert_leaves_an_empty_list_alone(self):
        self.assertEqual(_convert([]), [])

    def test_media_load_error_does_not_carry_the_payload(self):
        preprocessor = EncoderPreprocessor.__new__(EncoderPreprocessor)
        preprocessor.model_audio_sr = 16000
        payload = "A" * 100_000
        with patch(
            "sglang.srt.disaggregation.encoder.preprocessor.load_audio",
            side_effect=RuntimeError("decode failed"),
        ):
            with self.assertRaises(RuntimeError) as raised:
                preprocessor._load_single_item(payload, Modality.AUDIO)
        self.assertLess(len(str(raised.exception)), 1000)


class TestEncoderCacheIdentity(CustomTestCase):
    def test_direct_embedding_reuses_combined_identity(self):
        """A prefix-cache hit returns the embedding without re-encoding or rebuilding its key."""

        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.model_type = "qwen3_vl"
            encoder.cache_namespace = "sha256:" + "12" * 32
            encoder.preprocessor = EncoderPreprocessor.__new__(EncoderPreprocessor)
            encoder.preprocessor.model_type = encoder.model_type
            encoder.mm_cache = MultiModalStaticCache(4096)
            encoder.mm_cache_lock = asyncio.Lock()
            features = torch.arange(32, dtype=torch.float32).reshape(16, 2)
            grids = torch.tensor([[1, 2, 4], [1, 4, 2]])
            expected = torch.arange(8, dtype=torch.float32).reshape(4, 2)
            forward = Mock(return_value=expected)
            ctx = EncodeContext(
                req_id="cache-key-test",
                modality=Modality.IMAGE,
                preprocess_result=EncoderPreprocessResult(
                    mm_inputs={"pixel_values": features, "image_grid_thw": grids},
                    grid_thw=grids,
                    token_counts=[2, 2],
                ),
                get_feature_fn=forward,
                mm_feature=features,
                num_items=2,
                items_per_req=[2],
                aux_data={},
                str_mm_hashes=None,
                use_global_cache=False,
                is_health_check=False,
            )
            with patch.object(
                MultiModalStaticCache,
                "combine_hashes",
                wraps=MultiModalStaticCache.combine_hashes,
            ) as combine:
                for _ in range(2):
                    actual = await encoder._compute_direct_embedding(
                        ctx, keep_on_gpu=False
                    )
                    torch.testing.assert_close(actual, expected)
                self.assertEqual(combine.call_count, 2)
            forward.assert_called_once()
            self.assertEqual(len(encoder.mm_cache), 1)

        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(enable_prefix_mm_cache=True)
        override.install()
        try:
            asyncio.run(run())
        finally:
            override.restore()

    def test_grid_content_and_namespace_identify_each_embedding(self):
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.model_type = "qwen3_vl"
        encoder.cache_namespace = "sha256:" + "12" * 32
        encoder.preprocessor = EncoderPreprocessor.__new__(EncoderPreprocessor)
        encoder.preprocessor.model_type = encoder.model_type
        features = torch.zeros(16, 2)
        grids = torch.tensor([[1, 2, 4], [1, 4, 2]])
        result = EncoderPreprocessResult(
            mm_inputs={"pixel_values": features, "image_grid_thw": grids},
            grid_thw=grids,
            token_counts=[2, 2],
        )

        def keys():
            return encoder._calculate_hashes_from_features(
                features, result, Modality.IMAGE
            )

        original = keys()
        self.assertEqual([len(key) for key in original], [32, 32])
        self.assertNotEqual(original[0], original[1])
        self.assertEqual(original, keys())
        features[-1] = 1
        changed = keys()
        self.assertEqual(original[0], changed[0])
        self.assertNotEqual(original[1], changed[1])
        encoder.cache_namespace = "sha256:" + "34" * 32
        self.assertTrue(all(a != b for a, b in zip(changed, keys())))


class TestStagedNativeTransfer(CustomTestCase):
    @contextmanager
    def native_encoder(
        self,
        backend="zmq_to_scheduler",
        model_type="qwen3_vl",
        *,
        video_processor=True,
        video_decoder=True,
        send_sockets=(),
    ):
        from sglang.srt.disaggregation.encoder import grpc_server

        grids = [[1, 2, 2], [1, 4, 2]]
        if model_type == "kimi_vl":
            grids = [[2, 2], [4, 2]]
        elif model_type in {"inkling2", "muse_spark"}:
            grids = [[1, 1, 1], [1, 2, 1]]
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        enc = MMEncoder.__new__(MMEncoder)
        enc.model_type = model_type
        enc.model = SimpleNamespace(
            visual=Mock(),
            _encoder_processor=SimpleNamespace(temporal_compression_ratio=1),
        )
        enc.transfer_backend = backend
        enc.use_mooncake = enc.mm_global_cache_enabled = False
        enc.rank = 0
        enc.profiler = None
        enc.abandoned_req_ids = set()
        enc.req_states = {}
        enc.encode_dispatch_lock = asyncio.Lock()
        enc.preprocessor = EncoderPreprocessor.__new__(EncoderPreprocessor)
        enc.preprocessor.model_type = enc.model_type
        enc.preprocessor.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                vision_config=SimpleNamespace(merge_kernel_size=(2, 2))
            )
        )
        enc.preprocessor._supported_modalities = {Modality.IMAGE}
        if video_processor:
            enc.preprocessor._supported_modalities.add(Modality.VIDEO)
        enc.preprocessor.image_processor = SimpleNamespace(merge_size=2)
        enc.send_timeout = 1
        if backend == "nixl":
            enc.delivery = encoder_server.NixlDelivery(enc)
        else:
            enc.delivery = encoder_server.ZmqDelivery(enc, cleanup_receive_state=True)

        enc._embedding_dims = {Modality.IMAGE: 4, Modality.VIDEO: 4}
        enc._embedding_dtype = tensor.dtype
        enc._element_size = tensor.element_size()

        async def prepare(requests, modality, **kwargs):
            item_grids, counts, aux = grids, [1, 2], {}
            if model_type in {"kimi_k3", "kimi_k31"} and modality == Modality.IMAGE:
                aux["original_image_sizes"] = [[32, 48], [80, 27]]
            if modality == Modality.VIDEO:
                item_grids, counts = [[2, 2, 2], [3, 2, 2]], [2, 3]
                if model_type == "muse_spark":
                    item_grids = [[2, 1, 1], [3, 1, 1]]
                if model_type in {
                    "qwen3_vl",
                    "qwen3_vl_moe",
                    "qwen3_5",
                    "qwen3_5_moe",
                    "intern_s2_preview",
                    "glm4v",
                    "glm4v_moe",
                    "glm5_next",
                    "muse_spark",
                }:
                    aux["video_timestamps"] = [[0.0, 0.5], [0.0, 0.6, 1.2]]
                elif model_type == "mimo_v2":
                    aux["video_timestamps"] = [0.0, 0.5, 0.0, 0.6, 1.2]
                elif model_type == "minimax_m3_vl":
                    aux["video_timestamps"] = [[0.0, 0.5], None]
                elif model_type in {"qwen2_5_vl", "qwen3_omni_moe"}:
                    aux["second_per_grid_ts"] = torch.tensor([0.5, 0.75])
            ctx = SimpleNamespace(
                req_id=requests[0]["req_id"],
                items_per_req=[len(requests[0]["mm_items"])],
                preprocess_result=SimpleNamespace(
                    grid_thw=item_grids, token_counts=counts
                ),
                modality=modality,
                aux_data=aux,
            )
            # Publish the same early metadata as native preprocessing.
            await enc._publish_preprocess_metadata(ctx, requests)
            return ctx

        video_tensor = torch.arange(20, dtype=tensor.dtype).reshape(5, 4) + 100
        enc._compute_embedding = AsyncMock(
            side_effect=lambda ctx, **kwargs: (
                tensor if ctx.modality == Modality.IMAGE else video_tensor
            )
        )
        enc._prepare_encode_context_on_all_ranks = AsyncMock(side_effect=prepare)
        cfg = SimpleNamespace(encoder_transfer_backend=enc.transfer_backend)
        with (
            patch.object(encoder_server, "get_disagg", return_value=cfg),
            patch.object(
                grpc_server, "video_decoder_available", return_value=video_decoder
            ),
            patch.object(
                grpc_server,
                "get_parallel",
                return_value=SimpleNamespace(
                    tp_size=1 + len(send_sockets), num_dp_ranks=1, pp_size=1, nnodes=1
                ),
            ),
            patch.object(
                grpc_server,
                "get_model",
                return_value=SimpleNamespace(model_path="qwen"),
            ),
            patch.object(
                grpc_server,
                "get_serving",
                return_value=SimpleNamespace(
                    tokenizer_path=None, served_model_name="qwen-vl"
                ),
            ),
            zmq.Context() as enc.sync_context,
            ThreadPoolExecutor(4) as enc.executor,
            zmq.asyncio.Context() as receivers,
        ):
            svc = grpc_server.SGLangEncoderServer(enc, send_sockets, Mock())

            yield svc, enc, receivers, grids, tensor

    def test_cpu_tensor_is_staged_once_then_sent_to_late_receiver(self):
        from sglang.encoder.v1 import encoder_pb2 as pb
        from sglang.srt.managers.req_types import MultimodalDataItem

        async def run():
            with self.native_encoder() as (svc, enc, receivers, grids, tensor):
                transfer = (
                    await svc.Stage(
                        pb.StageRequest(request_id="request", mm_items=["a", "b"]),
                        Mock(),
                    )
                ).transfer
                self.assertEqual(
                    [x.identity for x in transfer.images],
                    [
                        MultimodalDataItem(
                            modality=Modality.IMAGE, precomputed_embeddings=chunk
                        ).cache_key.hex()
                        for chunk in [tensor[:1], tensor[1:]]
                    ],
                )
                self.assertEqual(list(enc.req_states), [transfer.part_id])
                with receivers.socket(zmq.PULL) as receiver:
                    port = receiver.bind_to_random_port("tcp://127.0.0.1")
                    await svc.Deliver(
                        pb.DeliverRequest(
                            transfer=transfer, receive_url=f"127.0.0.1:{port}"
                        ),
                        Mock(),
                    )
                    header, payload = await asyncio.wait_for(
                        receiver.recv_multipart(), 2
                    )
                    part = safe_pickle_loads(header)
                    self.assertEqual(
                        (part.req_id, part.part_idx, part.num_parts),
                        (transfer.part_id, 0, 1),
                    )
                    self.assertEqual(part.grid_dim, grids)
                    self.assertIsNone(part.embedding)
                    self.assertEqual(payload, tensor.numpy().tobytes())
                self.assertEqual(enc.req_states, {})
                self.assertEqual(enc._compute_embedding.await_count, 1)

                # Keep a port reserved with no listening peer. Native ZMQ must
                # abandon the send and retire its tensor without an RPC deadline.
                transfer = (
                    await svc.Stage(
                        pb.StageRequest(request_id="no-receiver", mm_items=["a", "b"]),
                        Mock(),
                    )
                ).transfer
                enc.send_timeout = 0.05
                context = AsyncMock()
                context.abort.side_effect = RuntimeError("delivery failed")
                with socket.socket() as unused:
                    unused.bind(("127.0.0.1", 0))
                    port = unused.getsockname()[1]
                    request = pb.DeliverRequest(
                        transfer=transfer, receive_url=f"127.0.0.1:{port}"
                    )
                    with self.assertRaisesRegex(RuntimeError, "delivery failed"):
                        await asyncio.wait_for(svc.Deliver(request, context), 2)
                self.assertEqual(enc.req_states, {})

        asyncio.run(run())

    def test_nixl_describes_and_writes_the_staged_tensor(self):
        from sglang.encoder.v1 import encoder_pb2 as pb
        from sglang.srt.disaggregation.encoder import nixl

        async def run():
            with self.native_encoder("nixl") as (svc, enc, _, _, tensor):
                response = await svc.Stage(
                    pb.StageRequest(request_id="request", mm_items=["a", "b"]), Mock()
                )
                transfer = response.transfer
                shape = await svc.Describe(
                    pb.DescribeRequest(transfer=transfer), Mock()
                )
                self.assertEqual(tuple(shape.shape), tuple(tensor.shape))
                self.assertEqual(shape.dtype, "float32")
                state = enc.req_states[transfer.part_id]

                async def inspect_write(source, descriptor, timeout):
                    self.assertEqual(state.active_sends, 1)
                    self.assertEqual(descriptor, b"destination")
                    self.assertEqual(timeout, enc.send_timeout)
                    torch.testing.assert_close(source, tensor)

                with patch.object(nixl, "write_embedding", new=inspect_write):
                    await svc.Deliver(
                        pb.DeliverRequest(
                            transfer=transfer, nixl_target=b"destination"
                        ),
                        Mock(),
                    )
                self.assert_retired(svc, enc, [state])
                enc._compute_embedding.assert_awaited_once()

        asyncio.run(run())

    def test_video_capability_checks_native_torchcodec_once(self):
        from sglang.srt.utils import video_decoder

        query_versions = Mock(side_effect=RuntimeError("cannot load libavformat"))
        with (
            patch.object(video_decoder, "_BACKEND", "torchcodec"),
            patch.dict(
                sys.modules,
                {
                    "torchcodec._core": SimpleNamespace(
                        get_ffmpeg_library_versions=query_versions
                    )
                },
            ),
        ):
            video_decoder.video_decoder_available.cache_clear()
            self.addCleanup(video_decoder.video_decoder_available.cache_clear)
            self.assertFalse(video_decoder.video_decoder_available())
            self.assertFalse(video_decoder.video_decoder_available())
            query_versions.assert_called_once_with()
            query_versions.side_effect = None
            video_decoder.video_decoder_available.cache_clear()
            self.assertTrue(video_decoder.video_decoder_available())

    def test_staging_images_without_a_usable_video_pipeline(self):
        from sglang.encoder.v1 import encoder_pb2 as pb

        async def run(processor, decoder):
            with self.native_encoder(
                video_processor=processor, video_decoder=decoder
            ) as (svc, _, _, _, _):
                info = await svc.GetEncoderInfo(pb.GetEncoderInfoRequest(), Mock())
                self.assertFalse(info.supports_staged_video)
                transfer = (
                    await svc.Stage(
                        pb.StageRequest(request_id="img", mm_items=["i1", "i2"]), Mock()
                    )
                ).transfer
                self.assertEqual(transfer.version, 1)
                await svc.Release(
                    pb.ReleaseRequest(transfer_id=transfer.transfer_id), Mock()
                )

        for processor, decoder in ((False, True), (True, False)):
            with self.subTest(processor=processor, decoder=decoder):
                asyncio.run(run(processor, decoder))

    def test_video_stage_matches_native_identities_and_packs_one_transfer(self):
        from sglang.encoder.v1 import encoder_pb2 as pb
        from sglang.srt.disaggregation.encoder import nixl
        from sglang.srt.managers.req_types import MultimodalDataItem

        async def run(backend, model, images):
            with self.native_encoder(backend, model) as (svc, enc, receivers, _, image):
                info = await svc.GetEncoderInfo(pb.GetEncoderInfoRequest(), Mock())
                self.assertTrue(info.supports_staged_video)
                transfer = (
                    await svc.Stage(
                        pb.StageRequest(
                            request_id="video", mm_items=images, video_urls=["a", "b"]
                        ),
                        Mock(),
                    )
                ).transfer
                self.assertEqual(
                    (transfer.version, transfer.image_count), (2, len(images))
                )
                self.assertEqual(len(transfer.images), len(images))
                self.assertEqual(list(enc.req_states), [transfer.part_id])
                self.assertEqual(len(svc._leases), 1)
                video = torch.arange(20, dtype=image.dtype).reshape(5, 4) + 100
                expected = torch.cat((image, video)) if images else video
                for i, (layout, tensor) in enumerate(
                    zip(transfer.videos, video.split([2, 3]))
                ):
                    self.assertEqual(
                        (layout.t, layout.h, layout.w), (tensor.shape[0], 2, 2)
                    )
                    if model in {
                        "qwen3_vl",
                        "qwen3_vl_moe",
                        "qwen3_5",
                        "qwen3_5_moe",
                        "intern_s2_preview",
                    }:
                        self.assertEqual(
                            list(layout.timestamps), [[0.0, 0.5], [0.0, 0.6, 1.2]][i]
                        )
                        chunks = tensor.split(1)
                    else:
                        self.assertEqual(list(layout.timestamps), [])
                        chunks = [tensor]
                    self.assertEqual(
                        list(layout.identities),
                        [
                            MultimodalDataItem(
                                modality=Modality.VIDEO, precomputed_embeddings=chunk
                            ).cache_key.hex()
                            for chunk in chunks
                        ],
                    )
                    self.assertEqual(
                        layout.HasField("second_per_grid_ts"),
                        model in {"qwen2_5_vl", "qwen3_omni_moe"},
                    )
                    if model in {"qwen2_5_vl", "qwen3_omni_moe"}:
                        self.assertEqual(layout.second_per_grid_ts, [0.5, 0.75][i])
                state = enc.req_states[transfer.part_id]
                if backend == "nixl":
                    spec = await svc.Describe(
                        pb.DescribeRequest(transfer=transfer), Mock()
                    )
                    self.assertEqual(tuple(spec.shape), tuple(expected.shape))
                    with patch.object(
                        nixl, "write_embedding", new_callable=AsyncMock
                    ) as write:
                        await svc.Deliver(
                            pb.DeliverRequest(transfer=transfer, nixl_target=b"target"),
                            Mock(),
                        )
                        torch.testing.assert_close(write.await_args.args[0], expected)
                else:
                    with receivers.socket(zmq.PULL) as sock:
                        port = sock.bind_to_random_port("tcp://127.0.0.1")
                        await svc.Deliver(
                            pb.DeliverRequest(
                                transfer=transfer, receive_url=f"127.0.0.1:{port}"
                            ),
                            Mock(),
                        )
                        header, data = await asyncio.wait_for(sock.recv_multipart(), 2)
                        envelope = safe_pickle_loads(header)
                        self.assertEqual(envelope.req_id, transfer.part_id)
                        self.assertEqual(tuple(envelope.shape), tuple(expected.shape))
                        self.assertEqual(data, expected.numpy().tobytes())
                self.assert_retired(svc, enc, [state])

        for model in (
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "intern_s2_preview",
            "qwen3_omni_moe",
        ):
            for backend in ("nixl", "zmq_to_scheduler"):
                for images in ([], ["i1", "i2"]):
                    with self.subTest(model=model, backend=backend, mixed=bool(images)):
                        asyncio.run(run(backend, model, images))

    def test_staged_native_model_layout(self):
        from sglang.encoder.v1 import encoder_pb2 as pb
        from sglang.srt.disaggregation.encoder.native_handoff import (
            NATIVE_IMAGE_MODEL_TYPES,
            NATIVE_VIDEO_MODEL_TYPES,
        )

        async def run(model, backend):
            with self.native_encoder(backend, model) as (svc, enc, _, _, _):
                info = await svc.GetEncoderInfo(pb.GetEncoderInfoRequest(), Mock())
                self.assertEqual(info.native_model_type, model)
                self.assertEqual(
                    info.supports_staged_video, model in NATIVE_VIDEO_MODEL_TYPES
                )
                videos = ["video1", "video2"] if info.supports_staged_video else []
                transfer = (
                    await svc.Stage(
                        pb.StageRequest(
                            request_id="native",
                            mm_items=["image1", "image2"],
                            video_urls=videos,
                        ),
                        Mock(),
                    )
                ).transfer
                self.assertEqual(transfer.version, 3)
                self.assertEqual(transfer.native_model_type, model)
                self.assertEqual(transfer.embedding_width, 4)
                self.assertEqual([x.embedding_rows for x in transfer.images], [1, 2])
                self.assertEqual([x.t for x in transfer.images], [1, 1])
                sizes = (
                    [[32, 48], [80, 27]]
                    if model in {"kimi_k3", "kimi_k31"}
                    else [[], []]
                )
                self.assertEqual(
                    [list(x.original_size) for x in transfer.images], sizes
                )
                if videos:
                    self.assertEqual(
                        [x.embedding_rows for x in transfer.videos], [2, 3]
                    )
                    self.assertEqual(
                        [len(x.identities) for x in transfer.videos],
                        [2, 3] if model == "mimo_v2" else [1, 1],
                    )
                    expected_times = [
                        [0.0, 0.5],
                        [] if model == "minimax_m3_vl" else [0.0, 0.6, 1.2],
                    ]
                    self.assertEqual(
                        [list(x.timestamps) for x in transfer.videos], expected_times
                    )
                else:
                    context = AsyncMock()
                    context.abort.side_effect = RuntimeError(
                        "staged video is unavailable"
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "staged video is unavailable"
                    ):
                        await svc.Stage(
                            pb.StageRequest(request_id="video", video_urls=["v"]),
                            context,
                        )
                await svc.Release(
                    pb.ReleaseRequest(transfer_id=transfer.transfer_id), Mock()
                )
                self.assertEqual(enc.req_states, {})

        for model in sorted(NATIVE_IMAGE_MODEL_TYPES):
            for backend in ("nixl", "zmq_to_scheduler"):
                with self.subTest(model=model, backend=backend):
                    asyncio.run(run(model, backend))

    def test_tp_stage_drains_cancelled_dispatch_before_next_modality(self):
        from sglang.encoder.v1 import encoder_pb2 as pb
        from sglang.srt.disaggregation.encoder import grpc_server

        async def run():
            started, resume = asyncio.Event(), asyncio.Event()
            calls = {1: [], 2: [], 0: []}
            with self.native_encoder(send_sockets=[1, 2]) as (svc, enc, *_):

                async def send(rank, wrapped):
                    request = unwrap_from_pickle(wrapped)
                    follower = Mock()

                    async def record(part, modality):
                        calls[rank].append((part["req_id"], modality))

                    follower.encode_staged_on_tp_ranks = record
                    await encoder_server._handle_encoder_worker_request(
                        follower, request
                    )
                    if rank == 1 and len(calls[rank]) == 1:
                        started.set()
                        await resume.wait()

                async def local(part, modality):
                    calls[0].append((part["req_id"], modality))
                    (result,) = await enc.batch_encode([part], modality)
                    return result

                enc.encode_staged_on_tp_ranks = local
                with patch.object(grpc_server, "async_sock_send", send):
                    first = asyncio.create_task(
                        svc.Stage(
                            pb.StageRequest(request_id="cancel", mm_items=["a", "b"]),
                            Mock(),
                        )
                    )
                    await asyncio.wait_for(started.wait(), 2)
                    first.cancel()
                    next_stage = asyncio.create_task(
                        svc.Stage(
                            pb.StageRequest(
                                request_id="mixed",
                                mm_items=["a", "b"],
                                video_urls=["c", "d"],
                            ),
                            Mock(),
                        )
                    )
                    await asyncio.sleep(0)
                    self.assertTrue(enc.encode_dispatch_lock.locked())
                    self.assertFalse(first.done())
                    self.assertFalse(calls[0] or calls[2])
                    resume.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await first
                    transfer = (await asyncio.wait_for(next_stage, 2)).transfer
                    self.assertEqual(calls[0], calls[1])
                    self.assertEqual(calls[0], calls[2])
                    self.assertEqual(
                        [x[1] for x in calls[0]],
                        [Modality.IMAGE, Modality.IMAGE, Modality.VIDEO],
                    )
                    self.assertEqual(set(enc.req_states), {transfer.part_id})
                    await svc.Release(
                        pb.ReleaseRequest(transfer_id=transfer.transfer_id), Mock()
                    )

        asyncio.run(run())

    def test_tp_follower_error_prevents_publication(self):
        from sglang.encoder.v1 import encoder_pb2 as pb
        from sglang.srt.disaggregation.encoder import grpc_server

        async def run():
            with self.native_encoder(send_sockets=[1]) as (svc, enc, *_):
                group = object()
                parallel = SimpleNamespace(tp_group=SimpleNamespace(cpu_group=group))
                context = AsyncMock()
                context.abort.side_effect = RuntimeError("tensor-parallel failure")

                def follower_status(status, *, op, group):
                    self.assertEqual(status.item(), 0)
                    self.assertIs(group, parallel.tp_group.cpu_group)
                    status.fill_(400)

                with (
                    patch.object(grpc_server, "async_sock_send", AsyncMock()),
                    patch.object(encoder_server, "get_parallel", return_value=parallel),
                    patch.object(torch.distributed, "all_reduce", follower_status),
                    self.assertRaisesRegex(RuntimeError, "tensor-parallel failure"),
                ):
                    await svc.Stage(
                        pb.StageRequest(request_id="failed", mm_items=["a", "b"]),
                        context,
                    )
                self.assertEqual(
                    context.abort.await_args.args[0],
                    grpc_server.grpc.StatusCode.INVALID_ARGUMENT,
                )
                self.assertFalse(svc._leases or enc.req_states or svc._active)

        asyncio.run(run())

    def test_failed_tp_dispatch_fences_the_group(self):
        from sglang.srt.disaggregation.encoder import grpc_server

        async def run():
            with self.native_encoder(send_sockets=[1, 2]) as (svc, enc, *_):
                enc.encode_staged_on_tp_ranks = AsyncMock()
                svc._on_dispatch_failure = Mock()
                send = AsyncMock(side_effect=[None, RuntimeError("broken IPC")])
                with patch.object(grpc_server, "async_sock_send", send):
                    with self.assertRaisesRegex(RuntimeError, "broken IPC"):
                        await svc._dispatch_staged({}, Modality.VIDEO)
                    with self.assertRaisesRegex(
                        RuntimeError, "dispatch is unavailable"
                    ):
                        await svc._dispatch_encode({})
                self.assertEqual(send.await_count, 2)
                svc._on_dispatch_failure.assert_called_once()
                enc.encode_staged_on_tp_ranks.assert_not_awaited()

        asyncio.run(run())

    def test_mimo_rejects_video_with_linked_audio(self):
        with self.native_encoder(model_type="mimo_v2") as (svc, *_):
            self.assertRaisesRegex(
                ValueError,
                "without an audio track",
                svc._video_layouts,
                SimpleNamespace(video_audio_feature_lens=torch.tensor([8])),
            )

    def test_unrecognized_model_does_not_advertise_staging(self):
        from sglang.encoder.v1 import encoder_pb2 as pb

        async def run():
            with self.native_encoder(model_type="unregistered_model") as (svc, *_):
                context = AsyncMock()
                context.abort.side_effect = RuntimeError("model lacks staged adapter")
                with self.assertRaisesRegex(RuntimeError, "model lacks staged adapter"):
                    await svc.GetEncoderInfo(pb.GetEncoderInfoRequest(), context)

        asyncio.run(run())

    def test_failed_or_cancelled_mixed_stage_retires_both_native_parts(self):
        from sglang.encoder.v1 import encoder_pb2 as pb

        async def run(cancel):
            with self.native_encoder() as (svc, enc, _, _, _):
                original_compute = enc._compute_embedding
                enc.mm_global_cache_enabled = True
                enc.mm_global_cache = Mock()
                enc._lookup_global_cache = AsyncMock(return_value=([], []))
                started, resume = asyncio.Event(), asyncio.Event()
                states = []

                async def compute(ctx, **kwargs):
                    if ctx.modality == Modality.VIDEO:
                        states.extend(enc.req_states.values())
                        started.set()
                        if cancel:
                            await resume.wait()
                        raise RuntimeError("video forward failed")
                    return await original_compute(ctx, **kwargs)

                enc._compute_embedding = compute
                request = pb.StageRequest(
                    request_id="abort", mm_items=["i1", "i2"], video_urls=["v1", "v2"]
                )
                context = AsyncMock()
                context.abort.side_effect = RuntimeError("video forward failed")
                call = asyncio.create_task(svc.Stage(request, context))
                await asyncio.wait_for(started.wait(), 2)
                if cancel:
                    call.cancel()
                    await asyncio.sleep(0)
                    self.assertEqual(svc._active, 1)
                    resume.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await call
                else:
                    with self.assertRaisesRegex(RuntimeError, "video forward failed"):
                        await call
                self.assertEqual(len(states), 2)
                self.assertCountEqual(
                    [
                        c.args[0]
                        for c in enc.mm_global_cache.discard_prefetch_waiter.call_args_list
                    ],
                    [state.req_id for state in states],
                )
                self.assert_retired(svc, enc, states)

        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                asyncio.run(run(cancel))

    def assert_retired(self, svc, enc, states):
        self.assertEqual(svc._leases, {})
        self.assertEqual(svc._active, 0)
        self.assertFalse(svc._cleanup_tasks)
        self.assertFalse(enc.has_pending_embeddings())
        for state in states:
            self.assertIsNone(state.embedding_data)
            self.assertEqual((state.active_encodes, state.active_sends), (0, 0))
            rid = state.req_id
            for registry in (
                enc.abandoned_req_ids,
                encoder_server.meta_registry._rid_to_meta,
                encoder_server.meta_registry._pending_at,
                encoder_server.rid_to_cond,
                encoder_server.rid_to_receive_endpoint,
                encoder_server.rid_to_receive_count,
            ):
                self.assertNotIn(rid, registry)

    def test_competing_stage_deliver_abort_release_and_expiry(self):
        from sglang.encoder.v1 import encoder_pb2 as pb

        async def run():
            with self.native_encoder() as (svc, enc, receivers, _, _):
                states = []

                async def stage():
                    t = (
                        await svc.Stage(
                            pb.StageRequest(
                                request_id="same-public-id", mm_items=["a", "b"]
                            ),
                            Mock(),
                        )
                    ).transfer
                    self.assertIn(t.part_id, meta_registry._rid_to_meta)
                    self.assertIn(t.part_id, encoder_server.rid_to_cond)
                    states.append(enc.req_states[t.part_id])
                    return t

                async def release(t):
                    await svc.Release(
                        pb.ReleaseRequest(transfer_id=t.transfer_id), Mock()
                    )

                # Use one public ID for every request. All native state must
                # remain isolated by the minted part ID.
                svc._ttl = 5
                initial = await asyncio.gather(*(stage() for _ in range(24)))
                svc._ttl = 0.05
                expiring = await asyncio.gather(*(stage() for _ in range(8)))
                svc._ttl = 5
                self.assertEqual(len(states), 32)

                aborting = {t.part_id for t in initial[8:16]}
                entered = set()
                sending = asyncio.Event()
                native_send = enc.delivery.send

                async def send(state, dest):
                    if state.req_id in aborting:
                        self.assertEqual(state.active_sends, 1)
                        entered.add(state.req_id)
                        if entered == aborting:
                            sending.set()
                        await asyncio.Event().wait()
                    await native_send(state, dest)

                async def fresh():
                    await release(await stage())

                enc.delivery.send = send
                with receivers.socket(zmq.PULL) as pull:
                    port = pull.bind_to_random_port("tcp://127.0.0.1")

                    def deliver(t):
                        return svc.Deliver(
                            pb.DeliverRequest(
                                transfer=t, receive_url=f"127.0.0.1:{port}"
                            ),
                            Mock(),
                        )

                    aborted = [asyncio.create_task(deliver(t)) for t in initial[8:16]]
                    mixed = asyncio.gather(
                        *(deliver(t) for t in initial[:8]),
                        *(release(t) for t in initial[16:]),
                        *(fresh() for _ in range(8)),
                    )
                    await asyncio.wait_for(sending.wait(), 2)
                    # Release on an already claimed handle cannot free the
                    # underlying tensor while the native send is using it.
                    await asyncio.gather(*(release(t) for t in initial[8:16]))
                    for t, task in zip(initial[8:16], aborted):
                        self.assertEqual(enc.req_states[t.part_id].active_sends, 1)
                        task.cancel()
                    errors = await asyncio.gather(*aborted, return_exceptions=True)
                    self.assertTrue(
                        all(isinstance(e, asyncio.CancelledError) for e in errors)
                    )
                    await mixed
                    received = []
                    for _ in range(8):
                        header, _ = await asyncio.wait_for(pull.recv_multipart(), 2)
                        received.append(safe_pickle_loads(header).req_id)
                    self.assertCountEqual(received, [t.part_id for t in initial[:8]])

                async def drained():
                    while svc._active or svc._cleanup_tasks:
                        await asyncio.sleep(0.001)

                await asyncio.wait_for(drained(), 2)
                self.assertEqual(enc._compute_embedding.await_count, 40)
                self.assertEqual(len({t.part_id for t in initial + expiring}), 32)
                self.assertEqual(len(states), 40)
                self.assert_retired(svc, enc, states)

        asyncio.run(run())

    def test_queued_and_preprocessing_stages_have_a_server_deadline(self):
        import grpc

        from sglang.encoder.v1 import encoder_pb2 as pb

        async def run():
            with self.native_encoder() as (svc, enc, _, _, _):
                prepared = asyncio.Event()
                states = []
                native_prepare = enc._prepare_encode_context_on_all_ranks

                async def stalled(*args, **kwargs):
                    await native_prepare(*args, **kwargs)
                    states.extend(enc.req_states.values())
                    prepared.set()
                    await asyncio.Event().wait()

                enc._prepare_encode_context_on_all_ranks = stalled
                svc._stage_timeout = 0.1

                async def time_out():
                    context = AsyncMock()
                    context.abort.side_effect = RuntimeError("stage timeout")
                    with self.assertRaisesRegex(RuntimeError, "stage timeout"):
                        await svc.Stage(
                            pb.StageRequest(request_id="timeout", mm_items=["a", "b"]),
                            context,
                        )
                    context.abort.assert_awaited_once_with(
                        grpc.StatusCode.DEADLINE_EXCEEDED, "image staging timed out"
                    )

                async with enc.encode_dispatch_lock:
                    # Timeout before entering encode must leave no native marker.
                    await asyncio.wait_for(time_out(), 1)
                    self.assertFalse(prepared.is_set())
                await asyncio.wait_for(time_out(), 1)
                self.assertEqual(len(states), 1)
                self.assertTrue(prepared.is_set())
                self.assertFalse(enc.encode_dispatch_lock.locked())
                self.assertEqual(enc._compute_embedding.await_count, 0)
                self.assert_retired(svc, enc, states)

                # A remote key lookup can stall before the cache owns copies.
                # It obeys the staging deadline even with the global cache on.
                from sglang.srt.mem_cache.embedding_cache_controller import (
                    EmbeddingCacheController,
                )

                enc.mm_global_cache_enabled = True
                cache = EmbeddingCacheController.__new__(EmbeddingCacheController)
                cache.lock = threading.Lock()
                cache.ongoing_prefetch = {}
                enc.mm_global_cache = cache
                enc._prepare_encode_context_on_all_ranks = native_prepare

                async def stalled_lookup(ctx):
                    states.append(enc.req_states[ctx.req_id])
                    await asyncio.Event().wait()

                enc._lookup_global_cache = stalled_lookup
                await asyncio.wait_for(time_out(), 1)
                self.assertEqual(len(states), 2)
                self.assert_retired(svc, enc, states)

                # Conversely a cache transfer already holding native references
                # must reach its safe point before its waiter/part can retire.
                started, resume = asyncio.Event(), asyncio.Event()

                async def in_flight_cache(ctx, **kwargs):
                    states.append(enc.req_states[ctx.req_id])
                    cache.ongoing_prefetch[ctx.req_id] = object()
                    started.set()
                    await resume.wait()

                enc._compute_embedding = in_flight_cache
                enc._lookup_global_cache = AsyncMock(return_value=([], [0, 1]))
                call = asyncio.create_task(
                    svc.Stage(
                        pb.StageRequest(request_id="aborted", mm_items=["a", "b"]),
                        Mock(),
                    )
                )
                await asyncio.wait_for(started.wait(), 1)
                call.cancel()
                await asyncio.sleep(0)
                self.assertEqual(svc._active, 1)
                self.assertEqual(states[-1].active_encodes, 1)
                resume.set()
                with self.assertRaises(asyncio.CancelledError):
                    await call
                self.assertEqual(cache.ongoing_prefetch, {})
                self.assert_retired(svc, enc, states)

        asyncio.run(run())


class TestEncoderMetaRegistry(CustomTestCase):
    def test_stale_releases_do_not_block_each_other(self):
        async def run():
            registry = EncoderMetaRegistry(wait_timeout=1, sweep_timeout=1)
            blocked_started = asyncio.Event()
            unblock = asyncio.Event()
            fast_released = asyncio.Event()

            async def release(req_id):
                if req_id == "blocked":
                    blocked_started.set()
                    await unblock.wait()
                else:
                    fast_released.set()

            registry.on_release = release
            registry._pending_at.update(blocked=0, fast=0)

            blocked_task = registry._schedule_stale_release("blocked")
            await asyncio.wait_for(blocked_started.wait(), timeout=1)
            fast_task = registry._schedule_stale_release("fast")
            await asyncio.wait_for(fast_released.wait(), timeout=1)
            await fast_task

            self.assertIn("blocked", registry._pending_at)
            self.assertNotIn("fast", registry._pending_at)

            unblock.set()
            await blocked_task
            self.assertNotIn("blocked", registry._pending_at)

        asyncio.run(run())

    def test_failed_stale_release_is_retried(self):
        async def run():
            registry = EncoderMetaRegistry(wait_timeout=1, sweep_timeout=1)
            attempts = 0

            async def release(_req_id):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("transient cleanup failure")

            registry.on_release = release
            registry._pending_at["req"] = 0

            await registry._release_stale("req")
            self.assertIn("req", registry._pending_at)
            retry_at = registry._pending_at["req"]
            self.assertGreater(retry_at, 0)

            await registry._release_stale("req")
            self.assertNotIn("req", registry._pending_at)
            self.assertEqual(attempts, 2)

        asyncio.run(run())

    def test_send_retries_do_not_release_before_all_destinations_finish(self):
        async def run():
            registry = EncoderMetaRegistry(wait_timeout=1, sweep_timeout=1)
            released = AsyncMock()
            registry.on_release = released

            await registry.note_send_done("req", 2, "10.0.0.1:5000")
            await registry.note_send_done("req", 2, "10.0.0.1:5000")
            released.assert_not_awaited()

            await registry.note_send_done("req", 2, "10.0.0.2:5000")
            released.assert_awaited_once_with("req")

        asyncio.run(run())

    def test_http_send_counts_the_normalized_destination(self):
        async def run():
            send = AsyncMock(return_value=True)
            note_send_done = AsyncMock()
            request = {
                "req_id": "req",
                "prefill_host": "127.0.0.1",
                "embedding_port": 5000,
                "session_id": "session",
                "buffer_address": 1234,
                "receive_count": 2,
            }
            with (
                patch.object(http_server, "dp_dispatcher", None),
                patch.object(http_server, "encoder", SimpleNamespace(send=send)),
                patch.object(
                    encoder_server.meta_registry,
                    "note_send_done",
                    note_send_done,
                ),
            ):
                response = await http_server.handle_send_request(request)

            self.assertEqual(response.status_code, 200)
            note_send_done.assert_awaited_once_with("req", 2, "127.0.0.1:5000")

        asyncio.run(run())

    def test_dp_send_counts_the_normalized_destination(self):
        async def run():
            encoder = SimpleNamespace(send=AsyncMock(return_value=True))
            note_send_done = AsyncMock()
            request = {
                "req_id": "req",
                "prefill_host": "127.0.0.1",
                "embedding_port": 5000,
                "session_id": "session",
                "buffer_address": 1234,
                "receive_count": 2,
            }
            with (
                patch.object(encoder_runtime, "async_sock_send", AsyncMock()),
                patch.object(
                    encoder_server.meta_registry,
                    "note_send_done",
                    note_send_done,
                ),
            ):
                await encoder_runtime._dp_worker_handle_request(
                    encoder,
                    None,
                    object(),
                    asyncio.Lock(),
                    0,
                    request,
                    "send",
                )

            note_send_done.assert_awaited_once_with("req", 2, "127.0.0.1:5000")

        asyncio.run(run())


class TestEncoderPreprocessorOffEventLoop(CustomTestCase):
    @staticmethod
    def _make_preprocessor():
        preprocessor = EncoderPreprocessor.__new__(EncoderPreprocessor)
        preprocessor.model_type = "minimax_m3_vl"
        preprocessor.image_processor = None
        preprocessor.use_image_processor_gpu = False
        preprocessor.vision_config = {"image": {}, "video": {}}
        preprocessor.encoder_media_processor_config = EncoderMediaProcessorConfig(
            preserve_media_metadata=True, preprocess_off_event_loop=True
        )
        preprocessor.io_executor = ThreadPoolExecutor(max_workers=2)
        preprocessor.preproc_executor = ThreadPoolExecutor(max_workers=1)
        return preprocessor

    @staticmethod
    def _run_with_loop_handshake(make_awaitable):
        """The hook blocks until a coroutine on the event loop answers it, so a hook
        that holds the loop thread starves that coroutine and times out."""
        started, answered = threading.Event(), threading.Event()

        def wait_for_loop():
            started.set()
            if not answered.wait(timeout=5):
                raise TimeoutError("the preprocess hook blocked the event loop")

        async def answer():
            while not started.is_set():
                await asyncio.sleep(0.001)
            answered.set()

        async def run():
            responder = asyncio.create_task(answer())
            result = await make_awaitable(wait_for_loop)
            await asyncio.wait_for(responder, timeout=5)
            return result

        return asyncio.run(run())

    def test_glm_video_timestamps_match_native_prompt(self):
        preprocessor = self._make_preprocessor()
        preprocessor.model_type = "glm4v"
        preprocessor._flatten_and_load_videos = AsyncMock(
            return_value=(
                [object()] * 3,
                {
                    "video_metadata": [
                        {"fps": 10, "frames_indices": [0, 1, 4, 5, 8]},
                        {"fps": 8, "frames_indices": [2, 3]},
                        {"fps": None, "frames_indices": []},
                    ]
                },
            )
        )
        preprocessor.video_processor = Mock(
            return_value={
                "video_grid_thw": torch.tensor([[2, 2, 2], [3, 2, 2], [1, 2, 2]])
            }
        )
        with preprocessor.preproc_executor, preprocessor.io_executor:
            output = asyncio.run(
                preprocessor._process_video_items(["a", "b", "c"], None)
            )
        self.assertEqual(output["video_timestamps"], [[0.0, 0.4], [0.25] * 3, [0.0]])

    def test_video_hook_runs_off_the_event_loop(self):
        preprocessor = self._make_preprocessor()
        preprocessor._load_single_item = lambda data, modality: "decoded"

        def process(wait_for_loop):
            def hook(videos, modality, config):
                wait_for_loop()
                return {"videos": videos}

            return preprocessor._process_video_items([{"url": "a.mp4"}], hook)

        output = self._run_with_loop_handshake(process)
        self.assertEqual(output, {"videos": ["decoded"]})

    def test_image_hook_runs_off_the_event_loop(self):
        preprocessor = self._make_preprocessor()
        preprocessor._flatten_and_load_images = AsyncMock(
            return_value=[torch.zeros(3, 4, 4)]
        )
        grid = torch.tensor([[1, 2, 2]])

        def process(wait_for_loop):
            def hook(images, modality, config):
                wait_for_loop()
                return {"image_grid_thw": grid}

            return preprocessor._process_image_items(["img"], hook)

        output = self._run_with_loop_handshake(process)
        self.assertIs(output["image_grid_thw"], grid)


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

    def test_inkling2_image_tokens_are_the_grid_cells(self):
        grid = torch.tensor([1, 3, 4])
        preprocessor = self._make_preprocessor("inkling2")
        self.assertEqual(preprocessor.get_num_patches(grid, Modality.IMAGE), 12)
        self.assertEqual(preprocessor.get_num_tokens(grid, Modality.IMAGE), 12)

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


class TestEncoderDPAbandonedRequest(CustomTestCase):
    @staticmethod
    def _make_dispatcher():
        return DPDispatcher(
            dp_size=1,
            dispatch_sockets=[object()],
            release_sockets=[object()],
            result_socket=object(),
            worker_processes=[],
        )

    def test_dispatch_timeout_notifies_worker_to_release(self):
        async def run():
            dispatcher = self._make_dispatcher()
            sent = []
            release_sent = asyncio.Event()

            async def send(socket, payload):
                message = unwrap_from_pickle(payload)
                sent.append((socket, message))
                if message.get("_dp_type") == _DP_RELEASE_AFTER_ENCODE:
                    release_sent.set()

            request = {"req_id": "timed-out", "modality": "image"}
            with (
                patch(
                    "sglang.srt.disaggregation.encoder.runtime.async_sock_send",
                    side_effect=send,
                ),
                patch(
                    "sglang.srt.disaggregation.encoder.runtime.server_module.ENCODER_REQ_TIMEOUT",
                    0.01,
                ),
            ):
                result = await dispatcher.dispatch(request)
                await asyncio.wait_for(release_sent.wait(), timeout=1)

            self.assertEqual(result["_error_type"], "TimeoutError")
            self.assertIs(sent[0][0], dispatcher.dispatch_sockets[0])
            self.assertEqual(sent[0][1], request)
            self.assertIs(sent[1][0], dispatcher.release_sockets[0])
            self.assertEqual(
                sent[1][1],
                {
                    "_dp_type": _DP_RELEASE_AFTER_ENCODE,
                    "req_id": "timed-out",
                },
            )
            self.assertEqual(dispatcher.pending_counts, [0])
            self.assertNotIn("timed-out", dispatcher.req_id_to_rank)

        asyncio.run(run())

    def test_dispatch_cancellation_notifies_worker_to_release(self):
        async def run():
            dispatcher = self._make_dispatcher()
            encode_sent = asyncio.Event()
            release_sent = asyncio.Event()

            async def send(_socket, payload):
                message = unwrap_from_pickle(payload)
                if message.get("_dp_type") == _DP_RELEASE_AFTER_ENCODE:
                    release_sent.set()
                else:
                    encode_sent.set()

            with patch(
                "sglang.srt.disaggregation.encoder.runtime.async_sock_send",
                side_effect=send,
            ):
                task = asyncio.create_task(
                    dispatcher.dispatch({"req_id": "cancelled", "modality": "image"})
                )
                await encode_sent.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.wait_for(release_sent.wait(), timeout=1)

            self.assertEqual(dispatcher.pending_counts, [0])
            self.assertNotIn("cancelled", dispatcher.req_id_to_rank)

        asyncio.run(run())

    def test_worker_marks_running_encode_abandoned(self):
        async def run():
            async def encode():
                await asyncio.Event().wait()

            encode_task = asyncio.create_task(encode())
            encoder = SimpleNamespace(
                abandon_request=AsyncMock(),
                release_request=AsyncMock(),
            )
            await _retire_abandoned_encode(encoder, encode_task, "abandoned")

            encoder.abandon_request.assert_awaited_once_with("abandoned")
            encoder.release_request.assert_not_awaited()
            encode_task.cancel()
            await asyncio.gather(encode_task, return_exceptions=True)

        asyncio.run(run())

    def test_worker_preserves_release_before_encode_task_exists(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 0
            encoder.req_states = {}
            encoder.abandoned_req_ids = set()
            encoder.delivery = SimpleNamespace(release=AsyncMock())

            await _retire_abandoned_encode(encoder, None, "abandoned")
            self.assertIn("abandoned", encoder.abandoned_req_ids)

            state = encoder._acquire_encode_ref("abandoned")
            self.assertTrue(state.release_requested)
            with patch.object(meta_registry, "discard", AsyncMock()):
                await encoder._release_encode_ref(state)

            encoder.delivery.release.assert_awaited_once_with(state)
            self.assertNotIn("abandoned", encoder.req_states)

        asyncio.run(run())

    def test_worker_release_survives_encode_failure(self):
        async def run():
            async def encode():
                raise RuntimeError("bad image")

            encode_task = asyncio.create_task(encode())
            await asyncio.sleep(0)
            encoder = SimpleNamespace(
                abandon_request=AsyncMock(),
                release_request=AsyncMock(),
            )
            await _retire_abandoned_encode(
                encoder,
                encode_task,
                "failed",
            )

            encoder.release_request.assert_awaited_once_with("failed")
            encoder.abandon_request.assert_not_awaited()
            await asyncio.gather(encode_task, return_exceptions=True)

        asyncio.run(run())


class TestMooncakeRegistration(CustomTestCase):
    def setUp(self):
        self.engine = MooncakeTransferEngine.__new__(MooncakeTransferEngine)
        self.engine.engine = unittest.mock.Mock()
        self.engine.ib_device = "mlx5_0"

    def test_register_raises_on_nonzero_status(self):
        self.engine.engine.register_memory.return_value = -1

        with self.assertRaisesRegex(RuntimeError, "registration failed.*ret=-1"):
            self.engine.register(1234, 4096)

    def test_deregister_preserves_backend_failure(self):
        backend_error = OSError("backend failed")
        self.engine.engine.unregister_memory.side_effect = backend_error

        with self.assertRaisesRegex(RuntimeError, "deregistration failed") as ctx:
            self.engine.deregister(1234)

        self.assertIs(ctx.exception.__cause__, backend_error)

    def test_batch_register_raises_on_nonzero_status(self):
        self.engine.engine.batch_register_memory.return_value = -202

        with self.assertRaises(RuntimeError):
            self.engine.batch_register([0x10000], [0x1000])

    def test_batch_register_tolerates_already_registered(self):
        self.engine.engine.batch_register_memory.return_value = -7

        with self.assertLogs(level="ERROR"):
            self.engine.batch_register([0x10000], [0x1000])


if __name__ == "__main__":
    unittest.main()
