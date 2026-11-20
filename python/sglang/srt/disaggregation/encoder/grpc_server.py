"""
gRPC Encoder Server for SGLang EPD (Encode-Prefill-Decode) mode.

This server provides gRPC-based encoding for multimodal inputs.

Usage:
    python -m sglang.launch_server --model-path <model> --encoder-only --grpc-mode
"""

import asyncio
import logging
import math
import multiprocessing as mp
import traceback
from concurrent import futures
from http import HTTPStatus
from itertools import pairwise
from typing import List

import grpc
import torch
import zmq
import zmq.asyncio
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from sglang.encoder.v1 import encoder_pb2, encoder_pb2_grpc
from sglang.srt.disaggregation.encoder.native_handoff import (
    FRAME_VIDEO_MODEL_TYPES,
    GLM_VIDEO_MODEL_TYPES,
    NATIVE_IMAGE_MODEL_TYPES,
    NATIVE_VIDEO_MODEL_TYPES,
    STAGED_VIDEO_MODEL_TYPES,
    TIMED_VIDEO_MODEL_TYPES,
)
from sglang.srt.disaggregation.encoder.receiver import create_part_req_id
from sglang.srt.disaggregation.encoder.runtime import validate_encode_request
from sglang.srt.disaggregation.encoder.server import (
    MMEncoder,
    SendDestination,
    await_task_completion_on_cancel,
    launch_encoder,
    meta_registry,
)
from sglang.srt.disaggregation.encoder.staged import StagedImageService
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import async_sock_send, wrap_as_pickle
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.multimodal.media_io import media_url_security_config, media_url_timeout
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import (
    get_disagg,
    get_model,
    get_parallel,
    get_serving,
    publish,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import random_uuid
from sglang.srt.utils.network import NetworkAddress, get_zmq_socket
from sglang.srt.utils.video_decoder import video_decoder_available

logger = logging.getLogger(__name__)
SGLangEncoderServicer = encoder_pb2_grpc.SglangEncoderServicer
add_SGLangEncoderServicer_to_server = (
    encoder_pb2_grpc.add_SglangEncoderServicer_to_server
)


class EncoderHealthServicer(health_pb2_grpc.HealthServicer):
    """
    Standard gRPC health check service for encoder server.
    Implements grpc.health.v1.Health for Kubernetes probes.
    """

    OVERALL_SERVER = ""
    ENCODER_SERVICE = "sglang.grpc.encoder.SglangEncoder"

    def __init__(self):
        self._serving = False

    def set_serving(self):
        self._serving = True

    def set_not_serving(self):
        self._serving = False

    async def Check(self, request, context) -> health_pb2.HealthCheckResponse:
        if self._serving:
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.SERVING
            )
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.NOT_SERVING
        )

    async def Watch(self, request, context):
        yield await self.Check(request, context)


class SGLangEncoderServer(StagedImageService, SGLangEncoderServicer):
    """
    gRPC service implementation for SGLang encoder.
    """

    def __init__(
        self,
        encoder: MMEncoder,
        send_sockets: List[zmq.Socket],
        server_args: ServerArgs,
        on_dispatch_failure=None,
    ):
        self.encoder = encoder
        self._on_dispatch_failure = on_dispatch_failure
        self._dispatch_failed = False
        self.send_sockets = send_sockets
        self.server_args = server_args
        info = None
        native = encoder.model_type in NATIVE_IMAGE_MODEL_TYPES and (
            encoder.model_type != "mimo_v2"
            or getattr(encoder.model, "visual", None) is not None
        )
        qwen = encoder.model_type in STAGED_VIDEO_MODEL_TYPES
        parallel = get_parallel()
        if (
            encoder.transfer_backend in ("zmq_to_scheduler", "nixl")
            and parallel.num_dp_ranks == parallel.pp_size == parallel.nnodes == 1
            and len(send_sockets) == parallel.tp_size - 1
            and (qwen or native)
            and encoder.preprocessor.supports_modality(Modality.IMAGE)
        ):
            model = get_model().model_path
            domains, max_mb, _, _ = media_url_security_config()
            # With no configured timeout, media I/O uses per-modality defaults.
            remote_timeout = media_url_timeout(0) or None
            info = encoder_pb2.GetEncoderInfoResponse(
                model_path=model,
                tokenizer_path=get_serving().tokenizer_path or model,
                served_model_name=get_serving().served_model_name or model,
                native_model_type=encoder.model_type if native else "",
                supports_staged_video=(
                    (qwen or encoder.model_type in NATIVE_VIDEO_MODEL_TYPES)
                    and encoder.preprocessor.supports_modality(Modality.VIDEO)
                    and video_decoder_available()
                ),
                staged_media_policy=encoder_pb2.StagedMediaPolicy(
                    allowed_domains=domains,
                    max_bytes=max_mb * 1024 * 1024,
                    timeout_secs=remote_timeout,
                ),
            )
        nixl_enabled = encoder.transfer_backend == "nixl"
        super().__init__(
            encode=self._stage_media,
            send=None if nixl_enabled else self._deliver_part,
            describe=self._describe_part,
            write=self._write_part if nixl_enabled else None,
            release=self._release_staged_part,
            make_part_id=create_part_req_id,
            info=info,
            stage_timeout=envs.SGLANG_ENCODER_REQ_TIMEOUT.get(),
            ttl=encoder.send_timeout,
            capacity=envs.SGLANG_ENCODER_MAX_STAGED_REQUESTS.get(),
        )

    async def _stage_media(self, part_id, images, videos):
        # The existing encoder batches one modality at a time. Only the root
        # part is published, so both transports can deliver a single tensor.
        async with self.encoder.encode_dispatch_lock:
            if not images:
                video = await self._encode_modality(part_id, videos, Modality.VIDEO)
                return [], self._video_layouts(video)

            image = await self._encode_modality(part_id, images, Modality.IMAGE)
            image_layouts = self._image_layouts(image)
            if not videos:
                return image_layouts, []

            video_id = f"{part_id}_video"
            try:
                video = await self._encode_modality(video_id, videos, Modality.VIDEO)
                video_layouts = self._video_layouts(video)
                if image.dtype != video.dtype or image.shape[1:] != video.shape[1:]:
                    raise RuntimeError("Incompatible image and video embedding formats")
                packed = torch.cat((image.embedding, video.embedding))
                image.embedding = packed
                image.shape = list(packed.shape)
                await meta_registry.publish(part_id, packed.nbytes, *packed.shape)
            finally:
                # Cancellation can arrive after preprocessing published the
                # temporary video state, or during its global-cache cleanup.
                await await_task_completion_on_cancel(
                    asyncio.create_task(self._release_staged_part(video_id)),
                    f"Retire staged video {video_id}",
                )
            return image_layouts, video_layouts

    async def _encode_modality(self, part_id, items, modality):
        request = {
            "modality": modality.name,
            "req_id": part_id,
            "num_parts": 1,
            "part_idx": 0,
            "mm_items": items,
        }
        if self.send_sockets:
            result = await await_task_completion_on_cancel(
                asyncio.create_task(self._dispatch_staged(request, modality)),
                f"Staged TP encode {part_id}",
            )
        else:
            (result,) = await self.encoder.batch_encode(
                [request], modality, drain_global_cache_on_cancel=True
            )
        _, _, _, error_msg, error_code = result
        if error_msg is not None:
            if error_code == HTTPStatus.BAD_REQUEST:
                raise ValueError(error_msg)
            raise RuntimeError(error_msg)
        return self.encoder.req_states[part_id].embedding_data

    def _item_embeddings(self, data):
        grids = data.get_grid()
        counts = (
            self.encoder.preprocessor.get_num_tokens(grid, data.modality)
            for grid in grids
        )
        return zip(grids, self.encoder.slice_embedding(data.embedding, counts))

    def _image_layouts(self, data):
        native = bool(self._info.native_model_type)
        original_sizes = getattr(data, "original_image_sizes", None)
        if self.encoder.model_type in {"kimi_k3", "kimi_k31"} and (
            original_sizes is None or len(original_sizes) != len(data.get_grid())
        ):
            raise RuntimeError("Missing Kimi-K3 original image sizes for staged images")
        layouts = []
        for index, (grid, tensor) in enumerate(self._item_embeddings(data)):
            if native and self.encoder.model_type == "kimi_vl" and len(grid) == 2:
                t, h, w = 1, *map(int, grid)
            else:
                t, h, w = map(int, grid)
            layout = encoder_pb2.ImageLayout(
                t=t, h=h, w=w, identity=self._item_identity(Modality.IMAGE, tensor)
            )
            if native:
                layout.embedding_rows = tensor.shape[0]
                if original_sizes is not None:
                    size = tuple(map(int, original_sizes[index]))
                    if len(size) != 2 or min(size) <= 0:
                        raise RuntimeError("Invalid staged original image size")
                    layout.original_size.extend(size)
            layouts.append(layout)
        return layouts

    @staticmethod
    def _item_identity(modality, tensor):
        return MultimodalDataItem(
            modality=modality, precomputed_embeddings=tensor
        ).cache_key.hex()

    def _video_layouts(self, data):
        timestamps = getattr(data, "video_timestamps", None)
        seconds = getattr(data, "second_per_grid_ts", None)
        timed_grid = self.encoder.model_type in TIMED_VIDEO_MODEL_TYPES
        frame_items = self.encoder.model_type in FRAME_VIDEO_MODEL_TYPES
        compression = 1
        if self.encoder.model_type == "mimo_v2":
            if (
                getattr(data, "video_audio_embedding", None) is not None
                or getattr(data, "video_audio_feature_lens", None) is not None
            ):
                raise ValueError(
                    "MiMo staged video requires a clip without an audio track"
                )
            # MiMo's native EPD metadata concatenates the timestamps of all clips.
            compression = (
                self.encoder.model._encoder_processor.temporal_compression_ratio
            )
            clips, offset = [], 0
            if timestamps is None or compression <= 0:
                raise RuntimeError("Missing MiMo staged video metadata")
            for grid in data.get_grid():
                frame_count, remainder = divmod(int(grid[0]), compression)
                if remainder or not frame_count:
                    raise RuntimeError("Invalid MiMo staged video temporal grid")
                clips.append(timestamps[offset : offset + frame_count])
                offset += frame_count
            if len(timestamps) != offset:
                raise RuntimeError("MiMo timestamps differ from its staged video grid")
            timestamps = clips
            frame_items = True
        requires_timestamps = (
            frame_items
            or self.encoder.model_type in GLM_VIDEO_MODEL_TYPES
            or self.encoder.model_type == "muse_spark"
        )
        optional_timestamps = self.encoder.model_type == "minimax_m3_vl"
        count = len(data.get_grid())
        if (
            requires_timestamps or (optional_timestamps and timestamps is not None)
        ) and (timestamps is None or len(timestamps) != count):
            raise RuntimeError("Missing native timestamps for staged video")
        if timed_grid and (seconds is None or len(seconds) != count):
            raise RuntimeError("Missing native grid duration for staged video")
        layouts = []
        for i, (grid, tensor) in enumerate(self._item_embeddings(data)):
            t, h, w = map(int, grid)
            frames = t // compression
            if min(t, h, w) <= 0 or tensor.shape[0] % frames:
                raise RuntimeError("Invalid staged video grid or embedding length")
            layout = encoder_pb2.VideoLayout(t=t, h=h, w=w)
            if self._info.native_model_type:
                layout.embedding_rows = tensor.shape[0]
            if requires_timestamps or (
                optional_timestamps
                and timestamps is not None
                and timestamps[i] is not None
            ):
                times = [float(x) for x in timestamps[i]]
                if (
                    len(times) != frames
                    or any(not math.isfinite(x) or x < 0 for x in times)
                    or any(a > b for a, b in pairwise(times))
                ):
                    raise RuntimeError("Invalid staged video frame timestamps")
                layout.timestamps.extend(times)
            chunks = tensor.tensor_split(frames) if frame_items else (tensor,)
            if timed_grid:
                duration = float(seconds[i])
                if not math.isfinite(duration) or duration <= 0:
                    raise RuntimeError("Invalid staged video grid duration")
                layout.second_per_grid_ts = duration
            layout.identities.extend(
                self._item_identity(Modality.VIDEO, chunk) for chunk in chunks
            )
            layouts.append(layout)
        return layouts

    def _describe_part(self, part_id):
        tensor = self.encoder.req_states[part_id].embedding_data.embedding
        return encoder_pb2.DescribeResponse(
            shape=tensor.shape, dtype=str(tensor.dtype).removeprefix("torch.")
        )

    async def _write_part(self, part_id, target):
        await self.encoder.send_to_destination(
            self.encoder.req_states[part_id], SendDestination(nixl_target=target)
        )

    async def _deliver_part(self, part_id, receive_url):
        await self.encoder.send_to_destination(
            self.encoder.req_states[part_id],
            SendDestination(endpoint=receive_url, ephemeral=True),
        )

    async def _release_staged_part(self, part_id):
        await self.encoder.release_request(part_id)
        if self.encoder.mm_global_cache_enabled:
            self.encoder.mm_global_cache.discard_prefetch_waiter(part_id)

    async def _broadcast_to_workers(self, request):
        if self._dispatch_failed:
            raise RuntimeError("Encoder TP dispatch is unavailable")
        message = wrap_as_pickle(request)
        try:
            for socket in self.send_sockets:
                await async_sock_send(socket, message)
        except Exception:
            # Some followers may already be inside this collective. The group
            # cannot accept another dispatch until the replica restarts.
            self._fail_worker_group()
            raise

    def _fail_worker_group(self):
        self._dispatch_failed = True
        if self._on_dispatch_failure is not None:
            self._on_dispatch_failure()

    async def _dispatch_staged(self, request, modality):
        await self._broadcast_to_workers({"type": "staged_encode", "request": request})
        try:
            return await self.encoder.encode_staged_on_tp_ranks(request, modality)
        except Exception:
            # batch_encode returns request errors. An exception here means the
            # ranks could not agree on the result or the collective itself failed.
            self._fail_worker_group()
            raise

    async def _dispatch_encode(self, request_dict: dict):
        await self._broadcast_to_workers(request_dict)
        return await self.encoder.encode_request(request_dict, Modality.IMAGE)

    async def Encode(
        self, request: encoder_pb2.EncodeRequest, context
    ) -> encoder_pb2.EncodeResponse:
        if self.encoder.transfer_backend == "nixl":
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "NIXL encoders require Stage/Deliver",
            )
        try:
            request_dict = {
                "mm_items": list(request.mm_items),
                "modality": Modality.IMAGE.name,
                "req_id": request.req_id,
                "num_parts": request.num_parts,
                "part_idx": request.part_idx,
            }
            if err := validate_encode_request(request_dict):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(err)
                return encoder_pb2.EncodeResponse()

            # gRPC encode is image-only; the request follows the configured
            # cache and transfer backend. Keep TP dispatch and rank-0 collective
            # launch order identical when gRPC handlers run concurrently.
            async with self.encoder.encode_dispatch_lock:
                encode_task = asyncio.create_task(self._dispatch_encode(request_dict))
                result = await await_task_completion_on_cancel(
                    encode_task, f"Encoder request {request.req_id}"
                )
                (
                    nbytes,
                    embedding_len,
                    embedding_dim,
                    error_msg,
                    error_code,
                ) = result
            if error_msg is not None:
                await self.encoder.release_request(request.req_id)
                context.set_code(
                    grpc.StatusCode.INVALID_ARGUMENT
                    if error_code == HTTPStatus.BAD_REQUEST
                    else grpc.StatusCode.INTERNAL
                )
                context.set_details(error_msg)
                return encoder_pb2.EncodeResponse()

            if get_disagg().encoder_transfer_backend == "mooncake":
                return encoder_pb2.EncodeResponse(
                    embedding_size=nbytes,
                    embedding_len=embedding_len,
                    embedding_dim=embedding_dim,
                )
            elif get_disagg().encoder_transfer_backend == "zmq_to_scheduler":
                embedding_ports = list(request.embedding_port)
                logger.info(f"embedding_port = {embedding_ports}")
                if not embedding_ports:
                    await self.encoder.send_with_url(req_id=request.req_id)
                else:
                    tasks = []
                    for embedding_port in embedding_ports:
                        tasks.append(
                            self.encoder.send(
                                req_id=request.req_id,
                                prefill_host=request.prefill_host,
                                embedding_port=embedding_port,
                            )
                        )
                    await asyncio.gather(*tasks)
                    await self.encoder.release_request(request.req_id)
                return encoder_pb2.EncodeResponse()
            elif get_disagg().encoder_transfer_backend == "zmq_to_tokenizer":
                embedding_port = (
                    request.embedding_port[0] if request.embedding_port else 0
                )
                await self.encoder.send(
                    req_id=request.req_id,
                    prefill_host=request.prefill_host,
                    embedding_port=embedding_port,
                )
                await self.encoder.release_request(request.req_id)
                return encoder_pb2.EncodeResponse()

            return encoder_pb2.EncodeResponse()

        except asyncio.CancelledError:
            try:
                await asyncio.shield(self.encoder.release_request(request.req_id))
            except Exception:
                logger.exception(
                    "Failed to release cancelled encoder request %s", request.req_id
                )
            raise
        except Exception as e:
            logger.error(f"Encode error: {e}")
            traceback.print_exc()
            await self.encoder.release_request(request.req_id)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return encoder_pb2.EncodeResponse()

    async def Send(
        self, request: encoder_pb2.SendRequest, context
    ) -> encoder_pb2.SendResponse:
        if self.encoder.transfer_backend == "nixl":
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "NIXL encoders require Stage/Deliver",
            )
        try:
            await self.encoder.send(
                req_id=request.req_id,
                prefill_host=request.prefill_host,
                embedding_port=request.embedding_port,
                session_id=request.session_id if request.session_id else None,
                buffer_address=(
                    request.buffer_address if request.buffer_address else None
                ),
            )
            await self.encoder.release_request(request.req_id)
            return encoder_pb2.SendResponse()

        except asyncio.CancelledError:
            try:
                await asyncio.shield(self.encoder.release_request(request.req_id))
            except Exception:
                logger.exception(
                    "Failed to release cancelled encoder request %s", request.req_id
                )
            raise
        except Exception as e:
            logger.error(f"Send error: {e}")
            traceback.print_exc()
            await self.encoder.release_request(request.req_id)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return encoder_pb2.SendResponse()

    async def SchedulerReceiveUrl(
        self, request: encoder_pb2.SchedulerReceiveUrlRequest, context
    ) -> encoder_pb2.SchedulerReceiveUrlResponse:
        if self.encoder.transfer_backend == "nixl":
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "NIXL encoders require Stage/Deliver",
            )
        try:
            await self.encoder.register_embedding_destinations(
                request.req_id,
                request.receive_count,
                [request.receive_url],
            )
            return encoder_pb2.SchedulerReceiveUrlResponse()

        except Exception as e:
            logger.error(f"SchedulerReceiveUrl error: {e}")
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return encoder_pb2.SchedulerReceiveUrlResponse()


async def serve_grpc_encoder(server_args: ServerArgs):
    publish(server_args, role="encoder")
    ctx = mp.get_context("spawn")
    zmq_ctx = zmq.asyncio.Context(10)
    ipc_path_prefix = random_uuid()
    port_args = PortArgs.init_new(server_args)

    if get_parallel().dist_init_addr:
        na = NetworkAddress.parse(get_parallel().dist_init_addr)
        dist_init_method = na.to_tcp()
    else:
        dist_init_method = NetworkAddress(
            get_serving().host or "127.0.0.1", port_args.nccl_port
        ).to_tcp()

    send_sockets: List[zmq.Socket] = []
    for rank in range(1, get_parallel().tp_size):
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"
        send_sockets.append(
            get_zmq_socket(zmq_ctx, zmq.PUSH, schedule_path, bind=False)
        )
        ctx.Process(
            target=launch_encoder,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        ).start()

    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
        ],
    )

    health_servicer = EncoderHealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    def stop_failed_group():
        logger.error("Stopping gRPC encoder after a TP dispatch failure")
        health_servicer.set_not_serving()
        asyncio.create_task(server.stop(grace=1))

    encoder_servicer = SGLangEncoderServer(
        on_dispatch_failure=stop_failed_group,
        encoder=encoder,
        send_sockets=send_sockets,
        server_args=server_args,
    )
    add_SGLangEncoderServicer_to_server(encoder_servicer, server)

    SERVICE_NAMES = (
        encoder_pb2.DESCRIPTOR.services_by_name["SglangEncoder"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    listen_addr = NetworkAddress(
        get_serving().host, get_serving().port
    ).to_host_port_str()
    server.add_insecure_port(listen_addr)

    await server.start()
    logger.info(f"gRPC encoder server listening on {listen_addr}")

    health_servicer.set_serving()

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC encoder server...")
        health_servicer.set_not_serving()
        await server.stop(grace=5)
