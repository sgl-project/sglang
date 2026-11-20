"""
gRPC Encoder Server for SGLang EPD (Encode-Prefill-Decode) mode.

This server provides gRPC-based encoding for multimodal inputs.

Usage:
    python -m sglang.launch_server --model-path <model> --encoder-only --grpc-mode
"""

import asyncio
import logging
import multiprocessing as mp
import traceback
from concurrent import futures
from http import HTTPStatus
from typing import List

import grpc
import zmq
import zmq.asyncio
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from sglang.encoder.v1 import encoder_pb2, encoder_pb2_grpc
from sglang.srt.disaggregation.encoder.receiver import create_part_req_id
from sglang.srt.disaggregation.encoder.runtime import validate_encode_request
from sglang.srt.disaggregation.encoder.server import (
    MMEncoder,
    SendDestination,
    await_task_completion_on_cancel,
    launch_encoder,
)
from sglang.srt.disaggregation.encoder.staged import StagedImageService
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import async_sock_send, wrap_as_pickle
from sglang.srt.managers.req_types import MultimodalDataItem
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
    ):
        self.encoder = encoder
        self.send_sockets = send_sockets
        self.server_args = server_args
        info = None
        if (
            encoder.transfer_backend in ("zmq_to_scheduler", "nixl")
            and get_parallel().tp_size == get_parallel().num_dp_ranks == 1
            and encoder.model_type
            in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}
        ):
            model = get_model().model_path
            info = encoder_pb2.GetEncoderInfoResponse(
                model_path=model,
                tokenizer_path=get_serving().tokenizer_path or model,
                served_model_name=get_serving().served_model_name or model,
            )
        nixl_enabled = encoder.transfer_backend == "nixl"
        super().__init__(
            encode=self._stage_ordered_images,
            send=None if nixl_enabled else self._deliver_part,
            describe=self._describe_part if nixl_enabled else None,
            write=self._write_part if nixl_enabled else None,
            release=self._release_staged_part,
            make_part_id=create_part_req_id,
            info=info,
            stage_timeout=envs.SGLANG_ENCODER_REQ_TIMEOUT.get(),
            ttl=encoder.send_timeout,
            capacity=envs.SGLANG_ENCODER_MAX_STAGED_REQUESTS.get(),
        )

    async def _stage_ordered_images(self, part_id, images):
        request = {
            "modality": Modality.IMAGE.name,
            "req_id": part_id,
            "num_parts": 1,
            "part_idx": 0,
            "mm_items": images,
        }
        # Stage is single-rank. A timeout can cancel queueing and preprocessing
        # directly. Once the global cache owns native copy handles it must drain
        # those copies before releasing its buffers.
        async with self.encoder.encode_dispatch_lock:
            (result,) = await self.encoder.batch_encode(
                [request], Modality.IMAGE, drain_global_cache_on_cancel=True
            )
        _, _, _, error_msg, error_code = result
        if error_msg is not None:
            if error_code == HTTPStatus.BAD_REQUEST:
                raise ValueError(error_msg)
            raise RuntimeError(error_msg)
        data = self.encoder.req_states[part_id].embedding_data
        grids = data.get_grid()
        counts = [
            self.encoder.preprocessor.get_num_tokens(g, Modality.IMAGE) for g in grids
        ]
        slices = self.encoder.slice_embedding(data.embedding, counts)
        # Match QwenVL's get_mm_data: identity is over each image's native
        # precomputed tensor (dtype, shape and content). Router labels never
        # substitute for content identity. This lets D reproduce P's pads.
        layouts = []
        for grid, tensor in zip(grids, slices):
            t, h, w = map(int, grid)
            item = MultimodalDataItem(
                modality=Modality.IMAGE, precomputed_embeddings=tensor
            )
            layouts.append(
                encoder_pb2.ImageLayout(t=t, h=h, w=w, identity=item.cache_key.hex())
            )
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

    async def _dispatch_encode(self, request_dict: dict):
        for socket in self.send_sockets:
            await async_sock_send(socket, wrap_as_pickle(request_dict))
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

    encoder_servicer = SGLangEncoderServer(
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
