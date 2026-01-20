"""
gRPC Encoder Server for SGLang EPD (Encode-Prefill-Decode) mode.

This server provides gRPC-based encoding for multimodal inputs (images, videos, audio).
It implements the SglangEncoder service defined in sglang_encoder.proto.

Usage:
    python -m sglang.launch_server --model-path <model> --encoder-only --grpc-mode
"""

import asyncio
import logging
import multiprocessing as mp
import traceback
from concurrent import futures
from typing import List

import grpc
import zmq
import zmq.asyncio
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from sglang.srt.disaggregation.encode_server import (
    MMEncoder,
    rid_lock,
    rid_to_receive_count,
    rid_to_receive_endpoint,
)
from sglang.srt.grpc import sglang_encoder_pb2, sglang_encoder_pb2_grpc
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_zmq_socket, random_uuid

logger = logging.getLogger(__name__)

# Shared scheduler receive URL state (use encode_server globals).


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
        """Standard health check for Kubernetes probes."""
        if self._serving:
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.SERVING
            )
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.NOT_SERVING
        )

    async def Watch(self, request, context):
        """Streaming health check."""
        yield await self.Check(request, context)


class SglangEncoderServicer(sglang_encoder_pb2_grpc.SglangEncoderServicer):
    """
    gRPC service implementation for SGLang encoder.
    Mirrors the HTTP endpoints in encode_server.py.
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

    async def Encode(
        self, request: sglang_encoder_pb2.EncodeRequest, context
    ) -> sglang_encoder_pb2.EncodeResponse:
        """
        Encode multimodal items (images/videos/audio).
        Mirrors handle_encode_request() from encode_server.py.
        """
        try:
            # Broadcast request to worker processes
            request_dict = {
                "mm_items": list(request.mm_items),
                "req_id": request.req_id,
                "num_parts": request.num_parts,
                "part_idx": request.part_idx,
            }
            for socket in self.send_sockets:
                socket.send_pyobj(request_dict)

            # Perform encoding
            nbytes, embedding_len, embedding_dim = await self.encoder.encode(
                mm_items=list(request.mm_items),
                req_id=request.req_id,
                num_parts=request.num_parts,
                part_idx=request.part_idx,
            )

            # Handle different transfer backends
            if self.server_args.encoder_transfer_backend == "mooncake":
                # Return embedding metadata for mooncake backend
                return sglang_encoder_pb2.EncodeResponse(
                    embedding_size=nbytes,
                    embedding_len=embedding_len,
                    embedding_dim=embedding_dim,
                )
            elif self.server_args.encoder_transfer_backend == "zmq_to_scheduler":
                embedding_ports = list(request.embedding_port)
                logger.info(f"embedding_port = {embedding_ports}")
                if not embedding_ports:
                    # Dynamic endpoint discovery
                    await self.encoder.send_with_url(req_id=request.req_id)
                else:
                    # Send to specified ports
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
                    self.encoder.embedding_to_send.pop(request.req_id, None)
                return sglang_encoder_pb2.EncodeResponse()
            elif self.server_args.encoder_transfer_backend == "zmq_to_tokenizer":
                # Send directly to tokenizer
                embedding_port = request.embedding_port[0] if request.embedding_port else 0
                await self.encoder.send(
                    req_id=request.req_id,
                    prefill_host=request.prefill_host,
                    embedding_port=embedding_port,
                )
                self.encoder.embedding_to_send.pop(request.req_id, None)
                return sglang_encoder_pb2.EncodeResponse()

            return sglang_encoder_pb2.EncodeResponse()

        except Exception as e:
            logger.error(f"Encode error: {e}")
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return sglang_encoder_pb2.EncodeResponse()

    async def Send(
        self, request: sglang_encoder_pb2.SendRequest, context
    ) -> sglang_encoder_pb2.SendResponse:
        """
        Send encoded embeddings to prefill server (mooncake backend).
        Mirrors handle_send_request() from encode_server.py.
        """
        try:
            await self.encoder.send(
                req_id=request.req_id,
                prefill_host=request.prefill_host,
                embedding_port=request.embedding_port,
                session_id=request.session_id if request.session_id else None,
                buffer_address=request.buffer_address if request.buffer_address else None,
            )
            self.encoder.embedding_to_send.pop(request.req_id, None)
            return sglang_encoder_pb2.SendResponse()

        except Exception as e:
            logger.error(f"Send error: {e}")
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return sglang_encoder_pb2.SendResponse()

    async def SchedulerReceiveUrl(
        self, request: sglang_encoder_pb2.SchedulerReceiveUrlRequest, context
    ) -> sglang_encoder_pb2.SchedulerReceiveUrlResponse:
        """
        Register scheduler receive URL (zmq_to_scheduler backend).
        Mirrors handle_scheduler_receive_url_request() from encode_server.py.
        """
        try:
            rid = request.req_id
            async with rid_lock:
                global rid_to_receive_endpoint, rid_to_receive_count
                if rid not in rid_to_receive_endpoint:
                    rid_to_receive_endpoint[rid] = set()
                    rid_to_receive_count[rid] = request.receive_count
                assert rid_to_receive_count[rid] == request.receive_count
                rid_to_receive_endpoint[rid].add(request.receive_url)
            return sglang_encoder_pb2.SchedulerReceiveUrlResponse()

        except Exception as e:
            logger.error(f"SchedulerReceiveUrl error: {e}")
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return sglang_encoder_pb2.SchedulerReceiveUrlResponse()


async def run_encoder_worker(
    server_args: ServerArgs, schedule_path: str, dist_init_method: str, rank: int
):
    """Run encoder worker process (same as HTTP version)."""
    encoder = MMEncoder(server_args, schedule_path, dist_init_method, rank)
    while True:
        request = await encoder.schedule_socket.recv_pyobj()
        await encoder.encode(
            mm_items=request["mm_items"],
            req_id=request["req_id"],
            num_parts=request["num_parts"],
            part_idx=request["part_idx"],
        )


def launch_encoder_worker(
    server_args: ServerArgs, schedule_path: str, dist_init_method: str, rank: int
):
    """Launch encoder worker process (same as HTTP version)."""
    try:
        asyncio.run(run_encoder_worker(server_args, schedule_path, dist_init_method, rank))
    except KeyboardInterrupt:
        logger.info(f"Exit rank {rank}")
    except Exception:
        traceback.print_exc()


async def serve_grpc_encoder(server_args: ServerArgs):
    """Start the gRPC encoder server."""

    # Initialize multiprocessing context and ZMQ
    ctx = mp.get_context("spawn")
    zmq_ctx = zmq.Context(10)
    ipc_path_prefix = random_uuid()
    port_args = PortArgs.init_new(server_args)

    # Determine distributed init method
    if server_args.dist_init_addr:
        dist_init_method = f"tcp://{server_args.dist_init_addr}"
    else:
        dist_init_method = f"tcp://127.0.0.1:{port_args.nccl_port}"

    # Launch worker processes for tensor parallelism
    send_sockets: List[zmq.Socket] = []
    for rank in range(1, server_args.tp_size):
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"
        send_sockets.append(
            get_zmq_socket(zmq_ctx, zmq.PUSH, schedule_path, bind=False)
        )
        ctx.Process(
            target=launch_encoder_worker,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        ).start()

    # Create main encoder (rank 0)
    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)

    # Create gRPC server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 256),
            ("grpc.max_receive_message_length", 1024 * 1024 * 256),
        ],
    )

    # Create and register health service
    health_servicer = EncoderHealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Create and register encoder service
    encoder_servicer = SglangEncoderServicer(
        encoder=encoder,
        send_sockets=send_sockets,
        server_args=server_args,
    )
    sglang_encoder_pb2_grpc.add_SglangEncoderServicer_to_server(encoder_servicer, server)

    # Enable reflection for debugging
    SERVICE_NAMES = (
        sglang_encoder_pb2.DESCRIPTOR.services_by_name["SglangEncoder"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    # Start server
    listen_addr = f"{server_args.host}:{server_args.port}"
    server.add_insecure_port(listen_addr)

    await server.start()
    logger.info(f"gRPC encoder server listening on {listen_addr}")

    # Mark as serving
    health_servicer.set_serving()

    # Wait for termination
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC encoder server...")
        health_servicer.set_not_serving()
        await server.stop(grace=5)
