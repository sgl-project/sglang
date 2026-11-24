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
from typing import List
from uuid import uuid4

import grpc
import torch
import zmq
import zmq.asyncio
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from sglang.encoder.v1 import encoder_pb2, encoder_pb2_grpc
from sglang.srt.disaggregation.encoder.handoff import staged_embedding_digest
from sglang.srt.disaggregation.encoder.handoff_auth import EncoderHandoffSigner
from sglang.srt.disaggregation.encoder.receiver import create_part_req_id
from sglang.srt.disaggregation.encoder.runtime import validate_encode_request
from sglang.srt.disaggregation.encoder.server import (
    MMEncoder,
    ReqState,
    SendDestination,
    await_task_completion_on_cancel,
    launch_encoder,
    meta_registry,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import async_sock_send, wrap_as_pickle
from sglang.srt.multimodal.media_io import media_url_security_config, media_url_timeout
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.staged_layout import staged_layout_for
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


class SGLangEncoderServer(SGLangEncoderServicer):
    """
    gRPC service implementation for SGLang encoder.
    """

    def __init__(
        self,
        encoder: MMEncoder,
        send_sockets: List[zmq.Socket],
        *,
        on_dispatch_failure=None,
    ):
        self.encoder = encoder
        self._on_dispatch_failure = on_dispatch_failure
        self._dispatch_failed = False
        self.send_sockets = send_sockets
        info = None
        parallel = get_parallel()
        staging_enabled = (
            encoder.transfer_backend in ("zmq_to_scheduler", "nixl")
            and parallel.num_dp_ranks == parallel.pp_size == parallel.nnodes == 1
            and len(send_sockets) == parallel.tp_size - 1
        )
        self._staged_layout = layout = (
            staged_layout_for(encoder.model_type) if staging_enabled else None
        )
        vision_enabled = layout is not None and layout.encoder_vision_enabled(encoder)
        images_enabled = bool(
            vision_enabled
            and layout.images
            and encoder.preprocessor.supports_modality(Modality.IMAGE)
        )
        videos_enabled = bool(
            vision_enabled
            and layout.video is not None
            and encoder.preprocessor.supports_modality(Modality.VIDEO)
            and video_decoder_available()
        )
        feature_model = getattr(encoder.model, "thinker", encoder.model)
        audio_enabled = bool(
            layout
            and layout.audio
            and encoder.preprocessor.supports_modality(Modality.AUDIO)
            and callable(getattr(feature_model, "get_audio_feature", None))
            and layout.encoder_audio_enabled(encoder)
        )
        if images_enabled or videos_enabled or audio_enabled:
            model = get_model().model_path
            domains, max_mb, _, _ = media_url_security_config()
            # With no configured timeout, media I/O uses per-modality defaults.
            remote_timeout = media_url_timeout(0) or None
            info = encoder_pb2.GetEncoderInfoResponse(
                model_path=model,
                tokenizer_path=get_serving().tokenizer_path or model,
                served_model_name=get_serving().served_model_name or model,
                native_model_type=encoder.model_type
                if (images_enabled or videos_enabled) and not layout.legacy_vision
                else "",
                staged_transfer_backend=encoder.transfer_backend,
                staged_layout_id=layout.wire_id(encoder.model_type),
                supports_staged_images=images_enabled,
                staged_audio_model_type=encoder.model_type if audio_enabled else "",
                supports_staged_video=videos_enabled,
                staged_media_policy=encoder_pb2.StagedMediaPolicy(
                    allowed_domains=domains,
                    max_bytes=max_mb * 1024 * 1024,
                    timeout_secs=remote_timeout,
                ),
            )
        self._configure_staging(
            info,
            stage_timeout=envs.SGLANG_ENCODER_REQ_TIMEOUT.get(),
            ttl=encoder.send_timeout,
            capacity=envs.SGLANG_ENCODER_MAX_STAGED_REQUESTS.get(),
            signer=EncoderHandoffSigner.from_config() if info is not None else None,
        )

    def _configure_staging(self, info, *, stage_timeout, ttl, capacity, signer=None):
        self._info = info
        self._stage_timeout = stage_timeout
        self._ttl = ttl
        self._capacity = capacity
        self._handoff_signer = signer
        if info is not None:
            info.instance_id = uuid4().bytes
            if signer is not None:
                info.handoff_key_id = signer.key_id
                info.handoff_peer = signer.peer

    async def _stage_media(self, part_id, images, videos, audios):
        # The native encoder batches one modality at a time. Publish one lease
        # for the entire request in image/video/audio order.
        groups = [
            (modality, sources)
            for modality, sources in (
                (Modality.IMAGE, images),
                (Modality.VIDEO, videos),
                (Modality.AUDIO, audios),
            )
            if sources
        ]
        layouts = {
            modality: []
            for modality in (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO)
        }
        tensors = []
        root = None
        async with self.encoder.encode_dispatch_lock:
            for index, (modality, sources) in enumerate(groups):
                name = part_id if index == 0 else f"{part_id}_{modality.name.lower()}"
                try:
                    data = await self._encode_modality(name, sources, modality)
                    if root is None:
                        root = data
                    if modality == Modality.IMAGE:
                        layouts[modality] = self._image_layouts(
                            data, explicit_rows=bool(audios)
                        )
                    elif modality == Modality.VIDEO:
                        layouts[modality] = self._video_layouts(
                            data, explicit_rows=bool(audios)
                        )
                    else:
                        layouts[modality] = self._audio_layouts(data)
                    tensor = data.embedding
                    if tensor.dtype != root.dtype or tensor.ndim != 2:
                        raise RuntimeError("Incompatible staged embedding formats")
                    if tensor.shape[1] != root.shape[1]:
                        if (
                            modality != Modality.AUDIO
                            or tensor.shape[1] > root.shape[1]
                        ):
                            raise RuntimeError("Incompatible staged embedding widths")
                        # DeepStack vision includes auxiliary channels. Restore
                        # the original audio width from AudioLayout on P/PD.
                        tensor = torch.nn.functional.pad(
                            tensor, (0, root.shape[1] - tensor.shape[1])
                        )
                    tensors.append(tensor)
                finally:
                    if index:
                        await await_task_completion_on_cancel(
                            asyncio.create_task(self._release_staged_modality(name)),
                            f"Retire staged {modality.name.lower()} {name}",
                        )
            if len(tensors) > 1:
                packed = torch.cat(tensors)
                root.embedding = packed
                root.shape = list(packed.shape)
                await meta_registry.publish(part_id, packed.nbytes, *packed.shape)
        return layouts[Modality.IMAGE], layouts[Modality.VIDEO], layouts[Modality.AUDIO]

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

    def _image_layouts(self, data, *, explicit_rows=False):
        native = explicit_rows or bool(self._info.native_model_type)
        original_sizes = getattr(data, "original_image_sizes", None)
        if self._staged_layout.original_image_sizes and (
            original_sizes is None or len(original_sizes) != len(data.get_grid())
        ):
            raise RuntimeError("Missing original image sizes for staged images")
        layouts = []
        for index, (grid, tensor) in enumerate(self._item_embeddings(data)):
            if native and len(grid) == 2:
                t, h, w = 1, *map(int, grid)
            else:
                t, h, w = map(int, grid)
            layout = encoder_pb2.ImageLayout(
                t=t,
                h=h,
                w=w,
                identity=staged_embedding_digest(Modality.IMAGE, tensor).hex(),
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

    def _audio_layouts(self, data):
        layouts = []
        for feature_length, tensor in self._item_embeddings(data):
            length = int(feature_length)
            if length <= 0 or tensor.ndim != 2 or min(tensor.shape) <= 0:
                raise ValueError("Empty audio or invalid audio embedding")
            layouts.append(
                encoder_pb2.AudioLayout(
                    identity=staged_embedding_digest(Modality.AUDIO, tensor).hex(),
                    feature_length=length,
                    embedding_rows=tensor.shape[0],
                    embedding_width=tensor.shape[1],
                )
            )
        return layouts

    def _video_layouts(self, data, *, explicit_rows=False):
        return self._staged_layout.encode_video_layouts(
            self.encoder,
            data,
            self._item_embeddings,
            lambda tensor: staged_embedding_digest(Modality.VIDEO, tensor).hex(),
            explicit_rows or bool(self._info.native_model_type),
        )

    def _describe_part(self, part_id):
        tensor = self.encoder.req_states[part_id].embedding_data.embedding
        return encoder_pb2.DescribeResponse(
            shape=tensor.shape, dtype=str(tensor.dtype).removeprefix("torch.")
        )

    async def _release_staged_modality(self, part_id):
        await self.encoder.release_request(part_id)
        if self.encoder.mm_global_cache_enabled:
            self.encoder.mm_global_cache.discard_prefetch_waiter(part_id)

    async def _check_staging_config(self, context):
        if (
            self._info is None
            or not 0 < self._stage_timeout < math.inf
            or not 0 < self._ttl < math.inf
            or self._capacity <= 0
        ):
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Staged media require a compatible processor, zmq_to_scheduler or nixl, "
                "a single-node TP group with PP=DP=1 and positive staging limits",
            )

    async def GetEncoderInfo(self, request, context):
        await self._check_staging_config(context)
        return self._info

    async def Stage(self, request, context):
        await self._check_staging_config(context)
        images, videos = list(request.mm_items), list(request.video_urls)
        audios = list(request.audio_urls)
        if (
            not request.request_id
            or not (images or videos or audios)
            or not all(images)
            or not all(videos)
            or not all(audios)
        ):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "request_id and nonempty media are required",
            )
        if (
            images
            and self._info.HasField("supports_staged_images")
            and not self._info.supports_staged_images
        ):
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "staged images are unavailable"
            )
        if audios and not self._info.staged_audio_model_type:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "staged audio is unavailable"
            )
        if videos and not self._info.supports_staged_video:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "staged video is unavailable"
            )
        states = self.encoder.req_states
        if sum(state.externally_staged for state in states.values()) >= self._capacity:
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED, "encoder staging full"
            )

        transfer_id = uuid4().hex
        part_id = create_part_req_id(transfer_id, 0)
        # Reserve before the dispatch lock can suspend. batch_encode acquires
        # its normal native encode reference on this same state.
        state = states[part_id] = ReqState(part_id, externally_staged=True)
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._stage_timeout
            image_layouts, video_layouts, audio_layouts = await asyncio.wait_for(
                self._stage_media(part_id, images, videos, audios), self._stage_timeout
            )
            if loop.time() >= deadline:
                raise asyncio.TimeoutError
            if (len(image_layouts), len(video_layouts), len(audio_layouts)) != (
                len(images),
                len(videos),
                len(audios),
            ):
                raise ValueError("staged layout count differs from media count")
            transfer = encoder_pb2.StagedImageTransfer(
                version=2 if videos else 1,
                request_id=request.request_id,
                transfer_id=transfer_id,
                part_id=part_id,
                part_idx=0,
                num_parts=1,
                image_count=len(images),
                images=image_layouts,
                videos=video_layouts,
                audios=audio_layouts,
            )
            native_type = (
                self._info.staged_audio_model_type
                if audios
                else self._info.native_model_type
            )
            if native_type:
                shape = self._describe_part(part_id).shape
                rows = sum(
                    item.embedding_rows
                    for item in (*image_layouts, *video_layouts, *audio_layouts)
                )
                if len(shape) != 2 or shape[0] != rows or shape[1] <= 0:
                    raise RuntimeError(
                        "Native staged layout does not match its embedding"
                    )
                transfer.version = 4 if audios else 3
                transfer.native_model_type = native_type
                transfer.embedding_width = shape[1]
            response = encoder_pb2.StageResponse(
                transfer=transfer, instance_id=self._info.instance_id
            )
            if self._handoff_signer is not None:
                response.handoff_mac = self._handoff_signer.sign(
                    transfer, self._info.instance_id
                )
            if states.get(part_id) is not state or state.release_requested:
                raise RuntimeError("Encoder request retired while staging")
            state.staged_transfer = transfer
            meta_registry.arm(part_id, self._ttl, self.encoder.release_request)
            logger.debug(
                "Stage complete request_id=%s part_id=%s images=%d videos=%d audios=%d",
                transfer.request_id,
                part_id,
                transfer.image_count,
                len(transfer.videos),
                len(transfer.audios),
            )
            return response
        except asyncio.CancelledError:
            await self._retire_staged(part_id)
            raise
        except asyncio.TimeoutError:
            await self._retire_staged(part_id)
            await context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED, "media staging timed out"
            )
        except Exception as exc:
            await self._retire_staged(part_id)
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT
                if isinstance(exc, ValueError)
                else grpc.StatusCode.INTERNAL,
                str(exc),
            )

    async def _retire_staged(self, part_id):
        await await_task_completion_on_cancel(
            asyncio.create_task(self.encoder.release_request(part_id)),
            f"Release staged encoder part {part_id}",
        )

    async def _lookup_transfer(self, transfer, context):
        part_id = create_part_req_id(transfer.transfer_id, 0)
        state = self.encoder.req_states.get(part_id)
        if state is not None and state.staged_transfer is not None:
            if meta_registry.expired(part_id):
                state.staged_transfer = None
                meta_registry.disarm(part_id)
                await self._retire_staged(part_id)
            elif state.staged_transfer == transfer:
                return state
            else:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "invalid transfer identity"
                )
        await context.abort(grpc.StatusCode.NOT_FOUND, "transfer expired or consumed")

    async def Describe(self, request, context):
        state = await self._lookup_transfer(request.transfer, context)
        if self.encoder.transfer_backend != "nixl":
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "NIXL is not enabled"
            )
        description = self._describe_part(state.req_id)
        description.instance_id = self._info.instance_id
        return description

    async def Deliver(self, request, context):
        state = await self._lookup_transfer(request.transfer, context)
        use_nixl = self.encoder.transfer_backend == "nixl"
        if use_nixl != bool(request.nixl_target) or use_nixl == bool(
            request.receive_url
        ):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "missing destination or incompatible encoder transfer backend",
            )
        # Claim before the native send acquires its reference. A second
        # Deliver cannot post another write even if the first RPC is cancelled.
        state.staged_transfer = None
        meta_registry.disarm(state.req_id)
        destination = (
            SendDestination(nixl_target=request.nixl_target)
            if use_nixl
            else SendDestination(endpoint=request.receive_url, ephemeral=True)
        )
        send_error = None
        try:
            async with asyncio.timeout(self._ttl):
                await self.encoder.send_to_destination(state, destination)
        except asyncio.TimeoutError:
            send_error = "staged media delivery timed out"
        except Exception as exc:
            send_error = str(exc)
        finally:
            await self._retire_staged(state.req_id)
        if send_error is not None:
            await context.abort(grpc.StatusCode.UNAVAILABLE, send_error)
        return encoder_pb2.DeliverResponse()

    async def Release(self, request, context):
        part_id = create_part_req_id(request.transfer_id, 0)
        state = self.encoder.req_states.get(part_id)
        if state is not None and state.staged_transfer is not None:
            state.staged_transfer = None
            meta_registry.disarm(part_id)
            await self._retire_staged(part_id)
        elif (
            state is not None
            and state.externally_staged
            and self.encoder.transfer_backend == "nixl"
        ):
            # A claimed Deliver owns retirement. Its native send reference
            # remains live through UCX completion, including on cancellation.
            async with state.lifecycle_condition:
                await state.lifecycle_condition.wait_for(
                    lambda: state.active_sends == 0
                )
        return encoder_pb2.ReleaseResponse(
            instance_id=self._info.instance_id if self._info is not None else b""
        )

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
