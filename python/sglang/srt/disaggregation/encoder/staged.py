"""One-shot gRPC handoff of an encoder-owned, already-produced multimodal part."""

import asyncio
import logging
import math
from dataclasses import dataclass
from uuid import uuid4

import grpc

from sglang.encoder.v1 import encoder_pb2 as pb

logger = logging.getLogger(__name__)


@dataclass
class _Lease:
    transfer: pb.StagedImageTransfer
    expiry: asyncio.TimerHandle


class StagedImageService:
    """The encoder owns the tensor. Only bounded control metadata lives here.

    All methods run on the gRPC event loop. Taking a lease is synchronous so
    delivery, expiration and explicit release have exactly one owner.
    """

    def __init__(
        self,
        *,
        encode,
        send,
        release,
        make_part_id,
        info,
        stage_timeout,
        ttl,
        capacity,
        describe=None,
        write=None,
    ):
        self._encode = encode
        self._describe = describe
        self._write = write
        self._send = send
        self._release = release
        self._make_part_id = make_part_id
        self._info = info
        self._stage_timeout = stage_timeout
        self._ttl = ttl
        self._capacity = capacity
        self._active = 0  # includes queued encodes, staged parts and active sends
        self._writing: dict[str, asyncio.Event] = {}
        self._leases: dict[str, _Lease] = {}
        self._cleanup_tasks: set[asyncio.Task] = set()

    async def _check_config(self, context):
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
        await self._check_config(context)
        return self._info

    async def Stage(self, request, context):
        await self._check_config(context)
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
        if self._active >= self._capacity:
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED, "encoder staging full"
            )

        transfer_id = uuid4().hex
        part_id = self._make_part_id(transfer_id, 0)
        self._active += 1
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._stage_timeout
            image_layouts, video_layouts, audio_layouts = await asyncio.wait_for(
                self._encode(part_id, images, videos, audios), self._stage_timeout
            )
            if loop.time() >= deadline:
                raise asyncio.TimeoutError
            if (len(image_layouts), len(video_layouts), len(audio_layouts)) != (
                len(images),
                len(videos),
                len(audios),
            ):
                raise ValueError("staged layout count differs from media count")
            transfer = pb.StagedImageTransfer(
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
                shape = self._describe(part_id).shape
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
            response = pb.StageResponse(transfer=transfer)
            expiry = loop.call_later(self._ttl, self._expire, transfer_id)
            self._leases[transfer_id] = _Lease(transfer, expiry)
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
            await asyncio.shield(self._start_retirement(part_id))
            raise
        except asyncio.TimeoutError:
            await asyncio.shield(self._start_retirement(part_id))
            await context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED, "media staging timed out"
            )
        except Exception as exc:
            await asyncio.shield(self._start_retirement(part_id))
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT
                if isinstance(exc, ValueError)
                else grpc.StatusCode.INTERNAL,
                str(exc),
            )

    async def _retire(self, part_id):
        try:
            await self._release(part_id)
        except Exception:
            logger.exception("Could not release staged encoder part %s", part_id)
            raise
        finally:
            self._active -= 1

    def _start_retirement(self, part_id):
        # A cancelled Release/Deliver RPC must not interrupt the native
        # cleanup while it is waiting on an active reference.
        task = asyncio.create_task(self._retire(part_id))
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        return task

    def _expire(self, transfer_id):
        lease = self._leases.pop(transfer_id, None)
        if lease is not None:
            lease.expiry.cancel()
            self._start_retirement(lease.transfer.part_id)

    async def _lookup(self, transfer, context):
        transfer_id = transfer.transfer_id
        lease = self._leases.get(transfer_id)
        # A long synchronous model forward can delay the timer callback. The
        # deadline itself governs a claim even if the callback is still queued.
        if (
            lease is not None
            and asyncio.get_running_loop().time() >= lease.expiry.when()
        ):
            self._expire(transfer_id)
            lease = None
        if lease is None:
            await context.abort(
                grpc.StatusCode.NOT_FOUND, "transfer expired or consumed"
            )
        if transfer != lease.transfer:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "invalid transfer identity"
            )
        return lease

    async def Describe(self, request, context):
        lease = await self._lookup(request.transfer, context)
        if self._write is None:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "NIXL is not enabled"
            )
        return self._describe(lease.transfer.part_id)

    async def Deliver(self, request, context):
        lease = await self._lookup(request.transfer, context)
        use_nixl = bool(request.nixl_target)
        if (use_nixl and (request.receive_url or self._write is None)) or (
            not use_nixl and (not request.receive_url or self._send is None)
        ):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "missing destination or incompatible encoder transfer backend",
            )
        # Claim before the first send. A retry cannot send the same tensor to
        # another scheduler while the first transfer is still in flight.
        self._leases.pop(lease.transfer.transfer_id)
        lease.expiry.cancel()
        logger.debug(
            "Deliver claimed request_id=%s part_id=%s",
            lease.transfer.request_id,
            lease.transfer.part_id,
        )
        send_error = None
        if use_nixl:
            # A cancelled Deliver may still be draining a posted WRITE. Release
            # is the receiver's fence for safely deregistering its destination.
            finished = self._writing[lease.transfer.transfer_id] = asyncio.Event()
        try:
            # Keep the deadline in the RPC task. A second cancellation must
            # never detach the native sender before it fences a posted write.
            async with asyncio.timeout(self._ttl):
                if use_nixl:
                    await self._write(lease.transfer.part_id, request.nixl_target)
                else:
                    await self._send(lease.transfer.part_id, request.receive_url)
        except asyncio.TimeoutError:
            send_error = "staged media delivery timed out"
        except Exception as exc:
            send_error = str(exc)
        finally:
            if use_nixl:
                self._writing.pop(lease.transfer.transfer_id)
                finished.set()
            await asyncio.shield(self._start_retirement(lease.transfer.part_id))
        if send_error is not None:
            await context.abort(grpc.StatusCode.UNAVAILABLE, send_error)
        logger.debug(
            "Deliver complete request_id=%s part_id=%s",
            lease.transfer.request_id,
            lease.transfer.part_id,
        )
        return pb.DeliverResponse()

    async def Release(self, request, context):
        lease = self._leases.pop(request.transfer_id, None)
        if lease is not None:
            lease.expiry.cancel()
            await asyncio.shield(self._start_retirement(lease.transfer.part_id))
        elif (finished := self._writing.get(request.transfer_id)) is not None:
            await finished.wait()
        return pb.ReleaseResponse()
