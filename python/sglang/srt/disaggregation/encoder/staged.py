"""One-shot gRPC handoff of an encoder-owned, already-produced image part."""

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
        self, *, encode, send, release, make_part_id, info, stage_timeout, ttl, capacity
    ):
        self._encode = encode
        self._send = send
        self._release = release
        self._make_part_id = make_part_id
        self._info = info
        self._stage_timeout = stage_timeout
        self._ttl = ttl
        self._capacity = capacity
        self._active = 0  # includes queued encodes, staged parts and active sends
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
                "Staged images require Qwen-VL, zmq_to_scheduler, TP=DP=1 "
                "and positive staging limits",
            )

    async def GetEncoderInfo(self, request, context):
        await self._check_config(context)
        return self._info

    async def Stage(self, request, context):
        await self._check_config(context)
        if not request.request_id or not request.mm_items or not all(request.mm_items):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "request_id and images are required"
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
            layouts = await asyncio.wait_for(
                self._encode(part_id, list(request.mm_items)), self._stage_timeout
            )
            if loop.time() >= deadline:
                raise asyncio.TimeoutError
            if len(layouts) != len(request.mm_items):
                raise ValueError("staged grid count differs from image count")
            transfer = pb.StagedImageTransfer(
                version=1,
                request_id=request.request_id,
                transfer_id=transfer_id,
                part_id=part_id,
                part_idx=0,
                num_parts=1,
                image_count=len(layouts),
                images=layouts,
            )
            response = pb.StageResponse(transfer=transfer)
            expiry = loop.call_later(self._ttl, self._expire, transfer_id)
            self._leases[transfer_id] = _Lease(transfer, expiry)
            logger.debug(
                "Stage complete request_id=%s part_id=%s images=%d",
                transfer.request_id,
                part_id,
                transfer.image_count,
            )
            return response
        except asyncio.CancelledError:
            await asyncio.shield(self._start_retirement(part_id))
            raise
        except asyncio.TimeoutError:
            await asyncio.shield(self._start_retirement(part_id))
            await context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED, "image staging timed out"
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

    async def Deliver(self, request, context):
        transfer_id = request.transfer.transfer_id
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
        if request.transfer != lease.transfer or not request.receive_url:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "invalid transfer identity or receiver",
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
        try:
            # Also bound clients that omit a gRPC deadline. The native ZMQ
            # sender drains any in-flight zero-copy operation on cancellation,
            # with its own finite SNDTIMEO and MessageTracker.wait timeout.
            await asyncio.wait_for(
                self._send(lease.transfer.part_id, request.receive_url),
                self._ttl,
            )
        except asyncio.TimeoutError:
            send_error = "staged image delivery timed out"
        except Exception as exc:
            send_error = str(exc)
        finally:
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
        return pb.ReleaseResponse()
