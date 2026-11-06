"""
Staging handler for CP and heterogeneous TP KV cache transfer.

Isolates staging scatter lifecycle from decode.py and conn.py.
Generic (backend-agnostic) code is at the top; mooncake-specific
protocol code is at the bottom.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import struct
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import zmq

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_disagg, get_schedule
from sglang.srt.utils import (
    create_device_event,
    create_device_stream,
    device_stream_context,
)

logger = logging.getLogger(__name__)

# Bounded wait for a watermark advance before re-enqueueing a deferred staging
# chunk, so the re-enqueue retry does not busy-spin a core.
STAGING_WATERMARK_WAIT_S = 0.001
# A capacity failure, distinct from a transport/registration/protocol error.
STAGING_MEMORY_UNAVAILABLE = -2

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest


# ======================================================================
# Generic staging state and handler (backend-agnostic)
# ======================================================================


def staging_enabled(*, is_mla: bool) -> bool:
    configured = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
    if (
        get_disagg().disaggregation_require_bulk
        or envs.SGLANG_DISAGG_REQUIRE_BULK.get()
    ) and not configured:
        raise ValueError(
            "--disaggregation-require-bulk requires SGLANG_DISAGG_STAGING_BUFFER=1"
        )
    if is_mla:
        if configured:
            raise ValueError("KV staging requires an MHA/GQA KV pool")
        return False
    if configured is not None:
        return configured
    # Preserve the 397B deployment default. Bulk staging changes the completion
    # protocol and must be enabled on both prefill and decode during rollout.
    return False


@dataclasses.dataclass
class DecodeStagingContext:
    """Staging-specific context for decode mode."""

    allocator: object = None
    room_bootstrap: dict = dataclasses.field(default_factory=dict)
    room_receivers: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PrefillStagingContext:
    """Staging-specific context for prefill mode."""

    buffers: list = dataclasses.field(default_factory=list)
    remote_watermarks: dict = dataclasses.field(default_factory=dict)
    watermark_cv: threading.Condition = dataclasses.field(
        default_factory=threading.Condition
    )
    # (room, chunk_idx, session_id) keys for chunks already requested.
    prefetch_requested: set = dataclasses.field(default_factory=set)
    # Rooms that have already had their full prefetch fan-out triggered. Used
    # to short-circuit per-room prefetch entry on every chunk after the first.
    prefetched_rooms: set = dataclasses.field(default_factory=set)
    prefetch_sockets: dict = dataclasses.field(default_factory=dict)


class DecodeStagingHandler:
    """Decode-side staging scatter lifecycle manager.

    Scatter submission can be called from the decode_thread (background) as
    soon as all writers/ranks have arrived, while event checking and freeing
    always run on the scheduler main thread.
    """

    def __init__(
        self,
        kv_manager,
        staging_allocator,
        kv_buffer_info: dict,
        decode_tp: int,
        total_kv_heads: int,
        tp_rank: int,
        scheduler,
    ):
        self.kv_manager = kv_manager
        self.staging_allocator = staging_allocator
        self.kv_buffer_info = kv_buffer_info
        self.decode_tp = decode_tp
        self.total_kv_heads = total_kv_heads
        self.tp_rank = tp_rank
        self.scheduler = scheduler
        # Submission, event publication and teardown must share a host lock.
        # Synchronizing a CUDA stream alone cannot exclude a future submission.
        self._lock = threading.RLock()
        # Same stall->Failed semantics (and knob) as _check_waiting_timeout,
        # which is unreachable once the receiver has concluded Success.
        self.completion_timeout = float(
            envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
        )
        self._room_to_decode_req: dict = {}
        # Stashed at registration: removal paths null decode_req.kv_receiver
        # before unregister runs, but release_room still needs it.
        self._room_to_receiver: dict = {}
        self._wm_subscribers: dict = {}
        self._pending_watermarks: dict = {}
        # room -> chunk_idx -> writer_id -> completion metadata. Keeping each
        # writer once prevents duplicated notifications from triggering scatter.
        self._writer_counts: dict = {}
        self._staging_lifecycle = threading.Condition(self._lock)
        self._room_tokens: dict[int, object] = {}
        self._inflight_submits: dict[object, int] = {}
        self._claimed_chunks: set[tuple[object, int]] = set()
        self._closing_rooms: set[int] = set()
        self._defer_scatter = envs.SGLANG_PLB_SCATTER_DEFER.get()
        # Keep each queued chunk tied to its original room generation.
        self._pending_scatters: deque[tuple[object, int, int, int, int, object]] = (
            deque()
        )

        k_buffers = kv_buffer_info["k_buffers"]
        assert k_buffers, "k_buffers must not be empty"
        self.device = k_buffers[0].device
        assert self.device.index is not None, (
            f"KV buffers must be on an indexed device, got {self.device}; "
            f"scatter would otherwise default to device 0."
        )
        self.device_module = torch.get_device_module(self.device)
        self.gpu_id = self.device.index
        self._completed_chunks: dict = {}

    def register_wm_subscriber(self, receiver, session_id: str) -> None:
        """Register a prefill's bootstrap connection for watermark broadcasts."""
        if receiver is None or not getattr(receiver, "bootstrap_infos", None):
            return
        key = self._watermark_peer_key(receiver)
        with self._lock:
            previous = self._wm_subscribers.get(key)
            if previous is not None and previous[1] == session_id:
                return
            self._wm_subscribers[key] = (receiver, session_id)
            # A new session must receive the current watermark even if no
            # allocation is freed after it registers.
            self._pending_watermarks[key] = self.staging_allocator.get_watermark()
            self._flush_watermarks()

    @staticmethod
    def _watermark_peer_key(receiver) -> tuple:
        return tuple(
            (info["rank_ip"], info["rank_port"]) for info in receiver.bootstrap_infos
        )

    def num_writers_for(self, receiver) -> int:
        """Compute all TP and PP writers expected for a staging chunk."""
        prefill_info = receiver.prefill_info
        prefill_tp = prefill_info.attn_tp_size
        if prefill_tp > self.decode_tp:
            tp_writers = prefill_tp // max(1, self.decode_tp)
        else:
            tp_writers = 1
        pp_writers = prefill_info.pp_size // self.kv_manager.pp_size
        cp_writers = (
            prefill_info.attn_cp_size
            if getattr(prefill_info, "ring_cp_sharded", False)
            else 1
        )
        return tp_writers * pp_writers * cp_writers

    @classmethod
    def create(cls, kv_manager, scheduler, tp_rank: int) -> DecodeStagingHandler:
        """Factory: create handler. Raises if staging infra is missing."""
        staging_allocator = kv_manager._staging_ctx.allocator
        if staging_allocator is None:
            raise RuntimeError(
                "Staging is enabled but kv_manager._staging_ctx.allocator is None. "
                "Check that the transfer backend correctly initializes the staging allocator."
            )
        kv_buffer_info = kv_manager.kv_buffer_tensors
        if kv_buffer_info is None:
            raise RuntimeError(
                "Staging is enabled but kv_manager.kv_buffer_tensors is None. "
                "Check that set_kv_buffer_tensors() was called during kv_manager init."
            )
        decode_tp = kv_manager.attn_tp_size

        from sglang.srt.disaggregation.common.staging_buffer import (
            resolve_total_kv_heads,
        )

        total_kv_heads = resolve_total_kv_heads(kv_manager.kv_args, decode_tp)
        return cls(
            kv_manager=kv_manager,
            staging_allocator=staging_allocator,
            kv_buffer_info=kv_buffer_info,
            decode_tp=decode_tp,
            total_kv_heads=total_kv_heads,
            tp_rank=tp_rank,
            scheduler=scheduler,
        )

    # ------------------------------------------------------------------
    # Registration: called from main thread (DecodeTransferQueue)
    # ------------------------------------------------------------------

    def register_decode_req(self, room: int, decode_req: DecodeRequest) -> None:
        # Called before send_metadata; extend can encounter the same request.
        with self._staging_lifecycle:
            if self._room_to_decode_req.get(room) is decode_req:
                return
        decode_req._staging_all_success = False
        decode_req._staging_success_ts = 0.0
        decode_req._staging_failed = False
        decode_req._staging_scatter_done = False
        decode_req._chunk_events = []
        # Cross-stream scatters must wait for the new req_to_token row.
        decode_req._req_to_token_ready = None
        if self.device.type == "xpu" or self._defer_scatter:
            decode_req._req_to_token_ready = create_device_event(self.device)
            decode_req._req_to_token_ready.record(self.scheduler.schedule_stream)
        with self._staging_lifecycle:
            while room in self._closing_rooms:
                self._staging_lifecycle.wait()
            self._room_tokens[room] = object()
            self._room_to_decode_req[room] = decode_req
            self._room_to_receiver[room] = decode_req.kv_receiver
        # Scatter offsets shift suffix-relative page_start by the decode prefix,
        # exact only when the prefix is page-aligned. Fail just this request on a
        # mismatch instead of raising, which would kill the prefill scheduler.
        page_size = self.kv_buffer_info["page_size"]
        if decode_req.req.kv.cache_protected_len % page_size != 0:
            logger.error(
                "[STAGING] decode prefix length %s is not page-aligned "
                "(page_size=%s); failing room=%s (staging scatter offsets "
                "would be wrong).",
                decode_req.req.kv.cache_protected_len,
                page_size,
                room,
            )
            decode_req._staging_failed = True

    def unregister_decode_req(self, room: int) -> None:
        with self._staging_lifecycle:
            while room in self._closing_rooms:
                self._staging_lifecycle.wait()
            self._closing_rooms.add(room)
            token = self._room_tokens.pop(room, None)
            decode_req = self._room_to_decode_req.pop(room, None)
            receiver = self._room_to_receiver.pop(room, None)
            self._writer_counts.pop(room, None)
            self._completed_chunks.pop(room, None)
            self._pending_scatters = deque(
                entry for entry in self._pending_scatters if entry[1] != room
            )
            while token is not None and self._inflight_submits.get(token, 0):
                self._staging_lifecycle.wait()

        try:
            if decode_req is not None:
                self.release_room(room, decode_req, receiver)
            if receiver is not None and getattr(receiver, "bootstrap_infos", None):
                key = self._watermark_peer_key(receiver)
                if not any(
                    getattr(peer, "bootstrap_infos", None)
                    and self._watermark_peer_key(peer) == key
                    for peer in self._room_to_receiver.values()
                ):
                    # Do not keep broadcasting to a P process after its last
                    # transfer ends. A later request registers the latest value.
                    self._wm_subscribers.pop(key, None)
                    self._pending_watermarks.pop(key, None)
            self.kv_manager._staging_ctx.room_receivers.pop(room, None)
            self.kv_manager._staging_ctx.room_bootstrap.pop(room, None)
        finally:
            with self._staging_lifecycle:
                self._closing_rooms.remove(room)
                self._staging_lifecycle.notify_all()

    def release_room(self, room: int, decode_req: DecodeRequest, receiver) -> None:
        """Free outstanding staging allocations of a room; no-op after a
        clean Success, releases watermark-pinning leaks on failure/abort."""
        # submit_chunk_scatter publishes its event before teardown reaches here.
        # Drain those scatters before freeing their staging and KV-pool slots.
        stream = self.staging_allocator._scatter_stream
        if stream is not None:
            stream.synchronize()
        if self._defer_scatter:
            for event, _ in decode_req._chunk_events:
                event.synchronize()
        chunk_infos = receiver.chunk_staging_infos if receiver is not None else []
        unscattered_allocs = []
        for chunk_idx, info in enumerate(chunk_infos):
            if info[0] >= 0:
                unscattered_allocs.append((chunk_idx, info[0]))
                chunk_infos[chunk_idx] = (-1, -1, 0, -1, 0)
        for chunk_idx, alloc_id in unscattered_allocs:
            logger.warning(
                "[STAGING] releasing unscattered staging allocation "
                "room=%s chunk=%s alloc_id=%s",
                room,
                chunk_idx,
                alloc_id,
            )
            self._free_and_send_watermark(alloc_id, decode_req)
        for _event, alloc_id in decode_req._chunk_events:
            self._free_and_send_watermark(alloc_id, decode_req)
        decode_req._chunk_events.clear()

    # ------------------------------------------------------------------
    # Scatter submission: called from decode_thread (background)
    # ------------------------------------------------------------------

    def flush_pending_scatters(self, stream) -> None:
        """Submit ready chunks between consecutive forwards on ``stream``."""
        while True:
            with self._staging_lifecycle:
                if not self._pending_scatters:
                    return
                room_token, room, chunk_idx, page_start, num_pages, mask = (
                    self._pending_scatters.popleft()
                )
                if self._room_tokens.get(room) is not room_token:
                    continue
            self.submit_chunk_scatter(
                room,
                chunk_idx,
                page_start,
                num_pages,
                stream=stream,
                room_token=room_token,
                writer_slot_mask=mask,
            )

    def submit_chunk_scatter(
        self,
        room: int,
        chunk_idx: int,
        page_start: int,
        num_pages: int,
        stream=None,
        room_token: object | None = None,
        writer_slot_mask=None,
    ) -> bool:
        """Submit a claimed chunk and publish its completion event."""
        from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator

        with self._staging_lifecycle:
            decode_req = self._room_to_decode_req.get(room)
            if decode_req is None:
                logger.warning(
                    "[STAGING] submit_chunk_scatter: room=%s not registered, "
                    "chunk_idx=%s. This should not happen if register_decode_req "
                    "is called at kv_receiver.init() time.",
                    room,
                    chunk_idx,
                )
                return False
            receiver = self._room_to_receiver.get(room)
            token = self._room_tokens.get(room)
            chunk_infos = receiver.chunk_staging_infos if receiver is not None else []
            if (
                token is None
                or (room_token is not None and token is not room_token)
                or chunk_idx >= len(chunk_infos)
            ):
                return False
            alloc_id, staging_offset, _, _, _ = chunk_infos[chunk_idx]
            if staging_offset == STAGING_MEMORY_UNAVAILABLE:
                chunk_infos[chunk_idx] = (
                    -1,
                    StagingAllocator.CHUNK_SCATTERED,
                    0,
                    -1,
                    0,
                )
                return True
            claim = (token, chunk_idx)
            if staging_offset < 0 or alloc_id < 0 or claim in self._claimed_chunks:
                return False
            self._claimed_chunks.add(claim)
            self._inflight_submits[token] = self._inflight_submits.get(token, 0) + 1

        try:
            ok = self._scatter_region(
                staging_offset,
                page_start,
                num_pages,
                decode_req,
                receiver,
                stream,
                writer_slot_mask,
            )
            if ok:
                event = create_device_event(self.device)
                event.record(
                    stream
                    if stream is not None
                    else self.staging_allocator._scatter_stream
                )
                # Append before zeroing so the completion check always sees either
                # the slot or the event.
                decode_req._chunk_events.append((event, alloc_id))
                chunk_infos[chunk_idx] = (
                    -1,
                    StagingAllocator.CHUNK_SCATTERED,
                    0,
                    -1,
                    0,
                )
            else:
                logger.warning(
                    "submit_chunk_scatter failed room=%s chunk_idx=%s tp_rank=%s",
                    room,
                    chunk_idx,
                    self.tp_rank,
                )
            return ok
        finally:
            with self._staging_lifecycle:
                self._claimed_chunks.remove(claim)
                remaining = self._inflight_submits[token] - 1
                if remaining:
                    self._inflight_submits[token] = remaining
                else:
                    del self._inflight_submits[token]
                    self._staging_lifecycle.notify_all()

    def is_staging_room(self, room: int) -> bool:
        """Check if a room is registered for staging scatter."""
        return room in self._room_to_decode_req

    def handle_chunk_arrived(
        self,
        room: int,
        chunk_idx: int,
        page_start: int,
        num_pages: int,
        writer_id: str,
        staged: bool = True,
        writer_index: Optional[int] = None,
        slot_indices: Optional[List[int]] = None,
    ) -> bool:
        """Fan in all writers, then submit or queue a complete chunk."""
        submit_now = False
        with self._staging_lifecycle:
            # Read from the stash, not decode_req.kv_receiver: a concurrent
            # teardown nulls the latter before unregister removes the room.
            receiver = self._room_to_receiver.get(room)
            room_token = self._room_tokens.get(room)
            if receiver is None or room_token is None:
                logger.warning(
                    "Staging chunk arrived for unregistered room=%s chunk=%d, skipping",
                    room,
                    chunk_idx,
                )
                return False
            completed = self._completed_chunks.setdefault(room, set())
            if chunk_idx in completed:
                return False
            room_counts = self._writer_counts.setdefault(room, {})
            arrivals = room_counts.setdefault(chunk_idx, {})
            arrival = (page_start, num_pages, staged, writer_index, slot_indices)
            if writer_id in arrivals:
                if arrivals[writer_id] != arrival:
                    raise ValueError(
                        "Conflicting staging writer completion: "
                        f"room={room}, chunk={chunk_idx}, writer={writer_id!r}, "
                        f"previous={arrivals[writer_id]!r}, incoming={arrival!r}"
                    )
                return False
            if any(a[:2] != arrival[:2] for a in arrivals.values()):
                received_ranges = {writer: a[:2] for writer, a in arrivals.items()}
                raise ValueError(
                    "Staging writers disagree on chunk page range: "
                    f"room={room}, chunk={chunk_idx}, writer={writer_id!r}, "
                    f"incoming={arrival[:2]!r}, received={received_ranges!r}"
                )
            arrivals[writer_id] = arrival
            if len(arrivals) < self.num_writers_for(receiver):
                return False
            mask = self._writer_slot_mask(receiver, arrivals.values())
            if self._defer_scatter:
                self._pending_scatters.append(
                    (room_token, room, chunk_idx, page_start, num_pages, mask)
                )
            else:
                submit_now = True
            completed.add(chunk_idx)
            del room_counts[chunk_idx]

        if submit_now:
            self.submit_chunk_scatter(
                room,
                chunk_idx,
                page_start,
                num_pages,
                room_token=room_token,
                writer_slot_mask=mask,
            )
        return True

    def _writer_slot_mask(self, receiver, arrivals):
        # New Mooncake messages identify the exact packed slots. A writer that
        # ran out of source memory wrote KV directly and must never be scattered.
        if all(a[3] is None for a in arrivals):
            return None  # Existing transport protocol.
        cp_size = (
            receiver.prefill_info.attn_cp_size
            if getattr(receiver.prefill_info, "ring_cp_sharded", False)
            else 1
        )
        writers = max(cp_size, receiver.prefill_info.attn_tp_size // self.decode_tp)
        slots = len(self.kv_buffer_info["k_buffers"]) * 2
        mask = [[False] * slots for _ in range(writers)]
        for _, _, staged, writer, indices in arrivals:
            if writer is None or not 0 <= writer < writers:
                raise ValueError("Invalid staging writer index")
            for index in indices or []:
                if not 0 <= index < slots:
                    raise ValueError("Invalid staging layer slot")
                if staged:
                    mask[writer][index] = True
        return mask

    def submit_last_scatter_async(self, room: int) -> bool:
        """Record all-ranks Success. Scatter is fully arrival-driven (every
        chunk, including the last); advance_scatter completes the room once
        no allocation is still waiting for its arrival."""
        with self._lock:
            decode_req = self._room_to_decode_req.get(room)
            if decode_req is None:
                logger.warning(
                    "[STAGING] submit_last_scatter_async: room=%s not registered. "
                    "This should not happen if register_decode_req is called at "
                    "kv_receiver.init() time.",
                    room,
                )
                return False
            if not decode_req._staging_all_success:
                # Set the timestamp before the flag so the deadline check never
                # reads a zero ts.
                decode_req._staging_success_ts = time.monotonic()
                decode_req._staging_all_success = True
            return True

    # ------------------------------------------------------------------
    # Event check + free: called from main thread (pop_transferred)
    # ------------------------------------------------------------------

    def is_done(self, decode_req: DecodeRequest) -> bool:
        """Return True if staging scatter is complete for this request."""
        return decode_req._staging_scatter_done and not decode_req._chunk_events

    def is_failed(self, decode_req: DecodeRequest) -> bool:
        """Return True if staging completion timed out for this request."""
        return decode_req._staging_failed

    def advance_scatter(self, decode_req: DecodeRequest) -> None:
        """Poll scatter events, free completed allocations, detect completion.

        The room is done once all ranks reported Success AND every allocation
        was scattered AND every event fired; gating on outstanding allocations
        keeps it open while a CHUNK_READY is still in flight after Success.
        Rooms incomplete past the disaggregation waiting timeout are failed.
        """
        with self._lock:
            self._flush_watermarks()
            chunk_events = decode_req._chunk_events
            if chunk_events:
                for i in range(len(chunk_events) - 1, -1, -1):
                    event, alloc_id = chunk_events[i]
                    if event.query():
                        chunk_events.pop(i)
                        self._free_and_send_watermark(alloc_id, decode_req)

            if not decode_req._staging_all_success:
                return
            room = decode_req.req.bootstrap_room
            receiver = self._room_to_receiver.get(room)
            chunk_infos = receiver.chunk_staging_infos if receiver is not None else []
            incomplete = bool(chunk_events) or any(info[0] >= 0 for info in chunk_infos)
            if not incomplete:
                decode_req._staging_scatter_done = True
                return
            elapsed = time.monotonic() - decode_req._staging_success_ts
            if elapsed > self.completion_timeout:
                logger.error(
                    "[STAGING] room=%s not complete %.0fs after all-ranks Success "
                    "(a scatter never arrived); failing the request.",
                    room,
                    elapsed,
                )
                decode_req._staging_failed = True

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _scatter_region(
        self,
        staging_offset: int,
        page_start: int,
        num_pages: int,
        decode_req: DecodeRequest,
        receiver,
        stream=None,
        writer_slot_mask=None,
    ) -> bool:
        """Scatter on the supplied forward stream or the background stream."""
        from sglang.srt.disaggregation.common.staging_buffer import (
            scatter_staging_to_kv,
        )

        k_buffers = self.kv_buffer_info["k_buffers"]
        v_buffers = self.kv_buffer_info["v_buffers"]
        page_size = self.kv_buffer_info["page_size"]
        dst_tp_rank = self.kv_manager.kv_args.engine_rank % self.decode_tp

        self.device_module.set_device(self.gpu_id)

        if stream is None:
            if self.staging_allocator._scatter_stream is None:
                self.staging_allocator._scatter_stream = create_device_stream(
                    self.device
                )
            scatter_stream = self.staging_allocator._scatter_stream
        else:
            scatter_stream = stream
        if decode_req._req_to_token_ready is not None:
            scatter_stream.wait_event(decode_req._req_to_token_ready)

        staging_view = self.staging_allocator.buffer.buffer[staging_offset:]

        req_pool_idx = decode_req.req.kv.req_pool_idx
        # page_start is suffix-relative (pages after the decode-side cached
        # prefix); req_to_token rows are absolute.
        prefix_tokens = decode_req.req.kv.cache_protected_len
        token_start = prefix_tokens + page_start * page_size
        token_end = token_start + num_pages * page_size
        prefill_tp = receiver.prefill_info.attn_tp_size

        with device_stream_context(scatter_stream):
            kv_indices = self.scheduler.req_to_token_pool.req_to_token[
                req_pool_idx, token_start:token_end
            ]
            if page_size > 1:
                page_idx_tensor = kv_indices[::page_size] // page_size
            else:
                page_idx_tensor = kv_indices

            scatter_staging_to_kv(
                staging_view,
                k_buffers,
                v_buffers,
                page_idx_tensor,
                page_size,
                prefill_tp,
                self.decode_tp,
                dst_tp_rank,
                self.total_kv_heads,
                cp_size=(
                    receiver.prefill_info.attn_cp_size
                    if getattr(receiver.prefill_info, "ring_cp_sharded", False)
                    else 1
                ),
                writer_slot_mask=(
                    torch.tensor(writer_slot_mask, device=self.device, dtype=torch.bool)
                    if writer_slot_mask is not None
                    else None
                ),
            )

        return True

    def _free_and_send_watermark(
        self, alloc_id: int, _decode_req: DecodeRequest
    ) -> None:
        """Free a staging allocation and broadcast watermark to all prefills."""
        self.staging_allocator.free(alloc_id)
        post_wm = self.staging_allocator.get_watermark()
        for key in self._wm_subscribers:
            self._pending_watermarks[key] = post_wm
        self._flush_watermarks()

    def _flush_watermarks(self) -> None:
        # Watermarks are cumulative: retain only the newest value per peer and
        # retry from scatter polling, including polls with no newly freed slots.
        for key, watermark in list(self._pending_watermarks.items()):
            receiver, session_id = self._wm_subscribers[key]
            if self._send_watermark(receiver, session_id, watermark):
                del self._pending_watermarks[key]

    @staticmethod
    def _send_watermark(receiver, session_id: str, watermark) -> bool:
        """Try every writer without blocking the decode scheduler on a peer."""
        wm_round, wm_tail = watermark
        wm_round_b = str(wm_round).encode("ascii")
        wm_tail_b = str(wm_tail).encode("ascii")
        sid_b = session_id.encode("ascii")
        sent = True
        for bootstrap_info in receiver.bootstrap_infos:
            try:
                sock, lock = receiver._connect_to_bootstrap_server(bootstrap_info)
                if not lock.acquire(blocking=False):
                    sent = False
                    continue
                try:
                    sock.send_multipart(
                        [b"WATERMARK", wm_round_b, wm_tail_b, sid_b],
                        flags=zmq.NOBLOCK,
                    )
                finally:
                    lock.release()
            except zmq.ZMQError:
                sent = False
        return sent


def is_watermark_ready(
    staging_state, session_id: str, alloc_round: int, alloc_end: int
) -> bool:
    """Non-blocking check: is the staging region safe to write?"""
    if alloc_round <= 0:
        return True
    prev_round = alloc_round - 1
    wm_round, wm_tail = staging_state.remote_watermarks.get(session_id, (0, 0))
    return prev_round < wm_round or (prev_round == wm_round and alloc_end <= wm_tail)


def handle_watermark_msg(staging_ctx, msg_parts) -> None:
    """Process a WATERMARK message and update remote watermark tracking."""
    wm_round = int(msg_parts[1].decode("ascii"))
    wm_tail = int(msg_parts[2].decode("ascii"))
    wm_session = msg_parts[3].decode("ascii") if len(msg_parts) > 3 else ""
    with staging_ctx.watermark_cv:
        prev = staging_ctx.remote_watermarks.get(wm_session, (0, 0))
        if (wm_round, wm_tail) > prev:
            staging_ctx.remote_watermarks[wm_session] = (
                wm_round,
                wm_tail,
            )
        staging_ctx.watermark_cv.notify_all()


def handle_staging_rsp(msg_parts, transfer_infos: dict) -> None:
    """Process a STAGING_RSP message and update transfer info with allocation."""
    stg_room = int(msg_parts[1].decode("ascii"))
    stg_chunk_idx = int(msg_parts[2].decode("ascii"))
    stg_offset = int(msg_parts[3].decode("ascii"))
    stg_round = int(msg_parts[4].decode("ascii"))
    stg_end = int(msg_parts[5].decode("ascii"))
    stg_session = msg_parts[6].decode("ascii")
    room_infos = transfer_infos.get(stg_room, {})
    tinfo = room_infos.get(stg_session)
    if tinfo is not None:
        if tinfo.staging is None:
            tinfo.staging = StagingTransferInfo()
        tinfo.staging.set_chunk(stg_chunk_idx, stg_offset, stg_round, stg_end)
    else:
        logger.warning(
            "STAGING_RSP RECV but tinfo=None room=%s chunk=%d session=%s",
            stg_room,
            stg_chunk_idx,
            stg_session,
        )


# ======================================================================
# Staging data structures and protocol utilities
# ======================================================================


@dataclasses.dataclass
class StagingTransferInfo:
    """Per-chunk staging allocation info attached to a TransferInfo."""

    offsets: List[int] = dataclasses.field(default_factory=lambda: [-1])
    rounds: List[int] = dataclasses.field(default_factory=lambda: [0])
    ends: List[int] = dataclasses.field(default_factory=lambda: [-1])

    def set_chunk(self, idx: int, offset: int, rnd: int, end: int):
        while len(self.offsets) <= idx:
            self.offsets.append(-1)
            self.rounds.append(0)
            self.ends.append(-1)
        self.offsets[idx] = offset
        self.rounds[idx] = rnd
        self.ends[idx] = end


@dataclasses.dataclass
class StagingRegisterInfo:
    """Staging buffer registration info attached to a KVArgsRegisterInfo."""

    base_ptr: int = 0
    total_size: int = 0
    # Version 0 peers advertise only the first entry stride and target count.
    # Version 1 describes the scatter order, which differs from registered
    # [K_target, V_target, K_draft, V_draft] pointer order.
    slot_layer_ids: List[int] = dataclasses.field(default_factory=list)
    layout_version: int = 0
    slot_ids: list = dataclasses.field(default_factory=list)
    slot_item_lens: List[int] = dataclasses.field(default_factory=list)

    def layout_bytes(self) -> bytes:
        from sglang.srt.disaggregation.common.staging_buffer import (
            validate_staging_kv_entries,
        )

        if self.layout_version != 1:
            raise ValueError("Cannot publish an unsupported staging layout version")
        validate_staging_kv_entries(self.slot_ids, self.slot_item_lens)
        return json.dumps(
            {"version": 1, "slots": self.slot_ids, "item_lens": self.slot_item_lens},
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_zmq_fields(
        cls,
        msg: list,
        msg_start_offset: int,
        layout_index: Optional[int] = None,
        slot_ids_index: Optional[int] = None,
    ) -> Optional[StagingRegisterInfo]:
        i = msg_start_offset
        base_ptr = (
            struct.unpack("Q", msg[i])[0] if len(msg) > i and len(msg[i]) == 8 else 0
        )
        total_size = (
            int(msg[i + 1].decode("ascii"))
            if len(msg) > i + 1 and len(msg[i + 1]) > 0
            else 0
        )
        if base_ptr == 0 and total_size == 0:
            return None
        layout_version, slot_ids, slot_item_lens = 0, [], []
        if layout_index is not None and len(msg) > layout_index and msg[layout_index]:
            from sglang.srt.disaggregation.common.staging_buffer import (
                validate_staging_kv_entries,
            )

            layout = json.loads(msg[layout_index])
            if (
                not isinstance(layout, dict)
                or type(layout.get("version")) is not int
                or layout["version"] != 1
            ):
                raise ValueError("Unsupported staging layout protocol version")
            slot_item_lens = layout.get("item_lens")
            if not isinstance(slot_item_lens, list):
                raise ValueError("Missing staging slot byte strides")
            slot_ids = validate_staging_kv_entries(layout.get("slots"), slot_item_lens)
            layout_version = 1
        return cls(
            base_ptr=base_ptr,
            total_size=total_size,
            slot_layer_ids=(
                list(
                    struct.unpack(
                        f"{len(msg[slot_ids_index]) // 8}Q", msg[slot_ids_index]
                    )
                )
                if slot_ids_index is not None
                and len(msg) > slot_ids_index
                and msg[slot_ids_index]
                else []
            ),
            layout_version=layout_version,
            slot_ids=slot_ids,
            slot_item_lens=slot_item_lens,
        )


class PrefillStagingStrategy:
    """Prefill-side staging transfer: readiness check + gather-RDMA execution.

    Encapsulates the decision logic (chunk index calculation, staging offset
    lookup, watermark readiness) and delegates actual RDMA to the kv_manager.
    """

    def __init__(self, kv_manager, staging_buffer):
        from sglang.srt.disaggregation.common.staging_buffer import (
            staging_grid_tokens,
        )

        self.kv_manager = kv_manager
        self.staging_buffer = staging_buffer
        page_size = kv_manager.kv_args.page_size
        self.full_chunk_pages = (
            staging_grid_tokens(get_schedule().chunked_prefill_size, page_size)
            // page_size
        )

    def check_ready(
        self,
        req,
        kv_chunk_index_start: int,
        num_chunk_pages: int,
        session_id: Optional[str] = None,
    ) -> Tuple[bool, int, int, int, int]:
        """Check if staging offset and watermark are ready for this chunk.

        Args:
            req: transfer request with a ``.staging`` attribute.
            kv_chunk_index_start: page-level start index for this chunk.
            num_chunk_pages: number of pages in this chunk.
            session_id: identifier used for watermark lookup. Falls back to
                ``req.mooncake_session_id`` when *None* (mooncake compat).

        Returns (ready, chunk_idx, offset, round, end).
        offset == ALLOC_OVERSIZED means permanent failure (fall back to slice).
        offset == -1 means allocation pending (re-enqueue).
        """
        from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator

        chunk_idx = (
            kv_chunk_index_start // self.full_chunk_pages
            if self.full_chunk_pages > 0
            else 0
        )

        stg = req.staging
        if stg is None or chunk_idx >= len(stg.offsets):
            return (False, chunk_idx, -1, 0, -1)

        c_offset = stg.offsets[chunk_idx]
        if c_offset == StagingAllocator.ALLOC_OVERSIZED:
            return (False, chunk_idx, StagingAllocator.ALLOC_OVERSIZED, 0, -1)
        if c_offset < 0:
            return (False, chunk_idx, -1, 0, -1)

        c_round = stg.rounds[chunk_idx]
        c_end = stg.ends[chunk_idx]

        if session_id is None:
            session_id = req.mooncake_session_id
        if not self.kv_manager._is_watermark_ready(session_id, c_round, c_end):
            return (False, chunk_idx, c_offset, c_round, c_end)

        return (True, chunk_idx, c_offset, c_round, c_end)

    def transfer(
        self,
        session_id: str,
        prefill_kv_indices,
        dst_staging_ptr: int,
        dst_staging_size: int,
        target_info,
    ) -> int:
        """Execute staged transfer (gather + RDMA).

        Mooncake returns STAGING_MEMORY_UNAVAILABLE for GPU capacity failures;
        registration, layout and transport failures propagate to the caller.
        """
        try:
            kwargs = {"staging_buffer": self.staging_buffer}
            if target_info.staging is not None and target_info.staging.layout_version:
                kwargs.update(
                    num_dst_target_kv_layers=-1,
                    dst_num_entries=len(target_info.dst_kv_ptrs),
                    dst_slot_ids=target_info.staging.slot_ids,
                    dst_slot_item_lens=target_info.staging.slot_item_lens,
                )
            else:
                kwargs["dst_slot_layer_ids"] = (
                    target_info.staging.slot_layer_ids
                    if target_info.staging is not None
                    else None
                )
            return self.kv_manager.send_kvcache_staged(
                session_id,
                prefill_kv_indices,
                dst_staging_ptr,
                dst_staging_size,
                target_info.dst_tp_rank,
                target_info.dst_attn_tp_size,
                target_info.dst_kv_item_len,
                target_info.dst_kv_layer_ids,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"[Staging] KV transfer via staging buffer failed: {e}. "
                f"session={session_id}"
            ) from e


def _get_custom_mem_pool(device: str):
    """Get custom memory pool for staging buffer allocation (backend-agnostic).

    Returns (custom_mem_pool, pool_type) tuple. custom_mem_pool may be None
    if no custom pool is configured.
    """
    from sglang.srt.disaggregation.mooncake.utils import (
        init_mooncake_custom_mem_pool,
    )

    _, custom_mem_pool, pool_type = init_mooncake_custom_mem_pool(device)
    if custom_mem_pool is None and torch.device(device).type == "xpu":
        logger.info(
            "Staging buffer using the default XPU allocator (no custom mem pool)."
        )
    elif custom_mem_pool is None:
        logger.info(
            "Staging buffer using cudaMalloc (no custom mem pool). "
            "This works for all GPU architectures. "
            "For NVLink/MNNVL transport, set SGLANG_MOONCAKE_CUSTOM_MEM_POOL."
        )
    return custom_mem_pool, pool_type


def _bare_device_type(device_type: str) -> str:
    """Strip any index, so server_args.device="cuda:0" cannot yield "cuda:0:0"."""
    return torch.device(device_type).type


def init_staging_buffers(
    register_fn,
    kv_args,
    count: int,
    chunked_prefill_size: int,
    *,
    device_type: str,
    allow_oom_fallback: bool = False,
) -> list:
    """Create prefill-side staging buffers, each sized to one prefill chunk.

    Sizing to one chunk (``chunked_prefill_size`` tokens of this rank's KV) means
    a chunk can never be too large for the buffer.
    """
    from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer

    full_chunk_pages = max(1, chunked_prefill_size // kv_args.page_size)
    size_bytes = full_chunk_pages * sum(kv_args.kv_item_lens)
    gpu_id = kv_args.gpu_id
    device = f"{_bare_device_type(device_type)}:{gpu_id}"

    custom_mem_pool, _ = _get_custom_mem_pool(device)

    buffers = []
    for _ in range(count):
        try:
            buf = StagingBuffer(
                size_bytes, device, gpu_id, custom_mem_pool=custom_mem_pool
            )
        except torch.OutOfMemoryError:
            if not allow_oom_fallback:
                raise
            logger.warning(
                "Staging GPU allocation failed (%s bytes); this worker uses direct KV writes",
                size_bytes,
            )
            buffers.append(None)
            continue
        register_fn(buf.get_ptr(), buf.get_size())
        buffers.append(buf)
    return buffers


def init_staging_allocator(
    register_fn, kv_args, *, device_type: str, allow_oom_fallback: bool = False
):
    """Create decode-side staging ring-buffer allocator and register with transport.

    ``register_fn`` is called as ``register_fn(ptr, size)`` to hand the region to
    the transport backend. Returns a StagingAllocator.
    """
    from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator
    from sglang.srt.environ import envs

    pool_size_mb = envs.SGLANG_DISAGG_STAGING_POOL_SIZE_MB.get()
    pool_size_bytes = pool_size_mb * 1024 * 1024
    gpu_id = kv_args.gpu_id
    device = f"{_bare_device_type(device_type)}:{gpu_id}"

    custom_mem_pool, _ = _get_custom_mem_pool(device)
    while pool_size_bytes > 0:
        try:
            allocator = StagingAllocator(
                pool_size_bytes, device, gpu_id, custom_mem_pool
            )
        except torch.OutOfMemoryError:
            if not allow_oom_fallback:
                raise
            # Keep bulk available with a smaller pool when possible. Chunks
            # larger than the surviving pool negotiate direct writes.
            pool_size_bytes //= 2
            if pool_size_bytes < 64 * 1024 * 1024:
                logger.warning("No GPU memory for KV staging; using direct KV writes")
                return None
            continue
        register_fn(allocator.get_base_ptr(), allocator.get_total_size())
        return allocator
    raise ValueError("Staging pool size must be positive")


def handle_staging_req(
    msg,
    staging_allocator,
    kv_args,
    attn_tp_size: int,
    prefill_attn_tp_size: int,
    kv_buffer_tensors,
    room_receivers: dict,
    room_bootstrap: dict,
):
    """Allocate staging for a chunk on-demand and send STAGING_RSP to prefill.

    Deduplicates: multiple prefill TP ranks requesting the same (room, chunk_idx)
    only allocate once.  Sends ALLOC_OVERSIZED on permanent failure.
    """
    from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator

    room = int(msg[1].decode("ascii"))
    chunk_idx = int(msg[2].decode("ascii"))
    chunk_num_pages = int(msg[3].decode("ascii"))
    session_id = msg[4].decode("ascii")
    requester_pp_rank = int(msg[5].decode("ascii")) if len(msg) > 5 else None

    if staging_allocator is None:
        logger.warning(
            "STAGING_REQ ignored: allocator is None room=%s chunk=%s",
            room,
            chunk_idx,
        )
        return

    receiver = room_receivers.get(room)
    if receiver is None:
        logger.warning(
            "STAGING_REQ dropped: no receiver for room=%s chunk=%s session=%s",
            room,
            chunk_idx,
            session_id,
        )
        return
    infos = receiver.chunk_staging_infos

    if chunk_idx < len(infos) and infos[chunk_idx][0] >= 0:
        _, offset, rnd, end, _ = infos[chunk_idx]
    elif (
        chunk_idx < len(infos)
        and infos[chunk_idx][1] == StagingAllocator.CHUNK_SCATTERED
    ):
        # Another prefill writer's request (sent on a separate socket) can land
        # after this chunk already scattered; a fresh allocation would never be
        # scattered and would pin the ring until the room times out.
        return
    elif (
        chunk_idx < len(infos)
        and infos[chunk_idx][1] == StagingAllocator.ALLOC_OVERSIZED
    ):
        offset, rnd, end = StagingAllocator.ALLOC_OVERSIZED, 0, -1
    else:
        from sglang.srt.disaggregation.common.staging_buffer import (
            compute_staging_layout,
            resolve_total_kv_heads,
        )

        page_size = kv_args.page_size
        kv_item_lens = kv_args.kv_item_lens
        num_kv_layers = len(kv_item_lens) // 2
        decode_bytes_per_token = kv_item_lens[0] // page_size
        total_kv_heads = resolve_total_kv_heads(kv_args, attn_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // max(1, attn_tp_size))
        bytes_per_head_per_token = decode_bytes_per_token // dst_heads_per_rank
        dst_tp_rank = kv_args.engine_rank % max(1, attn_tp_size)

        chunk_tokens = chunk_num_pages * page_size
        _, _, required = compute_staging_layout(
            prefill_attn_tp_size,
            attn_tp_size,
            dst_tp_rank,
            total_kv_heads,
            chunk_tokens,
            bytes_per_head_per_token,
            num_kv_layers,
            slot_head_bytes=[
                n // (page_size * dst_heads_per_rank) for n in kv_item_lens
            ],
            cp_size=(
                receiver.prefill_info.attn_cp_size
                if getattr(receiver.prefill_info, "ring_cp_sharded", False)
                else 1
            ),
        )
        result = staging_allocator.assign(required)
        if result is None:
            logger.error(
                "[STAGING_REQ] alloc failed room=%s chunk=%d (need %d bytes, "
                "buffer total=%d bytes). Increase SGLANG_DISAGG_STAGING_POOL_SIZE_MB.",
                room,
                chunk_idx,
                required,
                staging_allocator.total_size,
            )
            offset, rnd, end = StagingAllocator.ALLOC_OVERSIZED, 0, -1
            while len(infos) <= chunk_idx:
                infos.append((-1, -1, 0, -1, 0))
            infos[chunk_idx] = (
                -1,
                StagingAllocator.ALLOC_OVERSIZED,
                0,
                -1,
                chunk_num_pages,
            )
        else:
            alloc_id, offset, rnd = result
            end = offset + required
            while len(infos) <= chunk_idx:
                infos.append((-1, -1, 0, -1, 0))
            infos[chunk_idx] = (alloc_id, offset, rnd, end, chunk_num_pages)

    bootstrap_infos = room_bootstrap.get(room)
    if bootstrap_infos:
        for bi in bootstrap_infos:
            if requester_pp_rank is not None and bi["pp_rank"] != requester_pp_rank:
                continue
            try:
                sock, lock = receiver._connect_to_bootstrap_server(bi)
                with lock:
                    sock.send_multipart(
                        [
                            b"STAGING_RSP",
                            str(room).encode("ascii"),
                            str(chunk_idx).encode("ascii"),
                            str(offset).encode("ascii"),
                            str(rnd).encode("ascii"),
                            str(end).encode("ascii"),
                            session_id.encode("ascii"),
                        ]
                    )
            except Exception:
                pass


class StagingManagerMixin:
    """Shared STAGING_REQ handling for KV managers that support staging.

    Mixed into the managers whose decode thread receives STAGING_REQ messages
    (currently Mooncake and NIXL). Expects the concrete manager to provide
    ``_staging_handler``, ``_staging_ctx``, ``kv_args``, ``attn_tp_size`` and
    optionally ``kv_buffer_tensors``.
    """

    def _is_watermark_ready(
        self, session_id: str, alloc_round: int, alloc_end: int
    ) -> bool:
        return is_watermark_ready(self._staging_ctx, session_id, alloc_round, alloc_end)

    def _handle_staging_req(self, msg):
        room = int(msg[1].decode("ascii"))
        session_id = msg[4].decode("ascii")
        handler = self._staging_handler
        assert handler is not None, (
            "STAGING_REQ received before staging handler initialized"
        )
        with handler._lock:
            receiver = handler._room_to_receiver.get(room)
            if receiver is None:
                logger.warning("STAGING_REQ received for unregistered room=%s", room)
                return
            handle_staging_req(
                msg,
                self._staging_ctx.allocator,
                self.kv_args,
                self.attn_tp_size,
                receiver.prefill_info.attn_tp_size,
                getattr(self, "kv_buffer_tensors", None),
                self._staging_ctx.room_receivers,
                self._staging_ctx.room_bootstrap,
            )
            handler.register_wm_subscriber(receiver, session_id)


def prefetch_staging_reqs(
    room: int,
    transfer_infos: dict,
    kv_buffer_tensors: dict,
    chunked_prefill_size: int,
    staging_requested: set,
    prefetch_sockets: dict,
    requester_pp_rank: Optional[int] = None,
    upto_tokens: Optional[int] = None,
    logical_page_size: Optional[int] = None,
) -> None:
    """Send STAGING_REQ for all chunks before the prefill forward starts.

    Called from the scheduler right after batch formation, so that decode
    allocates staging during the GPU forward pass. With upto_tokens, only the
    chunks covering the first upto_tokens tokens are requested: the decode ring
    frees in FIFO order, so a long prefill that reserves all of its chunks up
    front holds back every later transfer until its last chunk is written.
    """
    import zmq

    from sglang.srt.disaggregation.common.staging_buffer import staging_grid_tokens
    from sglang.srt.utils.network import NetworkAddress

    page_size = logical_page_size or kv_buffer_tensors["page_size"]
    full_chunk_pages = staging_grid_tokens(chunked_prefill_size, page_size) // page_size

    for session_id, tinfo in transfer_infos[room].items():
        if tinfo.is_dummy:
            continue
        total_pages = len(tinfo.dst_kv_indices)
        if total_pages == 0:
            continue
        num_chunks = (total_pages + full_chunk_pages - 1) // full_chunk_pages
        if upto_tokens is not None:
            upto_pages = (upto_tokens + page_size - 1) // page_size
            num_chunks = min(
                num_chunks, (upto_pages + full_chunk_pages - 1) // full_chunk_pages
            )

        for chunk_idx in range(num_chunks):
            stg_key = (room, chunk_idx, session_id)
            if stg_key in staging_requested:
                continue
            staging_requested.add(stg_key)

            remaining = total_pages - chunk_idx * full_chunk_pages
            chunk_pages = min(full_chunk_pages, remaining)
            try:
                na = NetworkAddress(tinfo.endpoint, tinfo.dst_port)
                ep = na.to_tcp()
                if ep not in prefetch_sockets:
                    sock = zmq.Context.instance().socket(zmq.PUSH)
                    if na.is_ipv6:
                        sock.setsockopt(zmq.IPV6, 1)
                    sock.connect(ep)
                    prefetch_sockets[ep] = sock
                request = [
                    b"STAGING_REQ",
                    str(room).encode("ascii"),
                    str(chunk_idx).encode("ascii"),
                    str(chunk_pages).encode("ascii"),
                    session_id.encode("ascii"),
                ]
                if requester_pp_rank is not None:
                    request.append(str(requester_pp_rank).encode("ascii"))
                prefetch_sockets[ep].send_multipart(request)
            except Exception:
                staging_requested.discard(stg_key)
