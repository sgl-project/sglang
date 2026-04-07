"""
Staging handler for heterogeneous TP KV cache transfer.

Isolates staging scatter lifecycle from decode.py and conn.py.
Generic (backend-agnostic) code is at the top; mooncake-specific
protocol code is at the bottom.
"""

from __future__ import annotations

import dataclasses
import logging
import struct
import threading
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest


# ======================================================================
# Generic staging state and handler (backend-agnostic)
# ======================================================================


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
    prefetch_requested: set = dataclasses.field(default_factory=set)
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
        self._room_to_decode_req: dict = {}
        self._wm_subscribers: dict = {}

    def register_wm_subscriber(self, receiver, session_id: str) -> None:
        """Register a prefill's bootstrap connection for watermark broadcasts."""
        if receiver is None or not getattr(receiver, "bootstrap_infos", None):
            return
        key = tuple(str(bi) for bi in receiver.bootstrap_infos)
        if key not in self._wm_subscribers:
            self._wm_subscribers[key] = (receiver, session_id)

    def num_writers_for(self, decode_req) -> int:
        """Compute num_writers for a specific request based on its prefill TP."""
        prefill_tp = decode_req.kv_receiver.prefill_info.attn_tp_size
        if prefill_tp > self.decode_tp:
            return prefill_tp // max(1, self.decode_tp)
        return 1

    @classmethod
    def create(cls, kv_manager, scheduler, tp_rank: int) -> "DecodeStagingHandler":
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

    def register_decode_req(self, room: int, decode_req: "DecodeRequest") -> None:
        self._room_to_decode_req[room] = decode_req

    def unregister_decode_req(self, room: int) -> None:
        self._room_to_decode_req.pop(room, None)

    # ------------------------------------------------------------------
    # Scatter submission: called from decode_thread (background)
    # ------------------------------------------------------------------

    def submit_chunk_scatter(
        self, room: int, chunk_idx: int, page_start: int, num_pages: int
    ) -> bool:
        """Submit scatter for an intermediate chunk whose writers all arrived.

        Called from decode_thread.  Records a CUDA event on decode_req so
        the main thread can later check completion and free the allocation.
        """
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
        chunk_infos = getattr(decode_req.kv_receiver, "chunk_staging_infos", [])
        if chunk_idx >= len(chunk_infos):
            return False
        alloc_id, staging_offset, _, _, _ = chunk_infos[chunk_idx]
        if staging_offset < 0 or alloc_id < 0:
            return False

        ok = self._scatter_region(staging_offset, page_start, num_pages, decode_req)
        if ok:
            event = torch.cuda.Event()
            event.record(self.staging_allocator._scatter_stream)
            if not hasattr(decode_req, "_chunk_events"):
                decode_req._chunk_events = []
            decode_req._chunk_events.append((event, alloc_id))
            chunk_infos[chunk_idx] = (-1, -1, 0, -1, 0)
        else:
            logger.warning(
                "submit_chunk_scatter failed room=%s chunk_idx=%s tp_rank=%s",
                room,
                chunk_idx,
                self.tp_rank,
            )
        return ok

    def is_staging_room(self, room: int) -> bool:
        """Check if a room is registered for staging scatter."""
        return room in self._room_to_decode_req

    def submit_last_scatter_async(self, room: int) -> bool:
        """Submit scatter for the last chunk when all ranks report Success.

        Called from decode_thread.  Sets ``_scatter_event`` **before**
        ``_staging_last_scatter_submitted`` so the main thread sees the
        event when it checks the flag (CPython GIL guarantees ordering).
        """
        decode_req = self._room_to_decode_req.get(room)
        if decode_req is None:
            logger.warning(
                "[STAGING] submit_last_scatter_async: room=%s not registered. "
                "This should not happen if register_decode_req is called at "
                "kv_receiver.init() time.",
                room,
            )
            return False
        alloc_id = self._submit_last_scatter(decode_req)
        if alloc_id >= 0:
            event = torch.cuda.Event()
            event.record(self.staging_allocator._scatter_stream)
            decode_req._scatter_event = event
            decode_req._scatter_alloc_id = alloc_id
            decode_req._staging_last_scatter_submitted = True
        else:
            decode_req._staging_scatter_done = True
        return True

    # ------------------------------------------------------------------
    # Event check + free: called from main thread (pop_transferred)
    # ------------------------------------------------------------------

    def is_done(self, decode_req: "DecodeRequest") -> bool:
        """Return True if staging scatter is complete for this request."""
        if not getattr(decode_req, "_staging_scatter_done", False):
            return False
        return not getattr(decode_req, "_chunk_events", None)

    def advance_scatter(self, decode_req: "DecodeRequest") -> None:
        """Check CUDA events and free completed staging allocations.

        Scatter kernels have already been submitted by the decode_thread
        (via submit_chunk_scatter / submit_last_scatter_async).  This
        method only polls the recorded events and releases staging memory.
        """
        room = decode_req.req.bootstrap_room
        chunk_events = getattr(decode_req, "_chunk_events", None)
        if chunk_events:
            for i in range(len(chunk_events) - 1, -1, -1):
                event, alloc_id = chunk_events[i]
                if event.query():
                    chunk_events.pop(i)
                    self._free_and_send_watermark(alloc_id, decode_req)

        if not getattr(decode_req, "_staging_last_scatter_submitted", False):
            return

        event = getattr(decode_req, "_scatter_event", None)
        if event is not None and event.query():
            self._free_and_send_watermark(decode_req._scatter_alloc_id, decode_req)
            decode_req._scatter_event = None
            decode_req._scatter_alloc_id = -1
            decode_req._staging_scatter_done = True

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _scatter_region(
        self,
        staging_offset: int,
        page_start: int,
        num_pages: int,
        decode_req: "DecodeRequest",
    ) -> bool:
        """Submit scatter kernels for a staging region to scatter_stream.

        May be called from the decode_thread (background).  All GPU work
        runs on scatter_stream so that the decode_thread never blocks on
        the default stream (which carries the main-thread forward pass).
        """
        from sglang.srt.disaggregation.common.staging_buffer import (
            scatter_staging_to_kv,
        )

        k_buffers = self.kv_buffer_info["k_buffers"]
        v_buffers = self.kv_buffer_info["v_buffers"]
        page_size = self.kv_buffer_info["page_size"]
        dst_tp_rank = self.kv_manager.kv_args.engine_rank % self.decode_tp

        device = k_buffers[0].device
        torch.cuda.set_device(device)

        if not hasattr(self.staging_allocator, "_scatter_stream"):
            self.staging_allocator._scatter_stream = torch.cuda.Stream(device=device)

        scatter_stream = self.staging_allocator._scatter_stream

        staging_view = self.staging_allocator.buffer.buffer[staging_offset:]

        req_pool_idx = decode_req.req.req_pool_idx
        token_start = page_start * page_size
        token_end = token_start + num_pages * page_size
        prefill_tp = decode_req.kv_receiver.prefill_info.attn_tp_size

        with torch.cuda.stream(scatter_stream):
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
            )

        return True

    def _submit_last_scatter(self, decode_req: "DecodeRequest") -> int:
        """Submit scatter for the last chunk. Returns alloc_id >= 0, or -1."""
        receiver = decode_req.kv_receiver
        chunk_infos = getattr(receiver, "chunk_staging_infos", [])
        if not chunk_infos:
            return -1

        last_info = chunk_infos[-1]
        alloc_id, staging_offset, _, _, last_num_pages = last_info
        if staging_offset < 0 or alloc_id < 0:
            return -1

        seq_len = len(decode_req.req.origin_input_ids)
        ps = self.scheduler.token_to_kv_pool_allocator.page_size
        total_pages = (seq_len + ps - 1) // ps
        page_start = total_pages - last_num_pages

        ok = self._scatter_region(
            staging_offset, page_start, last_num_pages, decode_req
        )
        return alloc_id if ok else -1

    def _free_and_send_watermark(
        self, alloc_id: int, decode_req: "DecodeRequest"
    ) -> None:
        """Free a staging allocation and broadcast watermark to all prefills."""
        self.staging_allocator.free(alloc_id)
        post_wm = self.staging_allocator.get_watermark()
        room = decode_req.req.bootstrap_room
        wm_round, wm_tail = post_wm
        wm_round_b = str(wm_round).encode("ascii")
        wm_tail_b = str(wm_tail).encode("ascii")
        for _key, (receiver, session_id) in list(self._wm_subscribers.items()):
            sid_b = session_id.encode("ascii")
            for bootstrap_info in receiver.bootstrap_infos:
                try:
                    sock, lock = receiver._connect_to_bootstrap_server(bootstrap_info)
                    with lock:
                        sock.send_multipart(
                            [b"WATERMARK", wm_round_b, wm_tail_b, sid_b]
                        )
                except Exception:
                    pass


def is_watermark_ready(
    staging_state, session_id: str, alloc_round: int, alloc_end: int
) -> bool:
    """Non-blocking check: is the staging region safe to write?"""
    if alloc_round <= 0:
        return True
    prev_round = alloc_round - 1
    wm_round, wm_tail = staging_state.remote_watermarks.get(session_id, (0, 0))
    return prev_round < wm_round or (prev_round == wm_round and alloc_end <= wm_tail)


# ======================================================================
# Mooncake-specific staging protocol and utilities
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

    @classmethod
    def from_zmq_fields(
        cls, msg: list, msg_start_offset: int
    ) -> Optional["StagingRegisterInfo"]:
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
        return cls(base_ptr=base_ptr, total_size=total_size)


class PrefillStagingStrategy:
    """Prefill-side staging transfer: readiness check + gather-RDMA execution.

    Encapsulates the decision logic (chunk index calculation, staging offset
    lookup, watermark readiness) and delegates actual RDMA to the kv_manager.
    """

    def __init__(self, kv_manager, staging_buffer):
        self.kv_manager = kv_manager
        self.staging_buffer = staging_buffer
        page_size = kv_manager.kv_buffer_tensors["page_size"]
        cps = kv_manager.server_args.chunked_prefill_size or 8192
        self.full_chunk_pages = max(1, cps // page_size)

    def check_ready(
        self,
        req,
        kv_chunk_index_start: int,
        num_chunk_pages: int,
    ) -> Tuple[bool, int, int, int, int]:
        """Check if staging offset and watermark are ready for this chunk.

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

        if not self.kv_manager._is_watermark_ready(
            req.mooncake_session_id, c_round, c_end
        ):
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

        Returns 0 on success, -1 to signal fallback to slice path.
        """
        try:
            return self.kv_manager.send_kvcache_staged(
                session_id,
                prefill_kv_indices,
                dst_staging_ptr,
                dst_staging_size,
                target_info.dst_tp_rank,
                target_info.dst_attn_tp_size,
                target_info.dst_kv_item_len,
                staging_buffer=self.staging_buffer,
            )
        except Exception as e:
            raise RuntimeError(
                f"[Staging] KV transfer via staging buffer failed: {e}. "
                f"session={session_id}"
            ) from e


def init_staging_buffers(engine, kv_args, count: int) -> list:
    """Create prefill-side staging buffers and register them with the engine.

    Returns list of StagingBuffer instances.
    """
    from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
    from sglang.srt.disaggregation.mooncake.utils import (
        init_mooncake_custom_mem_pool,
    )
    from sglang.srt.environ import envs

    size_mb = envs.SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB.get()
    size_bytes = size_mb * 1024 * 1024
    gpu_id = kv_args.gpu_id
    device = f"cuda:{gpu_id}"

    _, custom_mem_pool, pool_type = init_mooncake_custom_mem_pool(device)
    if custom_mem_pool is None:
        logger.info(
            "Staging buffer using cudaMalloc (no custom mem pool). "
            "This works for all GPU architectures. "
            "For NVLink/MNNVL transport, set SGLANG_MOONCAKE_CUSTOM_MEM_POOL."
        )

    buffers = []
    for _ in range(count):
        buf = StagingBuffer(size_bytes, device, gpu_id, custom_mem_pool=custom_mem_pool)
        engine.batch_register([buf.get_ptr()], [buf.get_size()])
        buffers.append(buf)
    return buffers


def init_staging_allocator(engine, kv_args):
    """Create decode-side staging ring-buffer allocator and register with engine.

    Returns a StagingAllocator instance.
    """
    from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator
    from sglang.srt.disaggregation.mooncake.utils import (
        init_mooncake_custom_mem_pool,
    )
    from sglang.srt.environ import envs

    pool_size_mb = envs.SGLANG_DISAGG_STAGING_POOL_SIZE_MB.get()
    pool_size_bytes = pool_size_mb * 1024 * 1024
    gpu_id = kv_args.gpu_id
    device = f"cuda:{gpu_id}"

    _, custom_mem_pool, _ = init_mooncake_custom_mem_pool(device)
    allocator = StagingAllocator(pool_size_bytes, device, gpu_id, custom_mem_pool)
    engine.batch_register([allocator.get_base_ptr()], [allocator.get_total_size()])
    return allocator


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
    infos = getattr(receiver, "chunk_staging_infos", [])

    if chunk_idx < len(infos) and infos[chunk_idx][0] >= 0:
        _, offset, rnd, end, _ = infos[chunk_idx]
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


def prefetch_staging_reqs(
    room: int,
    transfer_infos: dict,
    kv_buffer_tensors: dict,
    chunked_prefill_size: int,
    staging_requested: set,
    prefetch_sockets: dict,
) -> None:
    """Send STAGING_REQ for all chunks before the prefill forward starts.

    Called from the scheduler right after batch formation, so that decode
    allocates staging during the GPU forward pass.
    """
    import zmq

    from sglang.srt.utils.network import NetworkAddress

    page_size = kv_buffer_tensors["page_size"]
    cps = chunked_prefill_size or 8192
    full_chunk_pages = max(1, cps // page_size)

    for session_id, tinfo in transfer_infos[room].items():
        if tinfo.is_dummy:
            continue
        total_pages = len(tinfo.dst_kv_indices)
        if total_pages == 0:
            continue
        num_chunks = (total_pages + full_chunk_pages - 1) // full_chunk_pages

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
                    sock = zmq.Context().socket(zmq.PUSH)
                    if na.is_ipv6:
                        sock.setsockopt(zmq.IPV6, 1)
                    sock.connect(ep)
                    prefetch_sockets[ep] = sock
                prefetch_sockets[ep].send_multipart(
                    [
                        b"STAGING_REQ",
                        str(room).encode("ascii"),
                        str(chunk_idx).encode("ascii"),
                        str(chunk_pages).encode("ascii"),
                        session_id.encode("ascii"),
                    ]
                )
            except Exception:
                staging_requested.discard(stg_key)
