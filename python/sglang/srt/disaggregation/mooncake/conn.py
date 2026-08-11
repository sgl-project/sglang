from __future__ import annotations

import concurrent.futures
import dataclasses
import itertools
import logging
import os
import queue
import struct
import threading
import time
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import zmq
from prometheus_client import Counter

from sglang.srt.disaggregation.base.conn import (
    KVArgs,
    KVPoll,
    KVTransferBarrierEscalation,
    StateType,
)
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
    KVTransferError,
)
from sglang.srt.disaggregation.common.staging_handler import (
    STAGING_WATERMARK_WAIT_S,
    DecodeStagingContext,
    PrefillStagingContext,
    StagingTransferInfo,
)
from sglang.srt.disaggregation.common.utils import (
    AuxDataCodec,
    FastQueue,
    TransferKVChunk,
    build_dcp_token_transfer_plan,
    group_concurrent_contiguous,
    pack_int_lists,
    submit_transfer_calls,
    unpack_int_lists,
)
from sglang.srt.disaggregation.mooncake.utils import (
    check_mooncake_custom_mem_pool_enabled,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    build_transfer_entry_pairs,
    compute_mamba_state_slice_byte_blocks,
    resolve_dcp_dst_entry_indices,
)
from sglang.srt.distributed.parallel_state import get_mooncake_transfer_engine
from sglang.srt.environ import envs
from sglang.srt.observability.mooncake_trace import (
    MooncakeRequestStage,
    mooncake_trace_func,
    mooncake_trace_slice,
)
from sglang.srt.observability.trace import (
    TraceNullContext,
    TraceReqContext,
    trace_set_thread_info,
)
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)

FAILED_SESSION_RECOVERIES = Counter(
    "sglang:failed_session_recoveries_total",
    "Number of mooncake_session_ids un-blacklisted via probe.",
)

TRANSFER_QUIESCE_TIMEOUTS = Counter(
    "sglang:kv_transfer_quiesce_timeouts_total",
    "Requests whose KV transfer quiescence deadline expired.",
)

# How long a failed request may wait for proof of transfer quiescence before
# the wait is reported as an error. The pages are never released without proof;
# this only bounds how long the condition stays silent. Keep it above the
# manager socket's 30s ZMQ send timeout, because a transfer holds its room
# while notifying the peer; a shorter value reports spurious timeouts whenever
# a decode stops reading.
QUIESCE_TIMEOUT_S = 60.0
# How many requests may be stuck without proof of quiescence before the worker
# fails, so that a restart releases every withheld page safely at once.
MAX_UNQUIESCED_ROOMS = 256
# An error return from a Mooncake engine call is NOT proof that its RDMA work
# stopped: the engine has no cancellation API, its sync wrapper's deadline
# return deliberately leaves the batch running (a known issue acknowledged in
# the engine's own source), and a failed batch is resubmitted internally so an
# earlier generation's writes can still land after the call returns. Posted
# work drains within the QP retransmit horizon times the engine's software
# re-posts, so a room whose engine call failed only counts as drained after
# this additional quarantine. Success returns need none: COMPLETED requires a
# terminal completion for every slice.
ENGINE_FAILURE_QUARANTINE_S = 30.0
# How often a decode rank re-sends an unacknowledged abort notification.
ABORT_RETRY_INTERVAL_S = 0.5
# How often the bookkeeping thread re-checks drained rooms and sweeps lifetimes.
TRANSFER_BOOKKEEPING_INTERVAL_S = 0.01
# How often that thread runs the (cheap) room-lifetime sweep.
ROOM_SWEEP_INTERVAL_S = 1.0
# How long an ABORT_ACK may stay owed to a peer that is not reading its socket.
ABORT_ACK_MAX_AGE_S = 60.0
# Hard cap on owed ABORT_ACKs, so an abort storm against a dead peer cannot grow
# the pending list, or the per-tick sweep over it, without bound.
MAX_PENDING_ABORT_ACKS = 4096
# Soft cap on tracked room lifetimes. Reaching it means the periodic sweep is not
# keeping up, so an insert also does a *bounded* scan for reclaimable rooms.
MAX_TRACKED_ROOMS = 4096
# Entries examined per emergency scan, so an insert can never become O(n).
MAX_EMERGENCY_SCAN = 256


class RoomTransferLifetime:
    """Ownership barrier for one bootstrap room's KV pages on this rank.

    Mooncake transfers hand raw KV pointers to native code (RDMA, the transfer
    executor, CUDA staging copies). Those readers and writers outlive the
    Python call that started them, so a request that becomes terminal while
    they run would let the allocator hand the same pages to another request.

    Every piece of native transfer work holds a *lease*. ``close()`` stops new
    leases from being handed out; the room is *quiesced* once it is closed and
    every outstanding lease has been returned. Only then is it safe to release
    the room's KV pages.

    The proof this provides is at the transfer-engine API boundary, and that
    contract is asymmetric. A success return means every slice reached a
    terminal completion, so no posted RDMA work can still land. An error
    return proves nothing: the engine has no cancellation, and its deadline
    and retry paths return with work still posted or queued for re-posting.
    A lease returned by a *failed* engine call therefore only counts toward
    quiescence after ``ENGINE_FAILURE_QUARANTINE_S``, which bounds the
    engine's residual drain.

    Abort tokens minted by decode peers are recorded here so that a late or
    duplicated abort for a recycled room cannot close a live room.
    """

    __slots__ = (
        "_cond",
        "_leases",
        "_open",
        "_abort_tokens",
        "created_at",
        "_claimed",
        "_quarantine_until",
    )

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._leases = 0
        self._open = True
        self._abort_tokens: set = set()
        self.created_at = time.monotonic()
        # True once a local sender has taken responsibility for the room. An
        # unclaimed room was created by decode metadata alone, so nothing will
        # ever release it and the sweep must.
        self._claimed = False
        # Monotonic deadline before which the room must not report quiesced,
        # armed whenever an engine call for this room returns an error.
        self._quarantine_until = 0.0

    def try_lease(self) -> bool:
        """Take a lease, or return False if the room no longer admits work."""
        with self._cond:
            if not self._open:
                return False
            self._leases += 1
            return True

    def end_lease(self) -> None:
        with self._cond:
            self._leases -= 1
            if self._leases == 0:
                self._cond.notify_all()

    def close(self) -> None:
        """Stop admitting transfer work (idempotent)."""
        with self._cond:
            self._open = False
            if self._leases == 0:
                self._cond.notify_all()

    def is_closed(self) -> bool:
        with self._cond:
            return not self._open

    def quarantine(self, duration_s: float) -> None:
        """Withhold quiescence for *duration_s* from now.

        Called when an engine call for this room returns an error: the engine
        cannot cancel posted RDMA work, so the returned lease alone is not
        proof of drain and the room must stay owned while the engine's
        residual work runs out.
        """
        with self._cond:
            self._quarantine_until = max(
                self._quarantine_until, time.monotonic() + duration_s
            )

    def _quiesced_locked(self) -> bool:
        return (
            not self._open
            and self._leases == 0
            and time.monotonic() >= self._quarantine_until
        )

    def is_quiesced(self) -> bool:
        with self._cond:
            return self._quiesced_locked()

    def outstanding_leases(self) -> int:
        with self._cond:
            return self._leases

    def claim(self) -> None:
        with self._cond:
            self._claimed = True

    def is_claimed(self) -> bool:
        with self._cond:
            return self._claimed

    def is_reclaimable(self) -> bool:
        """Whether this room has no active local transfer ownership.

        A quiesced room admits no work and has none running (including any
        engine-failure quarantine). An unclaimed room has no sender *yet*, so
        callers must additionally preserve it for the bootstrap grace period
        in which a metadata-late sender may still arrive.
        """
        with self._cond:
            return self._quiesced_locked() or not self._claimed

    def add_abort_token(self, token: bytes) -> None:
        if not token:
            return
        with self._cond:
            self._abort_tokens.add(token)

    def authorizes_abort(self, token: bytes) -> bool:
        """Whether *token* may close this room.

        A tokenless abort cannot be authenticated, and is honoured anyway:
        closing the room only fails one request (availability), while dropping
        the abort would leave this rank free to transfer into pages the decode
        has already released (the corruption this barrier exists to prevent).

        A room that has not received any decode metadata yet has no tokens to
        compare against, and accepts the abort for the same fail-safe reason:
        an abort can legitimately arrive before this rank has any metadata for
        the room (the decode gave up during bootstrap). Accepting risks one
        spurious request failure if a recycled bootstrap_room draws a delayed
        abort from its previous occupant, which requires a collision in a
        64-bit space. Distinguishing the two cases would need a generation tag
        on every room-scoped message; that is not worth a wire change here.
        """
        if not token:
            return True
        with self._cond:
            return not self._abort_tokens or token in self._abort_tokens


@dataclasses.dataclass
class _PendingAbortAck:
    """An ABORT_ACK owed to a decode rank once its room has drained."""

    room: int
    endpoint: str
    dst_port: int
    token: bytes
    lifetime: Optional[RoomTransferLifetime]
    # Give up after this: a peer that has not drained its socket for this long is
    # gone, and it has its own barrier deadline. Retrying forever would grow the
    # pending list, and the sweep over it, without bound.
    deadline: float = 0.0

    def key(self) -> Tuple[int, str, int, bytes]:
        return (self.room, self.endpoint, self.dst_port, self.token)


# decode
@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    dst_state_indices: List[List[int]]  # parallel to receiver's state_types
    required_dst_info_num: int
    is_dummy: bool
    decode_prefix_len: Optional[int] = None
    dst_device_kv_indices: Optional[npt.NDArray[np.int32]] = None
    # Nonce this decode rank will require us to echo in ABORT_ACK. Empty when the
    # peer predates the ownership-barrier protocol.
    abort_token: bytes = b""
    # Note: always put the optional staging field at the final (it will be set through 'STAGING_RSP' pkg when needed)
    staging: Optional[StagingTransferInfo] = None

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        if msg[4] == b"" and msg[5] == b"":
            is_dummy = True
            dst_kv_indices = np.array([], dtype=np.int32)
            dst_aux_index = None
            dst_state_indices = []
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int32)
            dst_aux_index = int(msg[5].decode("ascii"))
            dst_state_indices = unpack_int_lists(msg[6], "i")
            is_dummy = False
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            dst_state_indices=dst_state_indices,
            required_dst_info_num=int(msg[7].decode("ascii")),
            is_dummy=is_dummy,
            decode_prefix_len=(
                int(msg[8].decode("ascii")) if len(msg) > 8 and msg[8] != b"" else None
            ),
            dst_device_kv_indices=(
                np.frombuffer(msg[9], dtype=np.int32)
                if len(msg) > 9 and msg[9] != b""
                else None
            ),
            abort_token=msg[10] if len(msg) > 10 else b"",
        )


# decode
@dataclasses.dataclass
class KVArgsRegisterInfo:
    room: str
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]
    dst_state_data_ptrs: List[List[int]]  # parallel to state_types (same below)
    dst_tp_rank: int
    dst_attn_tp_size: int
    dst_kv_item_len: int
    # for mamba state different tp slice transfer
    dst_state_item_lens: List[List[int]]
    dst_state_dim_per_tensor: List[List[int]]
    dst_kv_layer_ids: List[int]
    dst_state_layer_ids: List[List[int]]
    dst_dcp_size: int = 1
    dst_dcp_rank: int = 0
    requires_dcp_relayout: bool = False
    dcp_token_item_lens: Optional[List[int]] = None
    staging_base_ptr: int = 0
    staging_total_size: int = 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
            dst_state_data_ptrs=unpack_int_lists(msg[6], "Q"),
            dst_tp_rank=int(msg[7].decode("ascii")),
            dst_attn_tp_size=int(msg[8].decode("ascii")),
            dst_kv_item_len=int(msg[9].decode("ascii")),
            dst_state_item_lens=(
                unpack_int_lists(msg[10], "I") if len(msg) > 10 else []
            ),
            dst_state_dim_per_tensor=(
                unpack_int_lists(msg[11], "I") if len(msg) > 11 else []
            ),
            dst_kv_layer_ids=(
                list(struct.unpack(f"{len(msg[12]) // 4}I", msg[12]))
                if len(msg) > 12 and msg[12] != b""
                else []
            ),
            dst_state_layer_ids=(
                unpack_int_lists(msg[13], "I")
                if len(msg) > 13 and msg[13] != b""
                else []
            ),
            staging_base_ptr=(
                struct.unpack("Q", msg[14])[0]
                if len(msg) > 14 and len(msg[14]) == 8
                else 0
            ),
            staging_total_size=(
                int(msg[15].decode("ascii")) if len(msg) > 15 and msg[15] != b"" else 0
            ),
            dst_dcp_size=(
                int(msg[16].decode("ascii")) if len(msg) > 16 and msg[16] != b"" else 1
            ),
            dst_dcp_rank=(
                int(msg[17].decode("ascii")) if len(msg) > 17 and msg[17] != b"" else 0
            ),
        )


class MooncakeKVManager(CommonKVManager):
    AUX_DATA_HEADER = b"AUX_DATA"

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self._unquiesced_rooms: set = set()
        self._unquiesced_lock = threading.Lock()
        self.init_engine()
        self.register_buffer_to_engine()
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.enable_trace = server_args.enable_trace
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # room -> ownership barrier for that room's KV pages. Shares its
            # lifecycle with self.transfer_infos: created when the room is first
            # seen (by the sender or by decode metadata, whichever comes first)
            # and dropped by MooncakeKVSender.clear().
            self._room_lifetimes: OrderedDict[int, RoomTransferLifetime] = OrderedDict()
            self._room_lifetimes_lock = threading.Lock()
            self._start_transfer_bookkeeping()
            self.start_prefill_thread()
            self.session_failures = defaultdict(int)
            self.failed_sessions = set()
            # Per-room count of chunks not yet transferred; teardown waits for
            # zero so a deferred chunk is not dropped by an early conclude.
            self._staging_outstanding = defaultdict(int)
            self.session_lock = threading.Lock()
            # Determine the number of threads to use for kv sender
            cpu_count = os.cpu_count()
            transfer_thread_pool_size = (
                envs.SGLANG_DISAGGREGATION_THREAD_POOL_SIZE.get()
            )
            if transfer_thread_pool_size is None:
                transfer_thread_pool_size = min(max(4, int(0.5 * cpu_count) // 8), 12)
            transfer_queue_size = envs.SGLANG_DISAGGREGATION_QUEUE_SIZE.get()
            self.transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(transfer_queue_size)
            ]
            assert transfer_thread_pool_size >= transfer_queue_size, (
                f"The environment variable SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} must be "
                f"greater than or equal to SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}."
            )
            self.executors = [
                concurrent.futures.ThreadPoolExecutor(
                    transfer_thread_pool_size // transfer_queue_size
                )
                for _ in range(transfer_queue_size)
            ]
            self.enable_custom_mem_pool, self.custom_mem_pool_type = (
                check_mooncake_custom_mem_pool_enabled()
            )
            self._staging_ctx = PrefillStagingContext() if self.enable_staging else None
            if self.enable_staging:
                self._init_staging_buffers(len(self.transfer_queues))
            for i, (queue, executor) in enumerate(
                zip(self.transfer_queues, self.executors)
            ):
                threading.Thread(
                    target=self.transfer_worker,
                    args=(
                        queue,
                        executor,
                        (
                            self._staging_ctx.buffers[i]
                            if self.enable_staging and self._staging_ctx.buffers
                            else None
                        ),
                        i,
                    ),
                    daemon=True,
                ).start()
            self.enable_failed_session_probe = (
                envs.SGLANG_ENABLE_FAILED_SESSION_PROBE.get()
            )
            if self.enable_failed_session_probe:
                self.failed_session_probe_interval = (
                    envs.SGLANG_FAILED_SESSION_PROBE_INTERVAL_S.get()
                )
                self._failed_session_probe_shutdown = threading.Event()
                threading.Thread(
                    target=self._failed_session_probe_loop,
                    name="MooncakeFailedSessionProbe",
                    daemon=True,
                ).start()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # room -> live receiver. Used to route ABORT_ACKs and to drop late
            # transport messages for rooms that are tearing down.
            self._receivers: Dict[int, MooncakeKVReceiver] = {}
            self._receivers_lock = threading.Lock()
            self._staging_ctx = DecodeStagingContext() if self.enable_staging else None
            if self.enable_staging:
                self._init_staging_allocator()
                self._staging_handler = None
            self.start_decode_thread()

    # ------------------------------------------------------------------
    # Transfer ownership barrier (prefill side)
    # ------------------------------------------------------------------

    def _room_lifetime(
        self, room: int, *, create: bool
    ) -> Optional[RoomTransferLifetime]:
        with self._room_lifetimes_lock:
            return self._room_lifetime_locked(room, create=create)

    def _room_lifetime_locked(
        self, room: int, *, create: bool
    ) -> Optional[RoomTransferLifetime]:
        """Return a room lifetime while the caller holds its generation lock."""
        lifetime = self._room_lifetimes.get(room)
        if lifetime is None and create:
            # Reclaim before inserting, so the entry being created can never be
            # the one that gets evicted.
            if len(self._room_lifetimes) >= MAX_TRACKED_ROOMS:
                self._emergency_reclaim_rooms_locked()
            lifetime = self._room_lifetimes[room] = RoomTransferLifetime()
        return lifetime

    def _retire_rooms_locked(self, rooms: List[int]) -> None:
        """Forget every piece of state keyed by these rooms, atomically.

        Must happen under ``_room_lifetimes_lock`` and in one step: dropping the
        lifetime first and the rest afterwards would let a request that draws the
        same bootstrap_room in between have its own destinations and status
        deleted, and leaving them behind would let it inherit the old decode's
        addresses.
        """
        for room in rooms:
            del self._room_lifetimes[room]
            self.transfer_infos.pop(room, None)
            self.req_to_decode_prefix_len.pop(room, None)
            self.request_status.pop(room, None)

    def _emergency_reclaim_rooms_locked(self) -> None:
        """Bounded reclaim when the periodic sweep is not keeping up.

        Scans at most ``MAX_EMERGENCY_SCAN`` of the oldest entries so an insert
        can never degrade to O(tracked rooms); the sweep does the rest. The
        normal bootstrap grace period still applies: the cap is soft when every
        candidate may still be waiting for its local sender.
        """
        # islice over the live view keeps this O(MAX_EMERGENCY_SCAN); the keys are
        # copied out first because the dict cannot be mutated while iterating.
        cutoff = time.monotonic() - self._room_sweep_ttl
        candidates = [
            room
            for room, lifetime in itertools.islice(
                self._room_lifetimes.items(), MAX_EMERGENCY_SCAN
            )
            if lifetime.created_at <= cutoff and lifetime.is_reclaimable()
        ]
        self._retire_rooms_locked(candidates)
        if not candidates:
            logger.warning_once(
                "Tracking more than %d Mooncake bootstrap rooms with none old "
                "enough and reclaimable; allowing the soft cap to grow while "
                "metadata-first rooms remain inside their bootstrap window.",
                MAX_TRACKED_ROOMS,
            )

    def _sweep_room_lifetimes(self) -> int:
        """Reclaim rooms that no local sender will ever release.

        Entries younger than the bootstrap timeout are always kept: that is the
        window in which a sender may still appear and claim the room, and in
        which a tombstone must keep rejecting a late abort. After it, a quiesced
        or never-claimed room is dead weight.
        """
        cutoff = time.monotonic() - self._room_sweep_ttl
        with self._room_lifetimes_lock:
            stale = [
                room
                for room, lifetime in self._room_lifetimes.items()
                if lifetime.created_at <= cutoff and lifetime.is_reclaimable()
            ]
            self._retire_rooms_locked(stale)
        if stale:
            logger.debug("Reclaimed %d abandoned Mooncake rooms", len(stale))
        return len(stale)

    def _forget_room_lifetime(self, room: int) -> None:
        with self._room_lifetimes_lock:
            self._room_lifetimes.pop(room, None)

    def try_lease_room(self, room: int) -> Optional[RoomTransferLifetime]:
        """Take a transfer lease on *room*, or None if it admits no more work.

        Deliberately does not create a missing lifetime: an absent room has
        already been released, so admitting work would reintroduce the
        use-after-free this barrier exists to prevent. Callers must return the
        lease with ``end_lease()`` from a ``finally`` block.
        """
        lifetime = self._room_lifetime(room, create=False)
        if lifetime is None or not lifetime.try_lease():
            return None
        return lifetime

    def try_lease_chunk(
        self, kv_chunk: TransferKVChunk
    ) -> Optional[RoomTransferLifetime]:
        """Lease the room for *kv_chunk*, matching on request identity.

        A bootstrap_room is recycled, so a chunk that outlived its request must
        not attach to whichever request holds that room now: its indices name the
        old request's pages and its destinations the old decode's. The chunk
        carries the lifetime it was queued against, and only that exact object
        may be leased.
        """
        owner = kv_chunk.owner
        lifetime = self._room_lifetime(kv_chunk.room, create=False)
        if lifetime is None or (owner is not None and owner is not lifetime):
            return None
        if not lifetime.try_lease():
            return None
        return lifetime

    def open_room_transfers(self, room: int) -> bool:
        """Claim *room*'s ownership barrier for a local sender.

        Returns False when the room has already been closed, which happens when
        a decode abort overtakes this rank's sender: transferring then would
        write into pages the decode instance has released.
        """
        lifetime = self._room_lifetime(room, create=True)
        lifetime.claim()
        return not lifetime.is_closed()

    def close_room_transfers(self, room: int) -> Optional[RoomTransferLifetime]:
        """Stop admitting transfers for *room* (no-op if already released)."""
        lifetime = self._room_lifetime(room, create=False)
        if lifetime is not None:
            lifetime.close()
        return lifetime

    def _close_room_for_abort(
        self, room: int, token: bytes
    ) -> Optional[RoomTransferLifetime]:
        """Close *room* on behalf of a decode abort, if the token authorizes it.

        The lifetime is created when missing so that an abort which overtakes
        the prefill sender still leaves a tombstone; the sender then fails the
        request instead of transferring into pages the decode has released.
        """
        lifetime = self._room_lifetime(room, create=False)
        if lifetime is not None and not lifetime.authorizes_abort(token):
            logger.debug(
                "Ignoring abort for room %s: token was not issued for this room",
                room,
            )
            return None
        if lifetime is None:
            lifetime = self._room_lifetime(room, create=True)
        lifetime.close()
        return lifetime

    def _start_transfer_bookkeeping(self) -> None:
        """Start the single thread that owns deferred ABORT_ACKs and room GC.

        An ABORT_ACK promises the decode rank that this rank can no longer touch
        the room's pages, so it may only be sent once the room has drained.
        Waiting for that on the bootstrap thread would stall KV metadata, and
        waiting on a thread per abort does not survive an abort storm, so one
        periodic thread does both jobs.
        """
        self._abort_ack_queue: queue.Queue[_PendingAbortAck] = queue.Queue()
        self._abort_ack_pending: set = set()
        self._abort_ack_lock = threading.Lock()
        self._room_sweep_ttl = max(1.0, float(self.bootstrap_timeout))
        threading.Thread(
            target=self._transfer_bookkeeping_loop,
            name="MooncakeTransferBookkeeping",
            daemon=True,
        ).start()

    def request_abort_ack(
        self,
        lifetime: Optional[RoomTransferLifetime],
        room: int,
        decode_ip: str,
        decode_port: int,
        token: bytes,
    ) -> None:
        pending = _PendingAbortAck(
            room,
            decode_ip,
            decode_port,
            token,
            lifetime,
            deadline=time.monotonic() + ABORT_ACK_MAX_AGE_S,
        )
        with self._abort_ack_lock:
            if pending.key() in self._abort_ack_pending:
                return
            if len(self._abort_ack_pending) >= MAX_PENDING_ABORT_ACKS:
                logger.warning_once(
                    "More than %d ABORT_ACKs are owed; dropping new ones. Their "
                    "decode peers keep withholding the pages and escalate.",
                    MAX_PENDING_ABORT_ACKS,
                )
                return
            self._abort_ack_pending.add(pending.key())
        self._abort_ack_queue.put(pending)

    def _transfer_bookkeeping_loop(self) -> None:
        waiting: List[_PendingAbortAck] = []
        next_sweep = time.monotonic() + ROOM_SWEEP_INTERVAL_S
        while True:
            # This thread is the only one that can honour an ABORT_ACK, and a
            # decode peer that never gets one waits out its whole quiescence
            # deadline. It must therefore never exit, whatever goes wrong.
            try:
                waiting = self._drain_abort_acks(waiting)
                now = time.monotonic()
                if now >= next_sweep:
                    next_sweep = now + ROOM_SWEEP_INTERVAL_S
                    self._sweep_room_lifetimes()
            except Exception:
                # Belt and braces: _drain_abort_acks already isolates individual
                # entries, but this thread is the only one that can honour an
                # ABORT_ACK, so it must outlive any bug in here.
                logger.exception(
                    "Mooncake transfer bookkeeping iteration failed; continuing"
                )
                time.sleep(TRANSFER_BOOKKEEPING_INTERVAL_S)

    def _drain_abort_acks(
        self, waiting: List[_PendingAbortAck]
    ) -> List[_PendingAbortAck]:
        """Send every owed ACK whose room has drained; return those still owed."""
        try:
            # Poll quickly while ACKs are owed, otherwise just often enough to
            # keep the room sweep running.
            pending = self._abort_ack_queue.get(
                timeout=(
                    TRANSFER_BOOKKEEPING_INTERVAL_S
                    if waiting
                    else ROOM_SWEEP_INTERVAL_S
                )
            )
        except queue.Empty:
            pass
        else:
            waiting.append(pending)

        still_waiting = []
        now = time.monotonic()
        for pending in waiting:
            # Isolated per entry: a single unsatisfiable ACK must not starve the
            # ones behind it, which would strand their peers until they time out.
            try:
                if now >= pending.deadline:
                    logger.warning(
                        "Giving up on the ABORT_ACK owed to %s:%s for room %s",
                        pending.endpoint,
                        pending.dst_port,
                        pending.room,
                    )
                    self._retire_abort_ack(pending)
                elif (
                    pending.lifetime is not None and not pending.lifetime.is_quiesced()
                ):
                    still_waiting.append(pending)
                elif not self._send_abort_ack(pending):
                    # The peer's socket is backed up. Retry on the next tick
                    # rather than blocking every other peer's ACK behind it.
                    still_waiting.append(pending)
            except Exception:
                logger.exception(
                    "Dropping unsendable ABORT_ACK for room %s; its decode peer "
                    "keeps withholding the pages and escalates",
                    pending.room,
                )
                self._retire_abort_ack(pending)
        return still_waiting

    def _retire_abort_ack(self, pending: _PendingAbortAck) -> None:
        with self._abort_ack_lock:
            self._abort_ack_pending.discard(pending.key())

    def _send_abort_ack(self, pending: _PendingAbortAck) -> bool:
        """Try to send one ACK without blocking. False means "retry later"."""
        try:
            self._send_manager_message(
                pending.endpoint,
                pending.dst_port,
                [
                    b"ABORT_ACK",
                    str(pending.room).encode("ascii"),
                    pending.token,
                ],
                nonblocking=True,
            )
        except zmq.Again:
            return False
        except Exception as e:
            # A broken endpoint must not owe an ACK forever; the decode peer
            # falls back to its own quiescence deadline.
            logger.debug("Failed to send ABORT_ACK for room %s: %s", pending.room, e)
        else:
            logger.debug(
                "Sent ABORT_ACK for room %s to %s:%s",
                pending.room,
                pending.endpoint,
                pending.dst_port,
            )
        self._retire_abort_ack(pending)
        return True

    # ------------------------------------------------------------------
    # Live receiver registry (decode side)
    # ------------------------------------------------------------------

    def register_receiver(self, receiver: MooncakeKVReceiver) -> bool:
        """Claim *receiver*'s room. False if the room is still owned elsewhere."""
        with self._receivers_lock:
            if (
                self._receivers.setdefault(receiver.bootstrap_room, receiver)
                is receiver
            ):
                return True
        return False

    def unregister_receiver(self, receiver: MooncakeKVReceiver) -> None:
        with self._receivers_lock:
            if self._receivers.get(receiver.bootstrap_room) is receiver:
                del self._receivers[receiver.bootstrap_room]

    def _is_tearing_down(self, room: int) -> bool:
        """Whether *room* is releasing ownership, so late messages must be dropped."""
        with self._receivers_lock:
            receiver = self._receivers.get(room)
        return receiver is not None and receiver.is_failure_quiescing()

    def _handle_abort_ack(self, msg: List[bytes]) -> None:
        room = int(msg[1].decode("ascii"))
        # Peers that predate the ownership barrier reply without echoing the
        # token; such an ACK carries no drain guarantee (see record_abort_ack).
        token = msg[2] if len(msg) > 2 else None
        logger.debug("Received ABORT_ACK for room %s", room)
        with self._receivers_lock:
            receiver = self._receivers.get(room)
        if receiver is not None:
            receiver.record_abort_ack(token)

    # ------------------------------------------------------------------

    def report_unquiesced(self, room: int, reason: str) -> None:
        """Record that *room*'s KV pages cannot be proven idle.

        The pages are never released without proof: a silent overwrite is worse
        than a stuck request. Once too many requests are stuck the worker fails
        instead, because a restart releases every withheld page safely at once.
        """
        with self._unquiesced_lock:
            first_report = room not in self._unquiesced_rooms
            self._unquiesced_rooms.add(room)
            stuck = len(self._unquiesced_rooms)
        if first_report:
            TRANSFER_QUIESCE_TIMEOUTS.inc()
            logger.error(
                "Withholding KV pages for bootstrap_room=%s: %s (%d/%d requests "
                "stuck). A page that cannot be proven idle is never reused.",
                room,
                reason,
                stuck,
                MAX_UNQUIESCED_ROOMS,
            )
        if stuck >= MAX_UNQUIESCED_ROOMS:
            # Unrecoverable: the only way to reclaim these pages safely is to
            # stop using this address space entirely. The scheduler's top-level
            # handler turns this into engine teardown.
            raise KVTransferBarrierEscalation(
                f"{stuck} KV transfers cannot be proven quiesced "
                f"(MAX_UNQUIESCED_ROOMS={MAX_UNQUIESCED_ROOMS}); refusing to "
                "reuse their pages. Restarting this worker is the only safe "
                "way to reclaim them. Check for an unreachable or wedged peer."
            )

    def forget_unquiesced(self, room: int) -> None:
        with self._unquiesced_lock:
            self._unquiesced_rooms.discard(room)

    def _send_manager_message(
        self,
        remote: str,
        dst_port: int,
        parts: List[bytes],
        nonblocking: bool = False,
    ) -> None:
        """Send one multipart message to a peer's manager socket.

        *nonblocking* raises ``zmq.Again`` instead of waiting for a backed-up
        peer; use it for messages the caller retries.
        """
        na = NetworkAddress(remote, dst_port)
        self._send_multipart_locked(
            na.to_tcp(),
            parts,
            is_ipv6=na.is_ipv6,
            flags=zmq.NOBLOCK if nonblocking else 0,
        )

    def init_engine(self):
        self.engine = get_mooncake_transfer_engine()

    def register_buffer_to_engine(self):
        # Batch register KV data buffers
        if self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
            self.engine.batch_register(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            )

        # Batch register auxiliary data buffers
        if self.kv_args.aux_data_ptrs and self.kv_args.aux_data_lens:
            self.engine.batch_register(
                self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
            )

        for ptrs, lens in zip(
            self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
        ):
            if ptrs and lens:
                self.engine.batch_register(ptrs, lens)

    def deregister_buffer_to_engine(self):
        if self.kv_args.kv_data_ptrs:
            self.engine.batch_deregister(self.kv_args.kv_data_ptrs)

        if self.kv_args.aux_data_ptrs:
            self.engine.batch_deregister(self.kv_args.aux_data_ptrs)

        for ptrs in self.kv_args.state_data_ptrs or []:
            if ptrs:
                self.engine.batch_deregister(ptrs)

        if hasattr(self, "connection_pool"):
            with self.connection_lock:
                self.connection_pool.clear()

    # ------------------------------------------------------------------
    # Staging buffer methods (all delegate to staging_handler.py)
    # ------------------------------------------------------------------

    def register_staging_room_bootstrap(self, room, bootstrap_infos, receiver):
        self._staging_ctx.room_bootstrap[room] = bootstrap_infos
        self._staging_ctx.room_receivers[room] = receiver

    def set_kv_buffer_tensors(self, k_buffers: list, v_buffers: list, page_size: int):
        self.kv_buffer_tensors = {
            "k_buffers": k_buffers,
            "v_buffers": v_buffers,
            "page_size": page_size,
        }

    def _init_staging_buffers(self, count: int):
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_buffers,
        )

        self._staging_ctx.buffers = init_staging_buffers(
            lambda ptr, size: self.engine.batch_register([ptr], [size]),
            self.kv_args,
            count,
            get_schedule().chunked_prefill_size,
        )
        self.kv_buffer_tensors = None

    def _init_staging_allocator(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_allocator,
        )

        self._staging_ctx.allocator = init_staging_allocator(
            lambda ptr, size: self.engine.batch_register([ptr], [size]),
            self.kv_args,
        )
        self.kv_buffer_tensors = None

    def _handle_staging_req(self, msg):
        from sglang.srt.disaggregation.common.staging_handler import (
            handle_staging_req,
        )

        room = int(msg[1].decode("ascii"))
        session_id = msg[4].decode("ascii")
        handler = self._staging_handler
        assert (
            handler is not None
        ), "STAGING_REQ received before staging handler initialized"
        decode_req = handler._room_to_decode_req.get(room)
        if decode_req is None:
            logger.warning(
                "STAGING_REQ received for unregistered room=%s, skipping",
                room,
            )
            return
        prefill_tp = decode_req.kv_receiver.prefill_info.attn_tp_size
        handle_staging_req(
            msg,
            self._staging_ctx.allocator,
            self.kv_args,
            self.attn_tp_size,
            prefill_tp,
            getattr(self, "kv_buffer_tensors", None),
            self._staging_ctx.room_receivers,
            self._staging_ctx.room_bootstrap,
        )

        receiver = self._staging_ctx.room_receivers.get(room)
        if receiver is not None:
            handler.register_wm_subscriber(receiver, session_id)

    def _is_watermark_ready(
        self, session_id: str, alloc_round: int, alloc_end: int
    ) -> bool:
        from sglang.srt.disaggregation.common.staging_handler import (
            is_watermark_ready,
        )

        return is_watermark_ready(self._staging_ctx, session_id, alloc_round, alloc_end)

    def _try_create_staging_strategy(self, staging_buffer):
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return None
        from sglang.srt.disaggregation.common.staging_handler import (
            PrefillStagingStrategy,
        )

        return PrefillStagingStrategy(self, staging_buffer)

    def _send_chunk_ready(self, req, chunk_idx, kv_chunk, prefill_unique_rank):
        """Notify decode that a staging chunk RDMA is complete (every chunk;
        scatter is arrival-driven)."""
        self._send_manager_message(
            req.endpoint,
            req.dst_port,
            [
                b"CHUNK_READY",
                str(req.room).encode("ascii"),
                str(chunk_idx).encode("ascii"),
                str(kv_chunk.index_slice.start).encode("ascii"),
                str(len(kv_chunk.prefill_kv_indices)).encode("ascii"),
                req.mooncake_session_id.encode("ascii"),
                str(prefill_unique_rank).encode("ascii"),
            ],
        )

    def _do_staging_transfer(
        self,
        staging_strategy,
        kv_chunk,
        req,
        target_info,
        chunked_dst_kv_indice,
        executor,
        queue,
        prefill_unique_rank,
    ):
        """Execute staging transfer for one chunk. Returns (ret, deferred).

        Handles readiness check, transfer, and CHUNK_READY notification; a chunk
        that cannot fit returns -1 (the caller fails only this room) instead of
        falling back to the slice path, which would leak the decode-side
        allocation. deferred=True means caller should re-enqueue and break.
        """
        ready, chunk_idx, c_offset, _, _ = staging_strategy.check_ready(
            req,
            kv_chunk.index_slice.start,
            len(kv_chunk.prefill_kv_indices),
        )
        if not ready:
            from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator

            if c_offset == StagingAllocator.ALLOC_OVERSIZED:
                # Fail this room, not the worker thread: the same prefill still
                # serves other (same-TP, non-staging) decode instances.
                logger.warning_once(
                    "[Staging] a chunk exceeds the staging ring; failing affected "
                    "requests. Increase SGLANG_DISAGG_STAGING_POOL_SIZE_MB or "
                    "reduce chunked_prefill_size."
                )
                return (-1, False)
            # Not ready yet: wait (bounded) for a watermark advance, then
            # re-enqueue to retry. A plain block-until-ready would head-of-line
            # block other rooms on this single worker thread.
            with self._staging_ctx.watermark_cv:
                self._staging_ctx.watermark_cv.wait(STAGING_WATERMARK_WAIT_S)
            queue.put(kv_chunk)
            return (-1, True)

        ret = staging_strategy.transfer(
            req.mooncake_session_id,
            kv_chunk.prefill_kv_indices,
            target_info.staging_base_ptr + c_offset,
            target_info.staging_total_size - c_offset,
            target_info,
        )
        if ret == -1:
            # Doesn't fit the ring: fail this room (caller's ret != 0 path), do
            # not fall back to the slice path (leaks the decode-side allocation).
            logger.warning_once(
                "[Staging] a chunk does not fit the staging ring; failing affected "
                "requests. Increase SGLANG_DISAGG_STAGING_POOL_SIZE_MB or "
                "reduce chunked_prefill_size."
            )
            return (-1, False)
        if ret == 0:
            self._send_chunk_ready(req, chunk_idx, kv_chunk, prefill_unique_rank)
        return (ret, False)

    def _prefetch_staging_reqs(self, room: int):
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return

        room_infos = self.transfer_infos.get(room, {})
        needs_staging = any(
            not tinfo.is_dummy
            and self.decode_kv_args_table.get(tinfo.mooncake_session_id) is not None
            and self.decode_kv_args_table[tinfo.mooncake_session_id].dst_attn_tp_size
            != self.attn_tp_size
            for tinfo in room_infos.values()
        )
        if not needs_staging:
            return

        from sglang.srt.disaggregation.common.staging_handler import (
            prefetch_staging_reqs,
        )

        prefetch_staging_reqs(
            room,
            self.transfer_infos,
            self.kv_buffer_tensors,
            get_schedule().chunked_prefill_size,
            self._staging_ctx.prefetch_requested,
            self._staging_ctx.prefetch_sockets,
        )

    def send_kvcache_staged(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_staging_ptr: int,
        dst_staging_size: int,
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        staging_buffer=None,
    ) -> int:
        """Transfer KV cache via staging buffers (gather -> bulk RDMA -> scatter on decode)."""
        from sglang.srt.disaggregation.common.staging_buffer import (
            compute_head_slice_params,
            compute_staging_layout,
            resolve_total_kv_heads,
        )

        if self.kv_buffer_tensors is None or staging_buffer is None:
            return -1

        k_buffers = self.kv_buffer_tensors["k_buffers"]
        v_buffers = self.kv_buffer_tensors["v_buffers"]
        page_size = self.kv_buffer_tensors["page_size"]
        num_layers = len(k_buffers)
        head_dim = k_buffers[0].shape[-1]
        dtype_size = k_buffers[0].element_size()

        total_kv_heads = resolve_total_kv_heads(self.kv_args, self.attn_tp_size)

        local_tp_rank = self.kv_args.engine_rank % self.attn_tp_size
        src_head_start, num_heads_to_send, _, _ = compute_head_slice_params(
            self.attn_tp_size,
            dst_attn_tp_size,
            local_tp_rank,
            dst_tp_rank,
            total_kv_heads,
        )

        num_tokens = len(prefill_kv_indices) * page_size
        per_layer_bytes = num_tokens * num_heads_to_send * head_dim * dtype_size
        per_rank_bytes = per_layer_bytes * num_layers * 2

        num_writers, writer_rank_bytes, total_staging_needed = compute_staging_layout(
            self.attn_tp_size,
            dst_attn_tp_size,
            dst_tp_rank,
            total_kv_heads,
            num_tokens,
            head_dim * dtype_size,
            num_layers,
        )
        writer_idx = local_tp_rank % num_writers if num_writers > 1 else 0
        rank_offset = sum(writer_rank_bytes[:writer_idx])

        if not staging_buffer.fits(per_rank_bytes):
            logger.warning(
                f"Prefill staging too small for {per_rank_bytes} bytes, falling back"
            )
            return -1
        if dst_staging_size < total_staging_needed:
            logger.warning(
                f"Decode staging too small: need {total_staging_needed} bytes "
                f"({num_writers if self.attn_tp_size > dst_attn_tp_size else 1} writers "
                f"x {per_rank_bytes} bytes/rank), have {dst_staging_size}, falling back"
            )
            return -1

        from sglang.srt.disaggregation.common.staging_buffer import (
            gather_all_layers_to_staging,
        )

        gather_all_layers_to_staging(
            k_buffers,
            v_buffers,
            prefill_kv_indices,
            staging_buffer,
            src_head_start,
            num_heads_to_send,
            page_size,
            self.kv_args.gpu_id,
        )

        dst_write_ptr = dst_staging_ptr + rank_offset
        ret = self._transfer_data(
            mooncake_session_id,
            [(staging_buffer.get_ptr(), dst_write_ptr, per_rank_bytes)],
        )
        if ret != 0:
            raise RuntimeError(
                f"[Staging] Bulk RDMA transfer failed with ret={ret}. "
                f"src_ptr=0x{staging_buffer.get_ptr():x}, "
                f"dst_ptr=0x{dst_write_ptr:x}, size={per_rank_bytes}. "
                f"The decode staging buffer may not be properly registered."
            )
        return ret

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return 0

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )

    def _send_kvcache_generic(
        self,
        mooncake_session_id: str,
        src_data_ptrs: list[int],
        dst_data_ptrs: list[int],
        item_lens: list[int],
        prefill_data_indices: npt.NDArray[np.int32],
        dst_data_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        state_type: Optional[StateType] = None,
        force_flat: bool = False,
        src_layer_ids: Optional[List[int]] = None,
        dst_layer_ids: Optional[List[int]] = None,
        dst_device_data_indices: Optional[npt.NDArray[np.int32]] = None,
        dst_device_data_ptrs: Optional[set[int]] = None,
    ) -> int:
        """
        Generic KV cache transfer supporting both MHA and MLA architectures.
        This method is used by both send_kvcache (full pool) and maybe_send_extra.

        ``force_flat`` uses the MLA-style flat (single-buffer-per-layer) layout
        even on a non-MLA backend, for K-only state buffers (e.g. MiniMax sparse
        index) whose per-layer list must not be half-split into K/V.
        """
        # Host and device buffers may use different destination page spaces.
        # Build both transfer plans once, then select per destination buffer.
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_data_indices, dst_data_indices
        )
        device_prefill_kv_blocks = device_dst_kv_blocks = None
        if dst_device_data_indices is not None:
            device_prefill_kv_blocks, device_dst_kv_blocks = (
                group_concurrent_contiguous(
                    prefill_data_indices, dst_device_data_indices
                )
            )

        layers_params = None

        # Decode pp size should be equal to prefill pp size or 1
        if self.is_mla_backend or self.is_hybrid_mla_backend or force_flat:
            # Layer IDs map PP-local buffers to global decode entries.
            # Registrations without them retain the existing PP mapping.
            if src_layer_ids or dst_layer_ids:
                pairs = build_transfer_entry_pairs(
                    src_layer_ids,
                    dst_layer_ids,
                    len(src_data_ptrs),
                    len(dst_data_ptrs),
                    allow_positional_fallback=self.pp_size == 1,
                )
                layers_params = [
                    (src_data_ptrs[i], dst_data_ptrs[j], item_lens[i]) for i, j in pairs
                ]
            else:
                src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                    self.get_mla_kv_ptrs_with_pp(
                        src_data_ptrs, dst_data_ptrs, state_type
                    )
                )
                layers_params = [
                    (
                        src_kv_ptrs[layer_id],
                        dst_kv_ptrs[layer_id],
                        item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )
            # item_lens structure: [k_layer0, k_layer1, ..., k_layerN, v_layer0, v_layer1, ..., v_layerN]
            # Use correct item lengths for K and V separately
            if layers_current_pp_stage > len(dst_k_ptrs):
                logger.error(
                    "Prefill transfer kvcache error, layers_current_pp_stage is out of range: "
                    f"layers_current_pp_stage={layers_current_pp_stage}, len(dst_k_ptrs)={len(dst_k_ptrs)}"
                )
                return -1
            layers_params = [
                (
                    src_k_ptrs[layer_id],
                    dst_k_ptrs[layer_id],
                    item_lens[layer_id],  # K item length
                )
                for layer_id in range(layers_current_pp_stage)
            ] + [
                (
                    src_v_ptrs[layer_id],
                    dst_v_ptrs[layer_id],
                    item_lens[layers_current_pp_stage + layer_id],  # V item length
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        assert layers_params is not None

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            transfer_blocks = []
            if dst_device_data_ptrs and int(dst_ptr) in dst_device_data_ptrs:
                assert (
                    device_prefill_kv_blocks is not None
                    and device_dst_kv_blocks is not None
                )
                src_blocks, dst_blocks = (
                    device_prefill_kv_blocks,
                    device_dst_kv_blocks,
                )
            else:
                src_blocks, dst_blocks = prefill_kv_blocks, dst_kv_blocks
            for prefill_index, decode_index in zip(src_blocks, dst_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            return submit_transfer_calls(
                executor,
                [
                    (process_layer, (src_ptr, dst_ptr, item_len))
                    for (src_ptr, dst_ptr, item_len) in layers_params
                ],
            )
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: Optional[List[int]] = None,
        dst_device_kv_indices: Optional[npt.NDArray[np.int32]] = None,
    ):
        dst_device_kv_ptrs = None
        if dst_device_kv_indices is not None:
            compression_ratios = self.kv_args.mla_compression_ratios
            assert compression_ratios is not None
            if len(dst_kv_ptrs) == len(self.kv_args.kv_data_ptrs):
                start = self.kv_args.prefill_start_layer
                end = self.kv_args.prefill_end_layer
                assert end is not None
                compression_ratios = compression_ratios[start:end]
            c4_layer_num = sum(ratio == 4 for ratio in compression_ratios)
            dst_device_kv_ptrs = set(dst_kv_ptrs[c4_layer_num:])
        return self._send_kvcache_generic(
            mooncake_session_id=mooncake_session_id,
            src_data_ptrs=self.kv_args.kv_data_ptrs,
            dst_data_ptrs=dst_kv_ptrs,
            item_lens=self.kv_args.kv_item_lens,
            prefill_data_indices=prefill_kv_indices,
            dst_data_indices=dst_kv_indices,
            executor=executor,
            src_layer_ids=self.kv_args.kv_layer_ids,
            dst_layer_ids=dst_layer_ids,
            dst_device_data_indices=dst_device_kv_indices,
            dst_device_data_ptrs=dst_device_kv_ptrs,
        )

    def send_kvcache_dcp(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        *,
        dcp_token_item_lens: List[int],
        dst_dcp_size: int,
        dst_dcp_rank: int,
        src_page_offset: int,
        decode_prefix_len: int,
        num_kv_tokens: int,
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_layer_ids: List[int],
    ) -> int:
        if num_kv_tokens is None:
            raise ValueError("PD DCP transfer requires num_kv_tokens")
        physical_page_size = self.kv_args.page_size
        plan = build_dcp_token_transfer_plan(
            prefill_kv_indices,
            dst_kv_indices,
            physical_page_size=physical_page_size,
            dcp_size=dst_dcp_size,
            dcp_rank=dst_dcp_rank,
            src_page_offset=src_page_offset,
            decode_prefix_len=decode_prefix_len,
            num_kv_tokens=num_kv_tokens,
        )
        if plan.src_token_indices.size == 0:
            return 0

        src_layer_ids = self.kv_args.kv_layer_ids
        if src_layer_ids or dst_layer_ids:
            dst_indices = resolve_dcp_dst_entry_indices(
                src_layer_ids,
                dst_layer_ids,
                len(self.kv_args.kv_data_ptrs),
                len(dst_kv_ptrs),
            )
            src_kv_ptrs = self.kv_args.kv_data_ptrs
            dst_kv_ptrs = [dst_kv_ptrs[j] for j in dst_indices]
        else:
            src_kv_ptrs, dst_kv_ptrs, _ = self.get_mla_kv_ptrs_with_pp(
                self.kv_args.kv_data_ptrs,
                dst_kv_ptrs,
            )
        layers_current_pp_stage = len(src_kv_ptrs)
        src_groups, dst_groups = group_concurrent_contiguous(
            plan.src_token_indices,
            plan.dst_token_indices,
        )

        layers_params = [
            (
                src_kv_ptrs[layer_id],
                dst_kv_ptrs[layer_id],
                dcp_token_item_lens[layer_id],
            )
            for layer_id in range(layers_current_pp_stage)
        ]

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, token_item_len: int
        ) -> List[Tuple[int, int, int]]:
            return [
                (
                    src_ptr + int(src_group[0]) * token_item_len,
                    dst_ptr + int(dst_group[0]) * token_item_len,
                    len(src_group) * token_item_len,
                )
                for src_group, dst_group in zip(src_groups, dst_groups)
            ]

        def process_layer(src_ptr: int, dst_ptr: int, token_item_len: int) -> int:
            return self._transfer_data(
                mooncake_session_id,
                set_transfer_blocks(src_ptr, dst_ptr, token_item_len),
            )

        if self.enable_custom_mem_pool:
            return submit_transfer_calls(
                executor,
                [
                    (process_layer, (src_ptr, dst_ptr, token_item_len))
                    for (src_ptr, dst_ptr, token_item_len) in layers_params
                ],
            )

        transfer_blocks = []
        for src_ptr, dst_ptr, token_item_len in layers_params:
            transfer_blocks.extend(
                set_transfer_blocks(src_ptr, dst_ptr, token_item_len)
            )
        return self._transfer_data(mooncake_session_id, transfer_blocks)

    def send_kvcache_slice(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """
        Sends KV cache slices from this Prefill rank to a target Decode rank,
        supporting generic M-to-N TP size configurations.

        NOTE: This implementation calls the transfer engine for each token slot within
        each page to ensure correctness for any page_size and head-slicing configuration.
        This may introduce performance overhead (increased TTFT) for long sequences.
        """
        # Extract configuration
        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        src_kv_item_len = self.kv_args.kv_item_lens[0]
        dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size
        page_size = self.kv_args.page_size

        # Use total KV head count (not per-rank) for correct head distribution.
        # Per-rank kv_head_num is max(1, total//tp) which loses info when total < tp.
        total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
        if total_kv_heads <= 0:
            total_kv_heads = self.kv_args.kv_head_num * self.attn_tp_size

        src_heads_per_rank = max(1, total_kv_heads // self.attn_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // dst_attn_tp_size)
        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        # GQA replication: how many prefill ranks share the same KV head
        src_replication = max(1, self.attn_tp_size // total_kv_heads)

        # Determine slicing parameters based on TP configuration
        if self.attn_tp_size > dst_attn_tp_size:
            # Send KVCache from multiple prefill instances to 1 decode instance
            src_head_start_offset = 0
            num_heads_to_send = src_heads_per_rank
            unique_head_idx = local_tp_rank_in_group // src_replication
            dst_head_start_offset = (
                unique_head_idx * src_heads_per_rank
            ) % dst_heads_per_rank
        else:
            # Send KVCache from 1 prefill instance to multiple decode instances
            # GQA replication (total_kv_heads < dst_attn_tp_size): consecutive decode
            # ranks share one KV head (QKVParallelLinear: tp_rank // num_kv_head_replicas),
            # so map by integer division NOT modulo or ranks 1..r-1 fetch the wrong head.
            dst_replication = max(1, dst_attn_tp_size // total_kv_heads)
            unique_dst_head_idx = dst_tp_rank_in_group // dst_replication
            src_head_start_offset = (
                unique_dst_head_idx * dst_heads_per_rank
            ) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start_offset = 0

        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
            self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
        )

        # Calculate precise byte offset and length for the sub-slice within the token
        src_head_slice_offset = src_head_start_offset * bytes_per_head_slice_to_send
        dst_head_slice_offset = dst_head_start_offset * bytes_per_head_slice_to_send
        heads_bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice_to_send

        # Sanity check: The data sub-slice to be sent should fit into the dst buffer.
        # This means heads_bytes_per_token_to_send <= (dst_kv_item_len // page_size)
        if heads_bytes_per_token_to_send > (dst_kv_item_len // page_size):
            logger.error(
                f"[{mooncake_session_id}] slice size ({heads_bytes_per_token_to_send}) exceeds "
                f"target token slot size ({dst_kv_item_len // page_size})"
            )
            return -1

        prefill_page_indices = prefill_kv_indices.reshape(-1, 1).astype(np.int64)
        decode_page_indices = dst_kv_indices.reshape(-1, 1).astype(np.int64)
        tokens_per_page = np.arange(page_size, dtype=np.int64).reshape(1, -1)
        bytes_per_token_on_prefill = src_kv_item_len // page_size
        bytes_per_token_on_decode = dst_kv_item_len // page_size
        src_token_slot_offsets = (
            tokens_per_page * bytes_per_token_on_prefill + src_head_slice_offset
        )
        dst_token_slot_offsets = (
            tokens_per_page * bytes_per_token_on_decode + dst_head_slice_offset
        )

        def process_layer_tp_aware(src_layer_ptr, dst_layer_ptr):
            src_page_base_addrs = src_layer_ptr + prefill_page_indices * src_kv_item_len
            dst_page_base_addrs = dst_layer_ptr + decode_page_indices * dst_kv_item_len
            src_slice_addrs = src_page_base_addrs + src_token_slot_offsets
            dst_slice_addrs = dst_page_base_addrs + dst_token_slot_offsets

            src_addr_list = src_slice_addrs.reshape(-1).tolist()
            if not src_addr_list:
                # Nothing to transfer for this layer.
                return 0
            dst_addr_list = dst_slice_addrs.reshape(-1).tolist()
            total_slices = len(src_addr_list)
            length_list = [heads_bytes_per_token_to_send] * total_slices
            return self.engine.batch_transfer_sync(
                mooncake_session_id, src_addr_list, dst_addr_list, length_list
            )

        calls = [
            (process_layer_tp_aware, (src_k_ptrs[i], dst_k_ptrs[i]))
            for i in range(layers_current_pp_stage)
        ]
        calls += [
            (process_layer_tp_aware, (src_v_ptrs[i], dst_v_ptrs[i]))
            for i in range(layers_current_pp_stage)
        ]
        return submit_transfer_calls(executor, calls)

    def send_aux(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        # TODO(shangming): Fix me when nvlink_transport of Mooncake is bug-free
        if (
            self.enable_custom_mem_pool and self.custom_mem_pool_type == "NVLINK"
        ) or envs.SGLANG_MOONCAKE_SEND_AUX_TCP.get():
            return self.send_aux_tcp(req, prefill_aux_index, dst_aux_ptrs)

        transfer_blocks = []
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i, dst_aux_ptr in enumerate(dst_aux_ptrs):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * req.dst_aux_index
            transfer_blocks.append((src_addr, dst_addr, length))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def send_aux_tcp(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i in range(len(prefill_aux_ptrs)):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            data = AuxDataCodec.serialize_data_from_buffer(src_addr, length)

            self.send_aux_data_to_endpoint(
                remote=req.endpoint,
                dst_port=req.dst_port,
                room=req.room,
                buffer_index=i,
                aux_index=req.dst_aux_index,
                data=data,
            )

        return 0

    def send_aux_data_to_endpoint(
        self,
        remote: str,
        dst_port: int,
        room: int,
        buffer_index: int,
        aux_index: int,
        data: bytes,
    ):
        self._send_manager_message(
            remote,
            dst_port,
            [
                MooncakeKVManager.AUX_DATA_HEADER,
                str(room).encode("ascii"),
                str(buffer_index).encode("ascii"),
                str(aux_index).encode("ascii"),
                struct.pack(">I", len(data)),
                data,
            ],
        )

    def _handle_aux_data(self, msg: List[bytes]):
        """Handle AUX_DATA messages received by the decode thread."""
        room = int(msg[1].decode("ascii"))
        buffer_index = int(msg[2].decode("ascii"))
        aux_index = int(msg[3].decode("ascii"))
        data_length = struct.unpack(">I", msg[4])[0]
        data = msg[5]

        if len(data) != data_length:
            logger.error(f"AUX_DATA length mismatch for bootstrap_room {room}")
            return

        AuxDataCodec.deserialize_data_to_buffer(
            self.kv_args, buffer_index, aux_index, data
        )

        logger.debug(
            f"Received AUX_DATA for bootstrap_room {room} with length:{len(data)}"
        )

    def _get_dsa_cache_transfer_skip_flags(
        self, info: Optional[KVArgsRegisterInfo]
    ) -> Tuple[bool, bool]:
        skip_kv = False
        skip_state = False

        # Must be checked before the non-hybrid early return below, or every CP
        # rank re-sends the same state and we transfer it cp_size times over.
        # Prefill CP all-gathers before writing the pool, so every CP rank holds
        # the full state regardless of whether the pool is hybrid. We assume no
        # structure about the state rows, so we don't split them across CP ranks
        # -- just let rank 0 send the whole thing (unless layer split already
        # shards it per rank).
        if (
            self.attn_cp_size > 1
            and self.attn_cp_rank != 0
            and not get_parallel().enable_dsa_cache_layer_split
        ):
            skip_state = True

        if not self.is_hybrid_mla_backend:
            return skip_kv, skip_state

        if info is not None and self.attn_tp_size > info.dst_attn_tp_size:
            sub_rank = (self.kv_args.engine_rank % self.attn_tp_size) % (
                self.attn_tp_size // info.dst_attn_tp_size
            )
            if sub_rank != 0:
                skip_kv = True
                # Hybrid-MLA KV is replicated across these source ranks, but
                # TP-sharded state needs every rank for the aggregation path.

        return skip_kv, skip_state

    def _is_generic_kvcache_state_type(self, st: StateType) -> bool:
        """State types sent via the page-indexed ``_send_kvcache_generic`` path
        (not the mamba-state path); subclasses extend for hardware components."""
        return st in (
            StateType.SWA,
            StateType.DSA,
            StateType.SWA_RING,
            StateType.C128_STATE,
        )

    def _requires_exact_state_index_match(self, st: StateType) -> bool:
        """State types whose page lists are positional and must not be truncated."""
        return st in (StateType.SWA_RING, StateType.C128_STATE)

    def maybe_send_extra(
        self,
        req: TransferInfo,
        prefill_state_indices: List,
        executor: concurrent.futures.ThreadPoolExecutor,
        target_rank_registration_info: Optional[KVArgsRegisterInfo] = None,
    ):
        rc = 0
        state_types = getattr(self.kv_args, "state_types", [])
        for i, st in enumerate(state_types):
            indices = (
                prefill_state_indices[i] if i < len(prefill_state_indices) else None
            )
            if indices is None:
                continue
            src_data_ptrs = self.kv_args.state_data_ptrs[i]
            src_item_lens = self.kv_args.state_item_lens[i]
            src_dim_per_tensor = (
                self.kv_args.state_dim_per_tensor[i]
                if i < len(self.kv_args.state_dim_per_tensor)
                else []
            )
            src_conv_shard_groups = getattr(self.kv_args, "state_conv_shard_groups", [])
            src_conv_shard_groups = (
                src_conv_shard_groups[i] if i < len(src_conv_shard_groups) else []
            )
            src_slice_outer_counts = getattr(
                self.kv_args, "state_slice_outer_counts", []
            )
            src_slice_outer_counts = (
                src_slice_outer_counts[i] if i < len(src_slice_outer_counts) else []
            )
            src_state_layer_ids = self.kv_args.state_layer_ids
            src_state_layer_ids = (
                src_state_layer_ids[i] if i < len(src_state_layer_ids) else []
            )
            if target_rank_registration_info is not None:
                dst_data_ptrs = (
                    target_rank_registration_info.dst_state_data_ptrs[i]
                    if i < len(target_rank_registration_info.dst_state_data_ptrs)
                    else []
                )
                dst_item_lens = (
                    target_rank_registration_info.dst_state_item_lens[i]
                    if i < len(target_rank_registration_info.dst_state_item_lens)
                    else []
                )
                dst_dim_per_tensor = (
                    target_rank_registration_info.dst_state_dim_per_tensor[i]
                    if i < len(target_rank_registration_info.dst_state_dim_per_tensor)
                    else []
                )
                dst_state_layer_ids = (
                    target_rank_registration_info.dst_state_layer_ids[i]
                    if i < len(target_rank_registration_info.dst_state_layer_ids)
                    else []
                )
            else:
                dst_data_ptrs, dst_item_lens, dst_dim_per_tensor = [], [], []
                dst_state_layer_ids = []
            dst_indices = (
                req.dst_state_indices[i] if i < len(req.dst_state_indices) else []
            )

            if st == StateType.MAMBA:
                if (
                    target_rank_registration_info is not None
                    and self.attn_tp_size
                    != target_rank_registration_info.dst_attn_tp_size
                ):
                    rc = (
                        self._send_mamba_state_slice(
                            req,
                            indices,
                            src_data_ptrs,
                            src_item_lens,
                            src_dim_per_tensor,
                            dst_data_ptrs,
                            dst_indices,
                            dst_item_lens,
                            dst_dim_per_tensor,
                            target_rank_registration_info.dst_tp_rank,
                            target_rank_registration_info.dst_attn_tp_size,
                            src_conv_shard_groups,
                            src_slice_outer_counts,
                            src_state_layer_ids,
                            dst_state_layer_ids,
                        )
                        or rc
                    )
                else:
                    rc = (
                        self._send_mamba_state(
                            req,
                            indices,
                            src_data_ptrs,
                            src_item_lens,
                            dst_data_ptrs,
                            dst_indices,
                            src_state_layer_ids,
                            dst_state_layer_ids,
                        )
                        or rc
                    )
            elif self._is_generic_kvcache_state_type(st):
                if (
                    target_rank_registration_info is not None
                    and not self.is_mla_backend
                    and self.attn_tp_size
                    != target_rank_registration_info.dst_attn_tp_size
                ):
                    raise RuntimeError(
                        f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {st.upper()} hybrid models yet."
                    )
                src_indices = list(indices)
                dst_indices_local = list(dst_indices)
                if (
                    st == StateType.C128_STATE
                    and len(src_indices) == 0
                    and len(dst_indices_local) == 0
                ):
                    continue
                if len(src_indices) != len(dst_indices_local):
                    # These components are position- or request-indexed:
                    # truncating silently misaligns rows and corrupts KV.
                    # Paged SWA/DSA tolerate a 1-page drift -> keep the
                    # lenient truncation below.
                    if self._requires_exact_state_index_match(st):
                        raise RuntimeError(
                            f"{st.upper()} state index length mismatch: "
                            f"prefill={len(src_indices)}, dst={len(dst_indices_local)}"
                        )
                    logger.warning(
                        f"len(prefill_state_indices) = {len(src_indices)}, len(dst_state_indices) = {len(dst_indices_local)}"
                    )
                    if len(src_indices) > len(dst_indices_local):
                        src_indices = src_indices[: len(dst_indices_local)]
                    else:
                        dst_indices_local = dst_indices_local[: len(src_indices)]
                rc = (
                    self._send_kvcache_generic(
                        mooncake_session_id=req.mooncake_session_id,
                        src_data_ptrs=src_data_ptrs,
                        dst_data_ptrs=dst_data_ptrs,
                        item_lens=src_item_lens,
                        prefill_data_indices=np.array(src_indices, dtype=np.int32),
                        dst_data_indices=np.array(dst_indices_local, dtype=np.int32),
                        executor=executor,
                        state_type=st,
                    )
                    or rc
                )
            elif st == StateType.MINIMAX_INDEX_K:
                # Equal-TP / PP=1 only. Sub-pools are compacted sparse-layer
                # lists, so PP>1 mis-slices and heterogeneous TP is unsupported.
                if self.pp_size is not None and self.pp_size > 1:
                    raise RuntimeError(
                        "PD disagg: PP>1 not supported for MiniMax sparse index yet."
                    )
                if (
                    target_rank_registration_info is not None
                    and self.attn_tp_size
                    != target_rank_registration_info.dst_attn_tp_size
                ):
                    raise RuntimeError(
                        "PD disagg: heterogeneous TP not supported for MiniMax "
                        "sparse index yet."
                    )
                src_indices = list(indices)
                dst_indices_local = list(dst_indices)
                if len(src_indices) > len(dst_indices_local):
                    src_indices = src_indices[: len(dst_indices_local)]
                elif len(src_indices) < len(dst_indices_local):
                    dst_indices_local = dst_indices_local[: len(src_indices)]
                rc = (
                    self._send_kvcache_generic(
                        mooncake_session_id=req.mooncake_session_id,
                        src_data_ptrs=src_data_ptrs,
                        dst_data_ptrs=dst_data_ptrs,
                        item_lens=src_item_lens,
                        prefill_data_indices=np.array(src_indices, dtype=np.int32),
                        dst_data_indices=np.array(dst_indices_local, dtype=np.int32),
                        executor=executor,
                        force_flat=True,
                    )
                    or rc
                )
        return rc

    def _send_mamba_state(
        self,
        req: TransferInfo,
        prefill_mamba_index: list,
        src_state_data_ptrs: list[int],
        src_state_item_lens: list[int],
        dst_state_data_ptrs: list[int],
        dst_mamba_index: list,
        src_layer_ids: Optional[List[int]] = None,
        dst_layer_ids: Optional[List[int]] = None,
    ):
        assert len(prefill_mamba_index) == 1, "Mamba should have single state index"

        transfer_blocks = []
        pairs = build_transfer_entry_pairs(
            src_layer_ids or [],
            dst_layer_ids or [],
            len(src_state_data_ptrs),
            len(dst_state_data_ptrs),
            allow_positional_fallback=self.pp_size == 1,
        )
        for i, j in pairs:
            dst_state_ptr = dst_state_data_ptrs[j]
            length = src_state_item_lens[i]
            src_addr = src_state_data_ptrs[i] + length * int(prefill_mamba_index[0])
            dst_addr = dst_state_ptr + length * int(dst_mamba_index[0])
            transfer_blocks.append((src_addr, dst_addr, length))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def _send_mamba_state_slice(
        self,
        req: TransferInfo,
        prefill_mamba_index: list,
        src_state_data_ptrs: list[int],
        src_state_item_lens: list[int],
        src_state_dim_per_tensor: list[int],
        dst_state_data_ptrs: list[int],
        dst_mamba_index: list,
        dst_state_item_lens: list[int],
        dst_state_dim_per_tensor: list[int],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        src_state_conv_shard_groups: list = None,
        src_state_slice_outer_counts: list[int] = None,
        src_layer_ids: Optional[List[int]] = None,
        dst_layer_ids: Optional[List[int]] = None,
    ):
        """Transfer Mamba states with TP slice support.

        Mamba state layout:
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

        The 3rd dimension is sliced by TP. When prefill and decode have different
        attn_tp_size, we slice the state accordingly. GDN conv_state is the
        concatenation [query | key | value] with each sub-block head-sharded
        independently, so on the scatter path it is sliced per sub-block via
        ``src_state_conv_shard_groups`` (see compute_mamba_state_slice_blocks).
        """
        logger.warning_once(
            "Using Mamba state slice transfer for different TP sizes between prefill and decode. "
            f"Prefill attn_tp_size={self.attn_tp_size}, Decode attn_tp_size={dst_attn_tp_size}. "
            "Performance may be affected."
        )
        assert len(prefill_mamba_index) == 1, "Mamba should have single state index"

        # If no dimension info available, fall back to regular transfer
        if not src_state_dim_per_tensor or not dst_state_dim_per_tensor:
            return self._send_mamba_state(
                req,
                prefill_mamba_index,
                src_state_data_ptrs,
                src_state_item_lens,
                dst_state_data_ptrs,
                dst_mamba_index,
                src_layer_ids,
                dst_layer_ids,
            )

        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size

        transfer_blocks = []
        pairs = build_transfer_entry_pairs(
            src_layer_ids or [],
            dst_layer_ids or [],
            len(src_state_data_ptrs),
            len(dst_state_data_ptrs),
            allow_positional_fallback=self.pp_size == 1,
        )
        for i, j in pairs:
            dst_state_ptr = dst_state_data_ptrs[j]
            src_item_len = src_state_item_lens[i]
            dst_item_len = dst_state_item_lens[j]
            src_dim = src_state_dim_per_tensor[i]
            dst_dim = dst_state_dim_per_tensor[j]

            conv_shard_groups = (
                src_state_conv_shard_groups[i]
                if src_state_conv_shard_groups and i < len(src_state_conv_shard_groups)
                else None
            )
            outer_count = (
                src_state_slice_outer_counts[i]
                if src_state_slice_outer_counts
                and i < len(src_state_slice_outer_counts)
                else 1
            )
            for (
                src_offset,
                dst_offset,
                bytes_to_send,
            ) in compute_mamba_state_slice_byte_blocks(
                src_item_len=src_item_len,
                dst_item_len=dst_item_len,
                src_dim=src_dim,
                dst_dim=dst_dim,
                outer_count=outer_count,
                src_attn_tp_size=self.attn_tp_size,
                dst_attn_tp_size=dst_attn_tp_size,
                dst_tp_rank_in_group=dst_tp_rank_in_group,
                local_tp_rank_in_group=local_tp_rank_in_group,
                conv_shard_groups=conv_shard_groups,
            ):
                src_addr = (
                    src_state_data_ptrs[i]
                    + src_item_len * int(prefill_mamba_index[0])
                    + src_offset
                )
                dst_addr = (
                    dst_state_ptr + dst_item_len * int(dst_mamba_index[0]) + dst_offset
                )
                transfer_blocks.append((src_addr, dst_addr, bytes_to_send))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def sync_status_to_decode_endpoint(
        self, remote: str, dst_port: int, room: int, status: int, prefill_rank: int
    ):
        self._send_manager_message(
            remote,
            dst_port,
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
                str(prefill_rank).encode("ascii"),
            ],
        )

    def transfer_worker(
        self,
        queue: FastQueue,
        executor: concurrent.futures.ThreadPoolExecutor,
        staging_buffer=None,
        worker_index=0,
    ):
        staging_strategy = None
        if self.enable_trace:
            trace_set_thread_info(
                f"mooncake transfer worker {worker_index}",
                tp_rank=self.attn_tp_rank,
                dp_rank=self.attn_dp_rank,
            )

        while True:
            lifetime = None
            try:
                kv_chunk: TransferKVChunk = queue.get()
                if self.enable_trace:
                    kv_chunk.trace_ctx.rebuild_thread_context()
                    kv_chunk.trace_ctx.trace_slice_start(
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                    )

                # Hold the room's transfer lease for the whole chunk. Until it is
                # returned the scheduler cannot observe the request as terminal,
                # so these KV pages stay owned by this transfer.
                lifetime = self.try_lease_chunk(kv_chunk)
                if (
                    lifetime is None
                    or kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Failed
                ):
                    logger.debug(
                        f"Skipping chunk for room {kv_chunk.room} because it has already failed or been aborted"
                    )
                    if self.enable_trace:
                        kv_chunk.trace_ctx.trace_slice_end(
                            MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                            MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                            thread_finish_flag=True,
                        )
                    self._staging_outstanding.pop(kv_chunk.room, None)
                    continue

                # Count each chunk once; the flag survives re-enqueue on defer.
                if not kv_chunk.staging_counted:
                    self._staging_outstanding[kv_chunk.room] += 1
                    kv_chunk.staging_counted = True

                if (
                    self.enable_staging
                    and staging_strategy is None
                    and staging_buffer is not None
                ):
                    staging_strategy = self._try_create_staging_strategy(staging_buffer)
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                # Unique id per prefill sender so decode's response set size matches expected_response_num.
                prefill_unique_rank = (
                    self.attn_tp_rank * (self.pp_size * self.attn_cp_size)
                    + self.pp_rank * self.attn_cp_size
                    + self.attn_cp_rank
                )
                # When staging transfer is not yet ready (watermark/allocation pending),
                # the chunk is re-enqueued and we break out of the req loop to retry later.
                staging_deferred = False
                for req in reqs_to_be_processed:
                    start_ts = time.perf_counter()
                    if not req.is_dummy:
                        # Early exit if the request has failed
                        with self.session_lock:
                            if req.mooncake_session_id in self.failed_sessions:
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Decode instance could be dead, remote mooncake session {req.mooncake_session_id} is not alive",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                    prefill_unique_rank,
                                )
                                break

                        target_rank_registration_info: KVArgsRegisterInfo = (
                            self.decode_kv_args_table[req.mooncake_session_id]
                        )
                        is_dcp_transfer = (
                            target_rank_registration_info.requires_dcp_relayout
                        )
                        chunked_dst_device_kv_indice = None
                        if is_dcp_transfer:
                            if req.dst_device_kv_indices is not None:
                                raise RuntimeError(
                                    "HiSparse destination device indices are not "
                                    "supported by PD DCP relayout"
                                )
                            chunked_dst_kv_indice = req.dst_kv_indices
                        else:
                            chunked_dst_kv_indice = req.dst_kv_indices[
                                kv_chunk.index_slice
                            ]
                            if req.dst_device_kv_indices is not None:
                                chunked_dst_device_kv_indice = (
                                    req.dst_device_kv_indices[kv_chunk.index_slice]
                                )

                            # NOTE: This is temporarily a workaround to deal with the case where the prefill_kv_indices
                            # is mismatched with the dst_kv_indices when page size > 1, this should never happen.
                            if len(chunked_dst_kv_indice) < len(
                                kv_chunk.prefill_kv_indices
                            ):
                                logger.warning(
                                    f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                                )
                                kv_chunk.prefill_kv_indices = (
                                    kv_chunk.prefill_kv_indices[
                                        : len(chunked_dst_kv_indice)
                                    ]
                                )
                            if chunked_dst_device_kv_indice is not None:
                                chunked_dst_device_kv_indice = (
                                    chunked_dst_device_kv_indice[
                                        : len(kv_chunk.prefill_kv_indices)
                                    ]
                                )

                        skip_kv, skip_state = self._get_dsa_cache_transfer_skip_flags(
                            target_rank_registration_info
                        )
                        if (
                            len(kv_chunk.prefill_kv_indices) == 0
                            or not self.kv_args.kv_data_ptrs
                            or skip_kv
                        ):
                            ret = 0
                        elif is_dcp_transfer:
                            dcp_token_item_lens = (
                                target_rank_registration_info.dcp_token_item_lens
                            )
                            assert dcp_token_item_lens is not None
                            ret = self.send_kvcache_dcp(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                dcp_token_item_lens=dcp_token_item_lens,
                                dst_dcp_size=target_rank_registration_info.dst_dcp_size,
                                dst_dcp_rank=target_rank_registration_info.dst_dcp_rank,
                                src_page_offset=kv_chunk.index_slice.start or 0,
                                decode_prefix_len=req.decode_prefix_len or 0,
                                num_kv_tokens=kv_chunk.num_kv_tokens,
                                executor=executor,
                                dst_layer_ids=(
                                    target_rank_registration_info.dst_kv_layer_ids
                                ),
                            )
                        elif (
                            self.is_mla_backend
                            or self.is_hybrid_mla_backend
                            or self.attn_tp_size
                            == target_rank_registration_info.dst_attn_tp_size
                        ):
                            ret = self.send_kvcache(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                executor,
                                dst_layer_ids=target_rank_registration_info.dst_kv_layer_ids,
                                dst_device_kv_indices=chunked_dst_device_kv_indice,
                            )
                        elif (
                            self.enable_staging
                            and staging_strategy is not None
                            and (
                                target_rank_registration_info.staging_base_ptr != 0
                                or target_rank_registration_info.staging_total_size != 0
                            )
                        ):
                            ret, deferred = self._do_staging_transfer(
                                staging_strategy,
                                kv_chunk,
                                req,
                                target_rank_registration_info,
                                chunked_dst_kv_indice,
                                executor,
                                queue,
                                prefill_unique_rank,
                            )
                            if deferred:
                                staging_deferred = True
                                # Chunk re-enqueued; stop processing remaining reqs for this chunk
                                break
                        else:
                            ret = self.send_kvcache_slice(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                target_rank_registration_info.dst_tp_rank,
                                target_rank_registration_info.dst_attn_tp_size,
                                target_rank_registration_info.dst_kv_item_len,
                                executor,
                            )
                        if ret != 0:
                            # A failed engine call can leave RDMA work running;
                            # keep the room owned while the engine drains.
                            lifetime.quarantine(ENGINE_FAILURE_QUARANTINE_S)
                            with self.session_lock:
                                self.session_failures[req.mooncake_session_id] += 1
                                # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                                if self.session_failures[req.mooncake_session_id] >= 1:
                                    self.failed_sessions.add(req.mooncake_session_id)
                                    logger.error(
                                        f"Session {req.mooncake_session_id} failed."
                                    )
                            self.record_failure(
                                kv_chunk.room,
                                f"Failed to send kv chunk of {kv_chunk.room} to "
                                f"{NetworkAddress(req.endpoint, req.dst_port).to_host_port_str()}",
                            )
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                                prefill_unique_rank,
                            )
                            break

                        if kv_chunk.is_last_chunk:
                            if kv_chunk.state_indices and not skip_state:
                                state_rc = self.maybe_send_extra(
                                    req,
                                    kv_chunk.state_indices,
                                    executor,
                                    target_rank_registration_info,
                                )
                                if state_rc != 0:
                                    lifetime.quarantine(ENGINE_FAILURE_QUARANTINE_S)
                                    with self.session_lock:
                                        self.session_failures[
                                            req.mooncake_session_id
                                        ] += 1
                                        self.failed_sessions.add(
                                            req.mooncake_session_id
                                        )
                                    self.record_failure(
                                        kv_chunk.room,
                                        f"Failed to send state components of {kv_chunk.room} to "
                                        f"{NetworkAddress(req.endpoint, req.dst_port).to_host_port_str()}",
                                    )
                                    self.update_status(kv_chunk.room, KVPoll.Failed)
                                    self.sync_status_to_decode_endpoint(
                                        req.endpoint,
                                        req.dst_port,
                                        req.room,
                                        KVPoll.Failed,
                                        prefill_unique_rank,
                                    )
                                    break

                            # Only the last chunk we need to send the aux data
                            ret = self.send_aux(
                                req,
                                kv_chunk.prefill_aux_index,
                                target_rank_registration_info.dst_aux_ptrs,
                            )
                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append(
                                (req.endpoint, req.dst_port, req.room)
                            )

                            # Only sync status when all the dst ranks have received the kvcache
                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint,
                                        dst_port,
                                        room,
                                        status,
                                        prefill_unique_rank,
                                    )
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        # Dummy request does not need to sync status to decode endpoint
                        if kv_chunk.is_last_chunk and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                    if self.enable_trace:
                        mooncake_trace_slice(
                            kv_chunk.trace_ctx,
                            MooncakeRequestStage.MOONCAKE_WORKER_SEND_SESSION,
                            start_ts,
                        )

                if self.enable_trace:
                    kv_chunk.trace_ctx.trace_slice_end(
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                        thread_finish_flag=True,
                    )

                if staging_deferred:
                    continue

                self._staging_outstanding[kv_chunk.room] -= 1
                # Tear down only when no chunk is still outstanding and the room
                # has concluded: already cleared, Success, or a Failed *last*
                # chunk. A non-last Failed chunk keeps the room (more chunks may
                # follow), not on the last chunk alone since an earlier deferred
                # chunk may still need to transfer.
                if self._staging_outstanding.get(kv_chunk.room, 0) <= 0 and (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                    or (
                        kv_chunk.is_last_chunk
                        and self.check_status(kv_chunk.room) == KVPoll.Failed
                    )
                ):
                    self._staging_outstanding.pop(kv_chunk.room, None)
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)
                    self.req_to_decode_prefix_len.pop(kv_chunk.room, None)
                    if self.enable_staging:
                        # Purge prefetch bookkeeping for the finished room.
                        # Snapshot first: the scheduler thread adds concurrently.
                        for key in list(self._staging_ctx.prefetch_requested):
                            if key[0] == kv_chunk.room:
                                self._staging_ctx.prefetch_requested.discard(key)
                        self._staging_ctx.prefetched_rooms.discard(kv_chunk.room)

            except Exception as e:
                # An exception may have interrupted an engine call mid-flight;
                # its RDMA work can still be running, so quarantine the room.
                if lifetime is not None:
                    lifetime.quarantine(ENGINE_FAILURE_QUARANTINE_S)
                # NOTE(shangming): Remove this when we make sure the transfer thread is bug-free
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )
            finally:
                if lifetime is not None:
                    lifetime.end_lease()

    def _handle_prefill_message(self, waiting_req_bytes) -> None:
        """Dispatch one message from the prefill-side manager socket."""
        room = waiting_req_bytes[0].decode("ascii")
        # Staging: decode reports consumption watermark back to prefill
        if room == "WATERMARK":
            from sglang.srt.disaggregation.common.staging_handler import (
                handle_watermark_msg,
            )

            handle_watermark_msg(self._staging_ctx, waiting_req_bytes)
            return
        # Staging: decode replies with allocated staging offset
        if room == "STAGING_RSP":
            from sglang.srt.disaggregation.common.staging_handler import (
                handle_staging_rsp,
            )

            handle_staging_rsp(waiting_req_bytes, self.transfer_infos)
            return
        # Decode-side abort notification: mark room as failed and ACK
        if room == "ABORT":
            self._handle_abort_notification(waiting_req_bytes)
            return
        mooncake_session_id = waiting_req_bytes[3].decode("ascii")
        if room == "None":
            decode_kv_args = KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
            decode_kv_args.requires_dcp_relayout = self.requires_dcp_relayout(
                decode_kv_args.dst_dcp_size,
                decode_kv_args.dst_dcp_rank,
            )
            if decode_kv_args.requires_dcp_relayout:
                decode_kv_args.dcp_token_item_lens = self.prepare_dcp_token_item_lens(
                    [decode_kv_args.dst_kv_item_len] * len(self.kv_args.kv_item_lens)
                )
            self.decode_kv_args_table[mooncake_session_id] = decode_kv_args
            with self.session_lock:
                if mooncake_session_id in self.failed_sessions:
                    self.failed_sessions.remove(mooncake_session_id)
                if mooncake_session_id in self.session_failures:
                    del self.session_failures[mooncake_session_id]
            logger.debug(f"Register KVArgs from {mooncake_session_id} successfully")
            return
        else:
            self._handle_bootstrap_metadata(waiting_req_bytes)

    def start_prefill_thread(self):
        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                # Sole reader of this socket, and the only thread that can
                # receive a decode ABORT: if it dies, this rank never closes
                # aborted rooms and never ACKs, so every decode peer withholds
                # its pages until MAX_UNQUIESCED_ROOMS tears it down. The loop
                # must outlive any bug in a handler; a poisoned message is
                # dropped loudly, never fatal. recv stays outside the guard so
                # a closed socket still ends the thread at shutdown.
                try:
                    self._handle_prefill_message(waiting_req_bytes)
                except Exception:
                    logger.exception(
                        "Mooncake prefill-side message handling failed; "
                        "dropping message and continuing"
                    )

        threading.Thread(target=bootstrap_thread).start()

    def _handle_abort_notification(self, msg: List[bytes]) -> None:
        """Handle an ABORT from a decode rank (runs on the bootstrap thread)."""
        room = int(msg[1].decode("ascii"))
        decode_ip = msg[2].decode("ascii")
        decode_port = int(msg[3].decode("ascii"))
        # Peers that predate the ownership barrier send no token; their aborts
        # are honoured unconditionally, as before.
        abort_token = msg[4] if len(msg) > 4 else b""

        lifetime = self._close_room_for_abort(room, abort_token)
        # No need to abort the room if it has already succeeded
        if (
            lifetime is not None
            and room in self.request_status
            and self.check_status(room) != KVPoll.Success
        ):
            self.update_status(room, KVPoll.Failed)
            logger.debug(
                f"Received abort notification for room {room}, marked as Failed"
            )
        else:
            logger.debug(
                f"Received abort notification for room {room}, "
                f"ignoring (already completed or unknown)"
            )
        # The ACK tells decode its pages are safe to reuse, so the pump sends it
        # once this room's transfers have drained.
        self.request_abort_ack(lifetime, room, decode_ip, decode_port, abort_token)

    def _handle_bootstrap_metadata(self, msg: List[bytes]) -> bool:
        """Record a decode rank's KV destinations (runs on the bootstrap thread).

        Returns False if the metadata was dropped.
        """
        room = int(msg[0].decode("ascii"))
        mooncake_session_id = msg[3].decode("ascii")
        required_dst_info_num = int(msg[7].decode("ascii"))
        transfer_info = TransferInfo.from_zmq(msg)

        # Metadata can arrive before this rank creates the sender, so the room's
        # ownership barrier is created by whichever side gets here first. A closed
        # barrier means the room was aborted or released, and accepting
        # destinations for it would let us transfer into pages decode has freed.
        # Lifetime validation and every room-keyed write are one generation
        # transaction. The sweeper takes this same lock, so it cannot retire the
        # lifetime between validation and publication and leave orphan metadata
        # that a later generation could inherit.
        with self._room_lifetimes_lock:
            lifetime = self._room_lifetime_locked(room, create=True)
            if lifetime.is_closed():
                logger.debug(
                    "Dropping KV metadata for room %s: transfers are closed", room
                )
                return False
            lifetime.add_abort_token(transfer_info.abort_token)

            infos = self.transfer_infos.setdefault(room, {})
            infos[mooncake_session_id] = transfer_info
            # NOTE: after bootstrapping we can mark the req as waiting for input
            if len(infos) == required_dst_info_num:
                self.resolve_kv_replica_factor(infos)
                self.req_to_decode_prefix_len[room] = next(
                    (
                        info.decode_prefix_len
                        for info in infos.values()
                        if info.decode_prefix_len is not None
                    ),
                    0,
                )
                self.update_status(room, KVPoll.WaitingForInput)
        return True

    def _handle_decode_message(self, msg) -> None:
        """Dispatch one message from the decode-side manager socket."""
        if msg[0] == MooncakeKVManager.AUX_DATA_HEADER:
            self._handle_aux_data(msg)
            return

        # Staging: prefill notifies a chunk written to staging buffer
        if msg[0] == b"CHUNK_READY":
            room = int(msg[1].decode("ascii"))
            if self._is_tearing_down(room):
                return
            chunk_idx = int(msg[2].decode("ascii"))
            page_start = int(msg[3].decode("ascii"))
            num_pages = int(msg[4].decode("ascii"))
            # Prefer the prefill's unique rank id when present so that
            # writers are counted per rank rather than per session.
            writer_id = (
                msg[6].decode("ascii") if len(msg) > 6 else msg[5].decode("ascii")
            )
            handler = self._staging_handler
            assert (
                handler is not None
            ), "CHUNK_READY received before staging handler initialized"
            handler.handle_chunk_arrived(
                room,
                chunk_idx,
                page_start,
                num_pages,
                writer_id,
            )
            return

        # Staging: prefill pre-requests staging allocation before forward
        if msg[0] == b"STAGING_REQ":
            if self._is_tearing_down(int(msg[1].decode("ascii"))):
                return
            self._handle_staging_req(msg)
            return

        # Prefill acknowledges abort notification
        if msg[0] == b"ABORT_ACK":
            self._handle_abort_ack(msg)
            return

        bootstrap_room, status, prefill_rank = msg
        status = int(status.decode("ascii"))
        bootstrap_room = int(bootstrap_room.decode("ascii"))
        prefill_rank = int(prefill_rank.decode("ascii"))

        if self._is_tearing_down(bootstrap_room):
            return

        if status == KVPoll.Success:
            if bootstrap_room in self.request_status:
                self.prefill_response_tracker[bootstrap_room].add(prefill_rank)
                expected_response_num = self.required_prefill_response_num_table[
                    bootstrap_room
                ]
                arrived_response_num = len(
                    self.prefill_response_tracker[bootstrap_room]
                )
                if arrived_response_num == expected_response_num:
                    if self.enable_staging:
                        handler = self._staging_handler
                        if handler.is_staging_room(bootstrap_room):
                            handler.submit_last_scatter_async(bootstrap_room)
                    self.update_status(bootstrap_room, KVPoll.Success)
        elif status == KVPoll.Failed:
            self.record_failure(
                bootstrap_room,
                "Failed to get kvcache from prefill instance, it might be dead",
            )
            self.update_status(bootstrap_room, status)

    def start_decode_thread(self):
        def decode_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                # Sole reader of this socket, and the only thread that can
                # record an ABORT_ACK: if it dies, no abort is ever proven
                # quiesced and every aborted room withholds its pages until
                # MAX_UNQUIESCED_ROOMS tears the engine down. The loop must
                # outlive any bug in a handler; a poisoned message is dropped
                # loudly, never fatal. recv stays outside the guard so a
                # closed socket still ends the thread at shutdown.
                try:
                    self._handle_decode_message(msg)
                except Exception:
                    logger.exception(
                        "Mooncake decode-side message handling failed; "
                        "dropping message and continuing"
                    )

        threading.Thread(target=decode_thread).start()
        self._start_heartbeat_checker_thread()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
        trace_ctx: Optional[Union[TraceReqContext, TraceNullContext]] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last_chunk or (is_last_chunk and aux_index is not None)

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        lifetime = self._room_lifetime(bootstrap_room, create=False)
        if lifetime is None or lifetime.is_closed():
            # The room was aborted (possibly by a decode notification that
            # overtook this rank's own bookkeeping) or already released.
            logger.debug(
                "Room %s no longer admits transfers, dropping chunk", bootstrap_room
            )
            return
        # Stamp the chunk with this request's identity so a recycled room cannot
        # pick it up (see try_lease_chunk).
        chunk_owner = lifetime

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        if trace_ctx is None:
            trace_ctx = TraceNullContext()

        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
                owner=chunk_owner,
                num_kv_tokens=num_kv_tokens,
                trace_ctx=trace_ctx,
            )
        )

    def get_session_id(self):
        return self.engine.get_session_id()

    def _on_heartbeat_success(self, bootstrap_addr: str):
        current_rooms = self.addr_to_rooms_tracker[bootstrap_addr].copy()
        for bootstrap_room in current_rooms:
            # Remove KVPoll.Success requests from the tracker
            if bootstrap_room not in self.request_status:
                self.addr_to_rooms_tracker[bootstrap_addr].discard(bootstrap_room)

    def _run_one_probe_pass(self) -> None:
        with self.session_lock:
            snapshot = list(self.failed_sessions)
        for session_id in snapshot:
            send_probe = getattr(self.engine, "send_probe", None)
            if send_probe is None:
                rc = -1
            else:
                try:
                    rc = send_probe(session_id)
                except Exception as e:
                    logger.warning("send_probe(%s) raised: %s", session_id, e)
                    continue
            if rc == 0:
                with self.session_lock:
                    was_blacklisted = session_id in self.failed_sessions
                    self.failed_sessions.discard(session_id)
                    self.session_failures.pop(session_id, None)
                if was_blacklisted:
                    logger.info(
                        "Session %s recovered via probe; un-blacklisted",
                        session_id,
                    )
                    FAILED_SESSION_RECOVERIES.inc()
            else:
                logger.debug("Probe still failing for %s (rc=%d)", session_id, rc)

    def _failed_session_probe_loop(self) -> None:
        logger.info(
            "Starting failed-session probe loop (interval=%.1fs)",
            self.failed_session_probe_interval,
        )
        while not self._failed_session_probe_shutdown.wait(
            self.failed_session_probe_interval
        ):
            self._run_one_probe_pass()


class MooncakeKVSender(CommonKVSender):

    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        super().__init__(
            mgr,
            bootstrap_addr,
            bootstrap_room,
            dest_tp_ranks,
            pp_rank,
            req_has_disagg_prefill_dp_rank,
        )
        self.conclude_state = None
        self.init_time = time.time()
        self._quiescing = False
        self._quiesce_deadline = float("inf")
        # Join the room's ownership barrier, which decode metadata may already
        # have created, and fail fast if an abort got here first.
        if not self.kv_mgr.open_room_transfers(self.bootstrap_room):
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                "Aborted by the decode instance before KV transfer started.",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        self._init_trace_ctx()

    @mooncake_trace_func(MooncakeRequestStage.MOONCAKE_SEND)
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
    ):
        kv_indices, index_slice, is_last_chunk, should_skip = (
            self._prepare_send_indices(kv_indices, state_indices)
        )
        if should_skip:
            return

        if not is_last_chunk:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                False,
                num_kv_tokens=num_kv_tokens,
                trace_ctx=self.trace_ctx.copy_for_thread(),
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
                state_indices=state_indices,
                num_kv_tokens=num_kv_tokens,
                trace_ctx=self.trace_ctx.copy_for_thread(),
            )
        self._record_transfer_indices(kv_indices, state_indices)

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            # Hold Success until all staging chunks transferred: a deferred
            # chunk can still be pending, and concluding now would drop it.
            if (
                status == KVPoll.Success
                and self.kv_mgr._staging_outstanding.get(self.bootstrap_room, 0) > 0
            ):
                return KVPoll.Transferring
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
                self.trace_ctx.trace_req_finish()
            elif status == KVPoll.Bootstrapping:
                timeout_result = self._check_bootstrap_timeout()
                if timeout_result is not None:
                    return timeout_result

            return status
        else:
            return self.conclude_state

    # ------------------------------------------------------------------
    # Transfer ownership barrier
    #
    # A failed request keeps its KV pages until this rank's transfer workers
    # have returned. ``poll()`` still reports the logical state; the scheduler
    # defers the terminal transition via ``advance_failure_quiescence`` (see
    # ``disaggregation.utils.poll_and_all_reduce``).
    # ------------------------------------------------------------------

    def _close_barrier(self) -> None:
        """Stop admitting transfer work for this room (idempotent)."""
        if self._quiescing:
            return
        self._quiescing = True
        self._quiesce_deadline = time.monotonic() + QUIESCE_TIMEOUT_S
        self.kv_mgr.close_room_transfers(self.bootstrap_room)

    def is_failure_quiescing(self) -> bool:
        return self._quiescing

    def advance_failure_quiescence(self) -> bool:
        self._close_barrier()
        lifetime = self.kv_mgr._room_lifetime(self.bootstrap_room, create=False)
        if lifetime is None or lifetime.is_quiesced():
            self.kv_mgr.forget_unquiesced(self.bootstrap_room)
            return True
        if time.monotonic() >= self._quiesce_deadline:
            self.kv_mgr.report_unquiesced(
                self.bootstrap_room,
                f"{lifetime.outstanding_leases()} transfer worker(s) still "
                f"hold it after {QUIESCE_TIMEOUT_S:.1f}s",
            )
        return False

    def clear(self) -> None:
        super().clear()
        self.kv_mgr.forget_unquiesced(self.bootstrap_room)
        self.kv_mgr._forget_room_lifetime(self.bootstrap_room)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "Failed due to an unknown reason from another rank"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )

    def _init_trace_ctx(self):
        if self.kv_mgr.enable_trace:
            self.trace_ctx = TraceReqContext(
                rid=str(hex(self.bootstrap_room)),
                bootstrap_room=self.bootstrap_room,
                role="Sender",
                module_name="mooncake",
            )
            if not self.trace_ctx.tracing_enable:
                self.trace_ctx = TraceNullContext()
        else:
            self.trace_ctx = TraceNullContext()

        self.trace_ctx.trace_req_start()

    def abort(self):
        super().abort()
        # Close the barrier now rather than on the next poll, so no further
        # chunks are queued for a request the scheduler has given up on.
        self._close_barrier()
        self.trace_ctx.abort(abort_info={"reason": "Aborted"})
        self.trace_ctx.trace_req_finish()


class MooncakeKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.session_id = mgr.get_session_id()
        self.init_time = None
        # Ownership barrier state. Only meaningful once send_metadata() has told
        # a prefill rank where to write: before that no peer can touch our pages.
        self._metadata_sent = False
        self._quiescing = False
        self._quiesce_complete = False
        self._quiesce_deadline = float("inf")
        self._abort_targets: List[Tuple[dict, bytes]] = []
        self._expected_abort_acks: set = set()
        self._received_abort_acks: set = set()
        self._last_abort_send = float("-inf")
        self._abort_lock = threading.Lock()
        # Per-room state (ABORT_ACK routing, staging teardown) may only be
        # touched by the receiver that owns the room. A second live receiver for
        # the same bootstrap_room means the room numbers collided upstream:
        # everything keyed by room is then ambiguous. Claim before the common
        # initializer writes any room-keyed state so the loser cannot change the
        # owner's status or address tracking.
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.kv_mgr = mgr
        self._owns_room = mgr.register_receiver(self)
        if not self._owns_room:
            self.conclude_state = KVPoll.Failed
            self.require_staging = False
            self.init_time = None
            self.abort_notified = False
            self._room_collision_error = (
                f"bootstrap_room {self.bootstrap_room} is already in use by an "
                "unfinished KV transfer"
            )
            logger.error(
                "%s; rejecting the colliding receiver before it can touch "
                "room-keyed state",
                self._room_collision_error,
            )
            return
        try:
            super().__init__(mgr, bootstrap_addr, bootstrap_room)
        except Exception:
            # Do not strand the room registration if construction fails before
            # the receiver becomes operational.
            mgr.unregister_receiver(self)
            raise

    def init(self, prefill_dp_rank: int):
        if not self._owns_room:
            return
        super().init(prefill_dp_rank)

    def _register_kv_args(self) -> bool:
        for bootstrap_info in self.bootstrap_infos:
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )
            packed_state_data_ptrs = pack_int_lists(
                self.kv_mgr.kv_args.state_data_ptrs, "Q"
            )
            packed_state_item_lens = pack_int_lists(
                self.kv_mgr.kv_args.state_item_lens, "I"
            )
            packed_state_dim_per_tensor = pack_int_lists(
                getattr(self.kv_mgr.kv_args, "state_dim_per_tensor", []) or [], "I"
            )
            packed_state_layer_ids = pack_int_lists(
                self.kv_mgr.kv_args.state_layer_ids, "I"
            )
            packed_kv_layer_ids = b"".join(
                struct.pack("I", layer_id)
                for layer_id in self.kv_mgr.kv_args.kv_layer_ids
            )
            # Note(shangming): No need to add pp rank here since decode pp size should be equal to prefill pp size or 1
            tp_rank = self.kv_mgr.kv_args.engine_rank
            # Some pools have no full-token contiguous KV (kv_item_lens empty)
            # and ship per-pool instead, so report 0.
            kv_item_len = (
                self.kv_mgr.kv_args.kv_item_lens[0]
                if self.kv_mgr.kv_args.kv_item_lens
                else 0
            )
            dst_tp_rank = str(tp_rank).encode("ascii")
            dst_attn_tp_size = str(self.kv_mgr.attn_tp_size).encode("ascii")
            dst_kv_item_len = str(kv_item_len).encode("ascii")
            dst_dcp_size = str(self.kv_mgr.dcp_size).encode("ascii")
            dst_dcp_rank = str(self.kv_mgr.dcp_rank).encode("ascii")
            if (
                self.kv_mgr.enable_staging
                and self.kv_mgr._staging_ctx.allocator is not None
            ):
                _alloc = self.kv_mgr._staging_ctx.allocator
                packed_staging_base_ptr = struct.pack("Q", _alloc.get_base_ptr())
                staging_total_size_str = str(_alloc.get_total_size()).encode("ascii")
            else:
                packed_staging_base_ptr = b""
                staging_total_size_str = b""

            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            try:
                with lock:
                    sock.send_multipart(
                        [
                            "None".encode("ascii"),
                            self.kv_mgr.local_ip.encode("ascii"),
                            str(self.kv_mgr.rank_port).encode("ascii"),
                            self.session_id.encode("ascii"),
                            packed_kv_data_ptrs,
                            packed_aux_data_ptrs,
                            packed_state_data_ptrs,
                            dst_tp_rank,
                            dst_attn_tp_size,
                            dst_kv_item_len,
                            packed_state_item_lens,
                            packed_state_dim_per_tensor,
                            packed_kv_layer_ids,
                            packed_state_layer_ids,
                            packed_staging_base_ptr,
                            staging_total_size_str,
                            dst_dcp_size,
                            dst_dcp_rank,
                        ]
                    )
            except zmq.ZMQError:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"_register_kv_args to prefill {bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')} failed",
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return False
        return True

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
        device_kv_indices: Optional[npt.NDArray[np.int32]] = None,
    ):
        if not self._owns_room:
            # The owner is the only receiver allowed to publish destinations for
            # a room. Sending the loser's addresses would redirect the owner's
            # transfer into unrelated pages.
            return
        if self.bootstrap_infos is None:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        self.chunk_staging_infos = []
        if (
            self.kv_mgr.enable_staging
            and self.kv_mgr._staging_ctx.allocator is not None
        ):
            self.kv_mgr.register_staging_room_bootstrap(
                self.bootstrap_room, self.bootstrap_infos, self
            )

        for bootstrap_info, abort_token in self._abort_targets_snapshot():
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]
            try:
                with lock:
                    # Register exposure before send so an abort racing a
                    # successful send cannot receive and discard an early ACK.
                    # ZeroMQ delivers multipart messages atomically, so a send
                    # that raises exposed no complete metadata and is removed.
                    self._record_metadata_exposure(abort_token)
                    try:
                        sock.send_multipart(
                            [
                                str(self.bootstrap_room).encode("ascii"),
                                self.kv_mgr.local_ip.encode("ascii"),
                                str(self.kv_mgr.rank_port).encode("ascii"),
                                self.session_id.encode("ascii"),
                                kv_indices.tobytes() if not is_dummy else b"",
                                (
                                    str(aux_index).encode("ascii")
                                    if not is_dummy
                                    else b""
                                ),
                                (
                                    pack_int_lists(state_indices, "i")
                                    if not is_dummy and state_indices
                                    else b""
                                ),
                                str(self.required_dst_info_num).encode("ascii"),
                                str(decode_prefix_len or 0).encode("ascii"),
                                (
                                    np.asarray(
                                        device_kv_indices, dtype=np.int32
                                    ).tobytes()
                                    if not is_dummy and device_kv_indices is not None
                                    else b""
                                ),
                                abort_token,
                            ]
                        )
                    except Exception:
                        self._discard_metadata_exposure(abort_token)
                        raise
            except zmq.ZMQError:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"send_metadata to prefill {bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')} failed",
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
        elif status == KVPoll.WaitingForInput:
            timeout_result = self._check_waiting_timeout()
            if timeout_result is not None:
                return timeout_result

        return status

    # ------------------------------------------------------------------
    # Transfer ownership barrier
    #
    # Our KV pages are the *destination* of the prefill's RDMA writes, so we
    # cannot release them on the strength of a local decision alone: every
    # prefill rank we handed indices to has to confirm it has stopped writing.
    # ------------------------------------------------------------------

    def abort(self):
        if not self._owns_room:
            # CommonKVReceiver.abort() writes failure state keyed only by room,
            # which belongs to the winning receiver.
            self.conclude_state = KVPoll.Failed
            self._close_barrier()
            return
        super().abort()
        # Stop admitting staging work now rather than on the next poll.
        self._close_barrier()

    def _close_barrier(self) -> None:
        """Stop admitting transfer work for this room (idempotent)."""
        if self._quiescing:
            return
        self._quiescing = True
        self._quiesce_deadline = time.monotonic() + QUIESCE_TIMEOUT_S

    def is_failure_quiescing(self) -> bool:
        return self._quiescing

    def advance_failure_quiescence(self) -> bool:
        self._close_barrier()
        if self._quiesce_complete or not self._metadata_sent:
            # No prefill rank was ever handed this request's page indices, so
            # nothing can be writing into them.
            return True
        if not self._owns_room:
            # Another live receiver owns this bootstrap_room, so our ABORT_ACKs
            # are routed to it and proof can never reach us. Report immediately
            # instead of pretending to wait for it.
            self.kv_mgr.report_unquiesced(
                self.bootstrap_room,
                f"bootstrap_room {self.bootstrap_room} is owned by another "
                "receiver, so its acknowledgements are unroutable",
            )
            return False
        if self._peers_quiesced():
            self._quiesce_complete = True
            self.kv_mgr.forget_unquiesced(self.bootstrap_room)
            return True
        if time.monotonic() >= self._quiesce_deadline:
            with self._abort_lock:
                missing = len(self._expected_abort_acks - self._received_abort_acks)
                total = len(self._expected_abort_acks)
            self.kv_mgr.report_unquiesced(
                self.bootstrap_room,
                f"{missing} of {total} prefill ranks unacknowledged after "
                f"{QUIESCE_TIMEOUT_S:.1f}s",
            )
        return False

    def _peers_quiesced(self) -> bool:
        self._send_abort_notification()
        with self._abort_lock:
            return self._expected_abort_acks <= self._received_abort_acks

    def _abort_targets_snapshot(self) -> List[Tuple[dict, bytes]]:
        """Per-peer abort nonces, minted once and reused across retries.

        One nonce per prefill rank lets an ABORT_ACK identify *which* peer has
        drained, and prevents a stale ACK for a recycled room from satisfying
        this request.
        """
        with self._abort_lock:
            if not self._abort_targets and self.bootstrap_infos:
                self._abort_targets = [
                    (bootstrap_info, os.urandom(16).hex().encode("ascii"))
                    for bootstrap_info in self.bootstrap_infos
                ]
            return list(self._abort_targets)

    def _record_metadata_exposure(self, token: bytes) -> None:
        """Require an ACK from a peer that may now write into this room."""
        with self._abort_lock:
            self._expected_abort_acks.add(token)
            self._metadata_sent = True

    def _discard_metadata_exposure(self, token: bytes) -> None:
        """Undo an exposure whose atomic multipart send did not complete."""
        with self._abort_lock:
            self._expected_abort_acks.discard(token)
            self._received_abort_acks.discard(token)
            self._metadata_sent = bool(self._expected_abort_acks)

    def record_abort_ack(self, token: Optional[bytes]) -> None:
        with self._abort_lock:
            if not token:
                # A prefill that predates the ownership barrier acknowledges the
                # notification without echoing our nonce, and does so before its
                # transfers have drained, so its ACK proves nothing. Mixed-version
                # PD deployments are unsupported: this rank keeps withholding the
                # request's pages and eventually escalates.
                logger.warning_once(
                    "Prefill peer acknowledged an abort without echoing a "
                    "transfer-quiescence token; it acknowledges before its "
                    "transfers drain, which proves nothing. Mixed-version PD "
                    "deployments are unsupported: upgrade the prefill instances."
                )
                return
            if token in self._expected_abort_acks:
                self._received_abort_acks.add(token)

    def _send_abort_notification(self):
        """(Re-)notify prefill ranks that have not acknowledged the abort."""
        targets = self._abort_targets_snapshot()
        if not targets:
            return
        now = time.monotonic()
        with self._abort_lock:
            if now - self._last_abort_send < ABORT_RETRY_INTERVAL_S:
                return
            self._last_abort_send = now
            targets = [
                (info, token)
                for info, token in targets
                if token not in self._received_abort_acks
            ]

        for bootstrap_info, token in targets:
            # Best-effort notification to prefill side that this request was
            # aborted. This runs on the scheduler loop, and these sockets have no
            # send timeout, so it must never wait for a peer: zmq.NOBLOCK turns a
            # backed-up or dead prefill into a retry on the next poll instead of
            # an unbounded stall of the whole engine.
            try:
                sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
                with lock:
                    sock.send_multipart(
                        [
                            b"ABORT",
                            str(self.bootstrap_room).encode("ascii"),
                            self.kv_mgr.local_ip.encode("ascii"),
                            str(self.kv_mgr.rank_port).encode("ascii"),
                            token,
                        ],
                        zmq.NOBLOCK,
                    )
            except zmq.Again:
                # Retried by the next _send_abort_notification() tick.
                with self._abort_lock:
                    self._last_abort_send = float("-inf")
            except Exception as e:
                logger.debug(
                    "Failed to send abort notification for room %s: %s",
                    self.bootstrap_room,
                    e,
                )

    def clear(self) -> None:
        if not self._owns_room:
            # Everything CommonKVReceiver.clear() drops is keyed by
            # bootstrap_room and therefore belongs to the receiver that owns it.
            # Tearing it down here would fail the owner's live request as well.
            logger.debug(
                "Skipping shared cleanup for bootstrap_room %s: owned elsewhere",
                self.bootstrap_room,
            )
            return
        super().clear()
        self.kv_mgr.forget_unquiesced(self.bootstrap_room)
        self.kv_mgr.unregister_receiver(self)

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        if not self._owns_room:
            raise KVTransferError(
                self.bootstrap_room,
                self._room_collision_error,
                is_from_another_rank=False,
            )

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "Failed due to an unknown reason from another rank"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )


class MooncakeKVBootstrapServer(CommonKVBootstrapServer):
    pass
