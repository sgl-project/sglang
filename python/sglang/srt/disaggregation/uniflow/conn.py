"""UniFlow KV transfer backend for SGLang prefill/decode disaggregation.

The lifecycle is prefill-driven, like the other SGLang backends: decode
registers its exported KV, aux, and state segments with prefill (REGISTER),
then sends per-request destination indices (METADATA); prefill writes the data
with one-sided UniFlow put() and reports the terminal status (STATUS). Decode
reports aborts and timeouts to prefill (ABORT). All control frames travel over
the CommonKVManager ZMQ side channel and start with GUARD:

    REGISTER  [GUARD, register, json(agent_id, export ids)]
    METADATA  [GUARD, metadata, room, endpoint, port, agent_id, dst_kv_indices,
               dst_aux_index, dst_state_indices, decode_prefix_len]
    STATUS    [GUARD, room, KVPoll, prefill rank, failure reason]
    ABORT     [GUARD, abort, room, reason, decode endpoint, decode port]

STATUS is the CommonKVManager status message with GUARD as its tag. An MLA
dummy rank gets METADATA with empty KV and aux frames; like the Mooncake
backend, it moves no data and sends no STATUS.

With SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE, prefill answers an ABORT
with the common [ABORT_ACK, room, prefill rank] frame once every put queued for
the room before the ABORT has returned. A room with a failed or timed-out put
is never acked, as with a failed Mori or NIXL write: only the RDMA tier's
shutdown waits for in-flight puts, so decode keeps the pages until its
deferred-release timeout. A decode transfer queue that does not honor the flag
releases the pages on Failed, so a failed put still shuts its connection down
before prefill reports Failed (see DESIGN.md).

Scope is intentionally narrow: CP=1, PP=1, DCP=1, no hisparse or custom memory
pool, one-to-one prefill/decode rank routing (MLA dummy ranks included), and
mamba, SWA, or DSA state.

The manager is the protocol of the Transport interface (RFC #46455): it plans
each put as Copy byte ranges over UniflowBufferKey names, and UniflowTransport
(transport.py) owns the UniFlow agent, segments, connections and puts.
"""

from __future__ import annotations

import functools
import json
import logging
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

import msgspec
import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import (
    KVArgs,
    KVPoll,
    KVTransferDestination,
    StateType,
)
from sglang.srt.disaggregation.common.conn import (
    AckTarget,
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
    KVTransferError,
)
from sglang.srt.disaggregation.common.utils import (
    group_concurrent_contiguous,
    pack_int_lists,
    unpack_int_lists,
)
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.transport.base import Copy, Status
from sglang.srt.disaggregation.uniflow.buffers import (
    BufferKeys,
    BufferKind,
    copy_ranges,
    group_units,
)
from sglang.srt.disaggregation.uniflow.transport import (
    Registration,
    UniflowRemoteBuffers,
    UniflowTransport,
    import_uniflow_core,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_memory, get_parallel
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# First frame of every UniFlow control message, and the STATUS message tag.
GUARD = b"UniflowMsgGuard"
REGISTER_MSG = b"register"
METADATA_MSG = b"metadata"
ABORT_MSG = b"abort"
# METADATA aux index of a request without aux data.
NO_AUX_INDEX = -1
# A late abort for a cleared room is never consumed, so the oldest are evicted.
MAX_PENDING_ABORTS = 4096
DEFAULT_ABORT_REASON = "UniFlow request aborted"
SUPPORTED_STATE_TYPES = frozenset({StateType.MAMBA, StateType.SWA, StateType.DSA})
# Paged state that tolerates an index-count drift, as in the Mooncake backend.
TRUNCATABLE_STATE_TYPES = frozenset({StateType.SWA, StateType.DSA})


class UniflowPutError(RuntimeError):
    """A put() that failed or timed out; its writes may still be in flight."""


def _flatten_component_indices(component: Any) -> list[int]:
    if component is None:
        return []
    if isinstance(component, np.ndarray):
        return component.astype(np.int32).reshape(-1).tolist()
    if isinstance(component, (list, tuple)):
        flat: list[int] = []
        for item in component:
            flat.extend(_flatten_component_indices(item))
        return flat
    return [int(component)]


def flatten_optional_indices(
    state_indices: Sequence[Any] | None,
) -> list[list[int]]:
    """Flattens per-component state indices (arrays, nested lists) to int lists."""
    if not state_indices:
        return []
    return [_flatten_component_indices(item) for item in state_indices]


class DecodePeerInfo(msgspec.Struct, frozen=True, kw_only=True):
    agent_id: str
    kv_export_ids: list[str]
    aux_export_ids: list[str]
    state_export_ids: list[list[str]]

    @classmethod
    def from_json_bytes(cls, data: bytes) -> DecodePeerInfo:
        payload = json.loads(data.decode("utf-8"))
        return cls(
            agent_id=str(payload["agent_id"]),
            kv_export_ids=[str(x) for x in payload["kv_export_ids"]],
            aux_export_ids=[str(x) for x in payload["aux_export_ids"]],
            state_export_ids=[
                [str(x) for x in component] for component in payload["state_export_ids"]
            ],
        )


class TransferInfo(msgspec.Struct, frozen=True, kw_only=True):
    room: int
    endpoint: str
    dst_port: int
    agent_id: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    dst_state_indices: list[list[int]]
    decode_prefix_len: int | None
    is_dummy: bool

    @classmethod
    def from_zmq(cls, msg: list[bytes]) -> TransferInfo:
        # A real rank always sends an aux index, NO_AUX_INDEX included, so empty
        # KV and aux frames mark an MLA dummy rank.
        is_dummy = msg[6] == b"" and msg[7] == b""
        return cls(
            room=int(msg[2].decode("ascii")),
            endpoint=msg[3].decode("ascii"),
            dst_port=int(msg[4].decode("ascii")),
            agent_id=msg[5].decode("ascii"),
            dst_kv_indices=np.frombuffer(msg[6], dtype=np.int32),
            dst_aux_index=NO_AUX_INDEX if is_dummy else int(msg[7].decode("ascii")),
            dst_state_indices=(
                unpack_int_lists(msg[8], "i") if len(msg) > 8 and msg[8] else []
            ),
            decode_prefix_len=(
                int(msg[9].decode("ascii")) if len(msg) > 9 and msg[9] else None
            ),
            is_dummy=is_dummy,
        )


class PeerBuffers(msgspec.Struct, frozen=True):
    """A decode peer as imported from its REGISTER: the transport's handle and
    the keys of its buffers."""

    remote: UniflowRemoteBuffers
    keys: BufferKeys


class UniflowKVManager(CommonKVManager):
    kv_status_msg_tag = GUARD
    kv_status_msg_carries_reason = True
    # The ABORT handler queues the ack behind every put already queued for the room.
    supports_deferred_decode_kv_release = True
    reports_transfer_latency = True

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: bool | None = False,
    ):
        parallel = get_parallel()
        if parallel.attn_cp_size != 1 or parallel.pp_size != 1:
            raise ValueError(
                "UniFlow disaggregation currently supports CP=1 and PP=1 only."
            )
        if parallel.attn_dcp_size > 1:
            raise ValueError(
                "UniFlow disaggregation does not support decode context "
                "parallelism (--dcp-size > 1); use another transfer backend."
            )
        transfer_timeout_ms = (
            envs.SGLANG_DISAGGREGATION_UNIFLOW_TRANSFER_TIMEOUT_MS.get()
        )
        if transfer_timeout_ms <= 0:
            raise ValueError(
                "SGLANG_DISAGGREGATION_UNIFLOW_TRANSFER_TIMEOUT_MS must be positive, "
                f"got {transfer_timeout_ms}."
            )
        if get_memory().enable_hisparse:
            raise ValueError(
                "UniFlow disaggregation does not support hisparse; "
                "use another transfer backend."
            )
        if envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() is not None:
            # This Mooncake-only mode allocates KV from Mooncake's pool and can
            # place aux buffers on the GPU, which UniFlow registers as DRAM.
            raise ValueError(
                "UniFlow disaggregation does not support "
                "SGLANG_MOONCAKE_CUSTOM_MEM_POOL; unset it."
            )
        self.state_types = [StateType(state_type) for state_type in args.state_types]
        unsupported = [
            state_type.value
            for state_type in self.state_types
            if state_type not in SUPPORTED_STATE_TYPES
        ]
        if unsupported:
            raise ValueError(
                f"UniFlow disaggregation does not support state types {unsupported}; "
                "use another transfer backend."
            )
        if len(self.state_types) != len(args.state_data_ptrs or []):
            raise ValueError(
                "UniFlow state type/component count mismatch: "
                f"types={len(self.state_types)} "
                f"components={len(args.state_data_ptrs or [])}"
            )

        # Fails before CommonKVManager.__init__ opens any socket or thread.
        uniflow_core = import_uniflow_core()

        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

        self.transport = UniflowTransport.create(
            uniflow_core,
            device_id=self.kv_args.gpu_id,
            name=f"sglang_uniflow_{disaggregation_mode.value}_{self.kv_args.engine_rank}",
            host=self.local_ip,
        )
        self.agent_id = self.transport.agent_id
        self._local_keys = BufferKeys.for_buffers(
            self.kv_args.kv_data_ptrs,
            self.kv_args.aux_data_ptrs,
            self.kv_args.state_data_ptrs or [],
        )
        # Keyed by decode agent_id. A peer is dropped after a failed put and
        # reconnected on next use; peers of departed decode instances are not
        # reclaimed.
        self.decode_peer_infos: dict[str, DecodePeerInfo] = {}
        self._peers: dict[str, PeerBuffers] = {}
        self.decode_peers_lock = threading.Lock()
        # Re-entrant (the default RLock) so helpers called under it can
        # re-acquire it.
        self.condition = threading.Condition()
        # Decode aborts (room -> reason) for rooms with no status. The sender's
        # __init__ consumes early ones; ones arriving after clear() age out.
        self.pending_aborts: OrderedDict[int, str] = OrderedDict()
        self.transfer_timeout_ms = transfer_timeout_ms
        # Prefill puts run on one FIFO worker so the scheduler never blocks on
        # completion. One worker keeps per-room chunk order, confines
        # Connection use to a single thread, and runs an ABORT_ACK only after
        # every chunk queued before it.
        self._transfer_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        if self.is_mla_backend:
            # One-to-one routing gives each prefill rank one decode destination.
            self._kv_replica_factor = 1

        self._register_buffers()
        self.transport.start_accepting()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._start_transfer_worker()
            self._start_message_thread(self._handle_prefill_message, "prefill")
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self._start_message_thread(self._handle_decode_message, "decode")
            # Fails in-flight rooms of a prefill whose bootstrap server stops
            # answering /health, as the other backends do.
            self._start_heartbeat_checker_thread()

    def _register_buffers(self) -> None:
        # No unwind on failure: a raise here aborts startup, and process exit
        # releases the registrations.
        if len(self.kv_args.kv_data_ptrs) != len(self.kv_args.kv_data_lens):
            raise ValueError(
                "UniFlow KV data ptr/len count mismatch: "
                f"ptrs={len(self.kv_args.kv_data_ptrs)} "
                f"lens={len(self.kv_args.kv_data_lens)}"
            )
        if len(self.kv_args.aux_data_ptrs) != len(self.kv_args.aux_data_lens):
            raise ValueError(
                "UniFlow aux data ptr/len count mismatch: "
                f"ptrs={len(self.kv_args.aux_data_ptrs)} "
                f"lens={len(self.kv_args.aux_data_lens)}"
            )
        state_data_ptrs = self.kv_args.state_data_ptrs or []
        state_data_lens = self.kv_args.state_data_lens or []
        if len(state_data_ptrs) != len(state_data_lens):
            raise ValueError(
                "UniFlow state data component-count mismatch: "
                f"ptrs={len(state_data_ptrs)} lens={len(state_data_lens)}"
            )
        for ptrs, lens in zip(state_data_ptrs, state_data_lens, strict=True):
            if len(ptrs) != len(lens):
                raise ValueError(
                    "UniFlow state data ptr/len mismatch within component: "
                    f"ptrs={len(ptrs)} lens={len(lens)}"
                )
        self.transport.register_all(self._kv_args_registrations())
        self.kv_export_ids = self.transport.export_ids(self._local_keys.kv)
        self.aux_export_ids = self.transport.export_ids(self._local_keys.aux)
        self.state_export_ids = [
            self.transport.export_ids(keys) for keys in self._local_keys.state
        ]

    def _kv_args_registrations(self) -> list[Registration]:
        """KV, aux and state buffers in REGISTER message order."""
        state_data_ptrs = self.kv_args.state_data_ptrs or []
        lengths = dict(
            self._local_keys.pair(
                self.kv_args.kv_data_lens,
                self.kv_args.aux_data_lens,
                self.kv_args.state_data_lens or [],
            )
        )
        return [
            Registration(
                key=key,
                ptr=ptr,
                length=lengths[key],
                # Aux buffers are host memory, which UniFlow registers as DRAM.
                device=-1 if key.kind == BufferKind.AUX else self.kv_args.gpu_id,
            )
            for key, ptr in self._local_keys.pair(
                self.kv_args.kv_data_ptrs, self.kv_args.aux_data_ptrs, state_data_ptrs
            )
        ]

    def _handle_prefill_message(self, msg: list[bytes]) -> None:
        if not msg or msg[0] != GUARD:
            logger.warning(
                "Ignoring non-UniFlow prefill message: frames=%d first=%r",
                len(msg),
                msg[0][:32] if msg else None,
            )
            return
        if len(msg) < 2:
            logger.warning(
                "Ignoring malformed UniFlow prefill message: frames=%d", len(msg)
            )
            return
        msg_type = msg[1]
        # Minimum frame counts; shorter frames are dropped with a warning.
        min_len = {REGISTER_MSG: 3, ABORT_MSG: 3, METADATA_MSG: 8}.get(msg_type)
        if min_len is not None and len(msg) < min_len:
            logger.warning(
                "Ignoring malformed UniFlow %s message: frames=%d",
                msg_type.decode("ascii", "replace"),
                len(msg),
            )
            return
        if msg_type == REGISTER_MSG:
            peer_info = DecodePeerInfo.from_json_bytes(msg[2])
            with self.condition:
                self.decode_peer_infos[peer_info.agent_id] = peer_info
                self.condition.notify_all()
            return
        if msg_type == ABORT_MSG:
            room = int(msg[2].decode("ascii"))
            reason = msg[3].decode("utf-8") if len(msg) > 3 else DEFAULT_ABORT_REASON
            with self.condition:
                status = self.request_status.get(room)
                if status in (KVPoll.Success, KVPoll.Failed):
                    logger.info(
                        "Ignoring late UniFlow abort for concluded room=%s",
                        room,
                    )
                else:
                    self.transfer_infos.pop(room, None)
                    self.req_to_decode_prefix_len.pop(room, None)
                    if status is None:
                        self.pending_aborts[room] = reason
                        while len(self.pending_aborts) > MAX_PENDING_ABORTS:
                            self.pending_aborts.popitem(last=False)
                    else:
                        # Failed is terminal in update_status, so
                        # every later check of this room sees it.
                        self.record_failure(room, reason)
                        self.update_status(room, KVPoll.Failed)
                    self.condition.notify_all()
            if self.enable_deferred_decode_kv_release and len(msg) >= 7:
                # Decode holds the aborted pages until this ack. It is
                # queued only after the room can no longer transfer,
                # and the single FIFO worker runs it after every chunk
                # queued before it has finished its put. A decode that
                # did not arm a generation does not wait for an ack.
                target = AckTarget(
                    msg[4].decode("ascii"),
                    int(msg[5].decode("ascii")),
                    int(msg[6].decode("ascii")),
                )
                self._transfer_queue.put(
                    functools.partial(
                        self._send_abort_ack_unless_poisoned, room, target
                    )
                )
            return
        if msg_type != METADATA_MSG:
            return
        transfer_info = TransferInfo.from_zmq(msg)
        with self.condition:
            self.transfer_infos[transfer_info.room] = transfer_info
            self.req_to_decode_prefix_len[transfer_info.room] = (
                transfer_info.decode_prefix_len or 0
            )
            self.update_status(transfer_info.room, KVPoll.WaitingForInput)
            self.condition.notify_all()

    def _start_message_thread(
        self, handle_message: Callable[[list[bytes]], None], role: str
    ) -> None:
        def message_thread() -> None:
            while True:
                try:
                    handle_message(self.server_socket.recv_multipart())
                except Exception:
                    # Survive socket errors; a dead thread would drop all
                    # control messages.
                    logger.exception("Failed to process UniFlow %s message", role)
                    time.sleep(0.1)

        threading.Thread(target=message_thread, daemon=True).start()

    def _handle_decode_message(self, msg: list[bytes]) -> None:
        # Common deferred-release ack, not GUARD-framed. Counted only for the
        # generation that currently holds the room's pages.
        if self.handle_abort_ack_message(msg):
            return
        if not msg or msg[0] != GUARD:
            logger.warning(
                "Ignoring non-UniFlow decode message: frames=%d first=%r",
                len(msg),
                msg[0][:32] if msg else None,
            )
            return
        parsed = self.parse_kv_status_message(msg)
        if parsed is None:
            return
        room, status, prefill_rank, failure_reason = parsed
        self.apply_prefill_status(
            bootstrap_room=room,
            status=status,
            prefill_rank=prefill_rank,
            failure_reason=failure_reason,
        )

    def build_receiver_registration_message(self) -> list[bytes]:
        payload = json.dumps(
            {
                "agent_id": self.agent_id,
                "kv_export_ids": self.kv_export_ids,
                "aux_export_ids": self.aux_export_ids,
                "state_export_ids": self.state_export_ids,
            }
        ).encode("utf-8")
        return [GUARD, REGISTER_MSG, payload]

    def build_receiver_metadata_message(
        self,
        *,
        bootstrap_room: int,
        required_dst_info_num: int,
        kv_indices: npt.NDArray[np.int32],
        aux_index: int | None,
        state_indices: list | None,
        decode_prefix_len: int | None,
        is_dummy: bool,
        device_kv_indices: npt.NDArray[np.int32] | None = None,
    ) -> list[bytes]:
        # One-to-one routing: the receiver rejects required_dst_info_num != 1.
        if is_dummy:
            kv_frame = aux_frame = state_frame = b""
        else:
            kv_frame = np.ascontiguousarray(kv_indices, dtype=np.int32).tobytes()
            aux_frame = str(NO_AUX_INDEX if aux_index is None else aux_index).encode(
                "ascii"
            )
            state_frame = pack_int_lists(flatten_optional_indices(state_indices), "i")
        return [
            GUARD,
            METADATA_MSG,
            str(bootstrap_room).encode("ascii"),
            self.local_ip.encode("ascii"),
            str(self.rank_port).encode("ascii"),
            self.agent_id.encode("ascii"),
            kv_frame,
            aux_frame,
            state_frame,
            (
                str(decode_prefix_len).encode("ascii")
                if decode_prefix_len is not None
                else b""
            ),
        ]

    def _get_or_connect_decode_peer(self, peer_info: DecodePeerInfo) -> PeerBuffers:
        with self.decode_peers_lock:
            cached = self._peers.get(peer_info.agent_id)
            if cached is not None:
                return cached

            local = self._local_keys
            if len(peer_info.kv_export_ids) != len(local.kv):
                raise ValueError(
                    "UniFlow KV segment count mismatch: "
                    f"local={len(local.kv)} "
                    f"remote={len(peer_info.kv_export_ids)}"
                )
            if len(peer_info.aux_export_ids) != len(local.aux):
                raise ValueError(
                    "UniFlow aux segment count mismatch: "
                    f"local={len(local.aux)} "
                    f"remote={len(peer_info.aux_export_ids)}"
                )
            if len(peer_info.state_export_ids) != len(local.state):
                raise ValueError(
                    "UniFlow state component count mismatch: "
                    f"local={len(local.state)} "
                    f"remote={len(peer_info.state_export_ids)}"
                )
            for local_component, remote_export_ids in zip(
                local.state, peer_info.state_export_ids, strict=True
            ):
                if len(local_component) != len(remote_export_ids):
                    raise ValueError(
                        "UniFlow state segment count mismatch within component: "
                        f"local={len(local_component)} remote={len(remote_export_ids)}"
                    )

            export_ids = (
                peer_info.kv_export_ids,
                peer_info.aux_export_ids,
                peer_info.state_export_ids,
            )
            keys = BufferKeys.for_buffers(*export_ids)
            remote = self.transport.import_registration(
                agent_id=peer_info.agent_id, export_ids=dict(keys.pair(*export_ids))
            )
            peer = PeerBuffers(remote=remote, keys=keys)
            self._peers[peer_info.agent_id] = peer
            return peer

    def wait_for_transfer_info(self, room: int) -> tuple[TransferInfo, DecodePeerInfo]:
        with self.condition:
            # UniflowKVSender.abort() and clear() notify, so a room its sender
            # concludes ends the wait instead of stalling the worker.
            found_transfer = self.condition.wait_for(
                lambda: (
                    self.request_status.get(room) in (None, KVPoll.Failed)
                    or (
                        room in self.transfer_infos
                        and self.transfer_infos[room].agent_id in self.decode_peer_infos
                    )
                ),
                timeout=float(self.bootstrap_timeout),
            )
            status = self.request_status.get(room)
            if status is None:
                # Cleared by its sender, which already consumed any failure.
                raise KVTransferError(
                    room, f"UniFlow room={room} concluded before its transfer started"
                )
            if status == KVPoll.Failed:
                with self.failure_lock:
                    reason = self.failure_records.get(room, DEFAULT_ABORT_REASON)
                raise KVTransferError(room, reason)
            if not found_transfer:
                reason = (
                    "UniFlow transfer metadata or decode peer registration did not "
                    f"arrive before bootstrap timeout for room={room}"
                )
                # Only clear() removes a room and it holds the condition, so the
                # room is still live here.
                self.record_failure(room, reason)
                self.update_status(room, KVPoll.Failed)
                raise KVTransferError(room, reason)
            info = self.transfer_infos[room]
            return info, self.decode_peer_infos[info.agent_id]

    def update_status_and_clear_transfer(self, room: int, status: KVPoll) -> None:
        with self.condition:
            self.update_status(room, status)
            self.transfer_infos.pop(room, None)
            # Normally consumed by pop_decode_prefix_len; covers rooms that fail
            # before send.
            self.req_to_decode_prefix_len.pop(room, None)

    def mark_transfer_success(self, info: TransferInfo) -> None:
        # Not conclude_transfer: one-to-one routing gives a room one
        # TransferInfo, and the status send stays outside the condition.
        with self.condition:
            status = self.request_status.get(info.room)
            if status is None:
                return
            failure_reason = None
            if status == KVPoll.Failed:
                # Aborted during the last put. Never promote Failed, but still
                # tell decode, which otherwise waits for its timeout.
                with self.failure_lock:
                    failure_reason = self.failure_records.get(
                        info.room, DEFAULT_ABORT_REASON
                    )
            self.update_status_and_clear_transfer(
                info.room, KVPoll.Success if failure_reason is None else KVPoll.Failed
            )
        if not info.is_dummy:
            self.send_kv_status_message(
                targets=[(info.endpoint, info.dst_port)],
                bootstrap_room=info.room,
                status=KVPoll.Success if failure_reason is None else KVPoll.Failed,
                failure_reason=failure_reason,
            )

    def mark_transfer_failure(
        self, info: TransferInfo, error: Exception, *, should_notify_decode: bool = True
    ) -> None:
        reason = f"UniFlow transfer failed: {error}"
        with self.condition:
            # A room the sender already concluded and cleared has no one left to
            # consume the failure; recording it would only leak the entry.
            if self.request_status.get(info.room) is not None:
                # Keep the first cause, such as a decode abort that raced the put.
                with self.failure_lock:
                    reason = self.failure_records.setdefault(info.room, reason)
                self.update_status_and_clear_transfer(info.room, KVPoll.Failed)
        if should_notify_decode:
            self.send_kv_status_message(
                targets=[(info.endpoint, info.dst_port)],
                bootstrap_room=info.room,
                status=KVPoll.Failed,
                failure_reason=reason,
            )

    def _close_decode_peer(self, agent_id: str) -> bool:
        """Drops a decode peer after a failed put; returns False if shutdown raised.

        Decode is told Failed only after shutdown() returns, but only the RDMA
        tier waits there for in-flight puts. On TCP, bytes already handed to the
        kernel may still land; on the intra-host GPU peer tier, queued copies
        still complete. The next transfer reconnects.
        """
        with self.decode_peers_lock:
            peer = self._peers.pop(agent_id, None)
        if peer is None:
            return True
        try:
            self.transport.disconnect(peer.remote)
        except Exception:
            logger.exception("UniFlow shutdown of decode peer %s failed", agent_id)
            return False
        return True

    def _send_abort_ack_unless_poisoned(self, room: int, target: AckTarget) -> None:
        # Runs on the transfer worker, after every put queued before the abort.
        if room in self._deferred_ack_poisoned_rooms:
            logger.warning(
                "Withholding UniFlow ABORT_ACK for room=%s: a put failed and its "
                "writes may still land",
                room,
            )
            return
        self._send_abort_ack(room, target)

    def _is_room_stopped(self, room: int) -> bool:
        """Whether decode or the sender aborted the room, or the sender cleared it."""
        with self.condition:
            return self.request_status.get(room) in (None, KVPoll.Failed)

    def _start_transfer_worker(self) -> None:
        def transfer_worker() -> None:
            while True:
                task = self._transfer_queue.get()
                try:
                    task()
                except Exception:
                    logger.exception("UniFlow transfer worker failed")

        threading.Thread(target=transfer_worker, daemon=True).start()

    def add_transfer_request(
        self,
        room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        aux_index: int | None = None,
        state_indices: list | None = None,
        num_kv_tokens: int | None = None,
        *,
        wait_event: Any | None = None,
        indices_event: Any | None = None,
        trace_ctx: Any | None = None,
    ) -> None:
        # UniflowKVSender does not set supports_host_tensor_indices, so the
        # scheduler hands over host indices and never an indices_event.
        assert indices_event is None
        self._transfer_queue.put(
            functools.partial(
                self._do_transfer_request,
                room,
                np.ascontiguousarray(kv_indices, dtype=np.int32),
                index_slice=index_slice,
                is_last=is_last_chunk,
                aux_index=aux_index,
                state_indices=state_indices,
                wait_event=wait_event,
            )
        )

    def _do_transfer_request(
        self,
        room: int,
        kv_indices: npt.NDArray[np.int32],
        *,
        index_slice: slice,
        is_last: bool,
        aux_index: int | None,
        state_indices: list | None,
        wait_event: Any | None,
    ) -> None:
        # Skip queued chunks of a room that already concluded (failed, aborted,
        # or cleared by its sender) rather than re-waiting for metadata until
        # bootstrap timeout, which would stall the worker for every other room.
        if self.request_status.get(room) in (None, KVPoll.Failed, KVPoll.Success):
            return
        try:
            info, peer_info = self.wait_for_transfer_info(room)
        except KVTransferError as e:
            # The room is already Failed with its reason recorded.
            logger.warning("UniFlow transfer for room=%s not started: %s", room, e)
            return
        if info.is_dummy:
            if is_last:
                self.mark_transfer_success(info)
            return
        is_complete = False
        try:
            peer = self._get_or_connect_decode_peer(peer_info)
            # Decode may have aborted the room while it waited in the queue. The
            # re-check narrows but does not close that window; only a decode
            # transfer queue that honors
            # SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE keeps the pages
            # until prefill acks.
            if self._is_room_stopped(room):
                return
            dst_kv_indices = info.dst_kv_indices[index_slice]
            if len(dst_kv_indices) != len(kv_indices):
                raise ValueError(
                    "UniFlow source and destination KV slice sizes differ: "
                    f"source={len(kv_indices)} destination={len(dst_kv_indices)}"
                )
            if wait_event is not None:
                # Early send: the prior forward may still be writing these pages.
                wait_event.synchronize()
            if len(kv_indices) > 0:
                self._put_kv(peer, kv_indices, dst_kv_indices)
            if is_last:
                # A room stopped during an earlier put gets no further writes;
                # mark_transfer_success still reports a Failed room to decode.
                if self._local_keys.aux:
                    if aux_index is None or info.dst_aux_index == NO_AUX_INDEX:
                        raise ValueError("UniFlow last chunk requires aux_index")
                    if not self._is_room_stopped(room):
                        self._put_aux(peer, aux_index, info.dst_aux_index)
                if not self._is_room_stopped(room):
                    self._put_state(peer, state_indices, info.dst_state_indices)
                is_complete = True
            else:
                with self.condition:
                    self.update_status(room, KVPoll.Transferring)
        except UniflowPutError as e:
            # Its writes may still land (only the RDMA tier's shutdown waits for
            # them), so the room is never acked. Shut down before reporting
            # Failed, which a decode without deferred release acts on at once;
            # if shutdown raised, leave decode to its timeouts.
            if self.enable_deferred_decode_kv_release:
                self.poison_deferred_ack_room(room)
            is_shut_down = self._close_decode_peer(info.agent_id)
            self.mark_transfer_failure(info, e, should_notify_decode=is_shut_down)
            return
        except Exception as e:
            self.mark_transfer_failure(info, e)
            return

        if is_complete:
            self.mark_transfer_success(info)

    def _put_kv(
        self,
        peer: PeerBuffers,
        src_indices: npt.NDArray[np.int32],
        dst_indices: npt.NDArray[np.int32],
    ) -> None:
        # One run of pages contiguous on both sides is one range per KV buffer.
        units = group_units(*group_concurrent_contiguous(src_indices, dst_indices))
        copies = [
            Copy(src=src, dst=dst, ranges=units * np.uint64(item_len))
            for src, dst, item_len in zip(
                self._local_keys.kv,
                peer.keys.kv,
                self.kv_args.kv_item_lens,
                strict=True,
            )
        ]
        self._put(peer, copies, "KV put")

    def _put_aux(self, peer: PeerBuffers, src_index: int, dst_index: int) -> None:
        copies = [
            Copy(
                src=src,
                dst=dst,
                ranges=copy_ranges(
                    [(src_index * item_len, dst_index * item_len, item_len)]
                ),
            )
            for src, dst, item_len in zip(
                self._local_keys.aux,
                peer.keys.aux,
                self.kv_args.aux_item_lens,
                strict=True,
            )
        ]
        self._put(peer, copies, "aux put")

    def _put_state(
        self,
        peer: PeerBuffers,
        src_state_indices: list | None,
        dst_state_indices: list[list[int]],
    ) -> None:
        src_components = flatten_optional_indices(src_state_indices)
        if not src_components:
            return
        if len(src_components) != len(dst_state_indices):
            raise ValueError(
                "UniFlow source and destination state component counts differ: "
                f"source={len(src_components)} destination={len(dst_state_indices)}"
            )

        copies = []
        for component_id, (src_indices, dst_indices) in enumerate(
            zip(src_components, dst_state_indices, strict=True)
        ):
            count = min(len(src_indices), len(dst_indices))
            if len(src_indices) != len(dst_indices):
                state_type = self.state_types[component_id]
                if state_type not in TRUNCATABLE_STATE_TYPES:
                    raise ValueError(
                        "UniFlow source and destination state index counts differ: "
                        f"component={component_id} type={state_type.value} "
                        f"source={len(src_indices)} destination={len(dst_indices)}"
                    )
                logger.warning(
                    "Truncating UniFlow %s state indices: source=%d destination=%d",
                    state_type.value,
                    len(src_indices),
                    len(dst_indices),
                )
            pairs = list(zip(src_indices[:count], dst_indices[:count], strict=True))
            for src, dst, item_len in zip(
                self._local_keys.state[component_id],
                peer.keys.state[component_id],
                self.kv_args.state_item_lens[component_id],
                strict=True,
            ):
                if item_len == 0:
                    # An elided DSA index-K entry is never registered and moves
                    # nothing, as in the other backends.
                    continue
                copies.append(
                    Copy(
                        src=src,
                        dst=dst,
                        ranges=copy_ranges(
                            (s * item_len, d * item_len, item_len) for s, d in pairs
                        ),
                    )
                )
        self._put(peer, copies, "state put")

    def _put(self, peer: PeerBuffers, copies: list[Copy], action: str) -> None:
        # Returns only after the batch completes, so a later raise leaves no
        # write in flight. Each call is one put batch, so device KV never
        # shares a batch with host aux and keeps the GPU peer tier.
        handles = self.transport.write_all(peer.remote, copies)
        if not handles:
            return
        result = self.transport.wait_all(handles, timeout_ms=self.transfer_timeout_ms)
        if result.status is Status.PENDING:
            raise UniflowPutError(f"UniFlow {action} timed out")
        if result.status is Status.FAILED:
            # A binding error may leave the batch on the wire;
            # the caller shuts the connection down.
            raise UniflowPutError(f"UniFlow {action} failed: {result.reason}")


class UniflowKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: UniflowKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        super().__init__(
            mgr,
            bootstrap_addr,
            bootstrap_room,
            req_has_disagg_prefill_dp_rank,
        )
        with mgr.condition:
            reason = mgr.pending_aborts.pop(bootstrap_room, None)
            if reason is not None:
                mgr.record_failure(bootstrap_room, reason)
                mgr.update_status(bootstrap_room, KVPoll.Failed)

    def clear(self) -> None:
        # Atomic with the transfer worker, and wakes one waiting on this room.
        with self.kv_mgr.condition:
            super().clear()
            self.kv_mgr.condition.notify_all()

    def abort(self) -> None:
        # Keep a cause already recorded for this room, such as a failed put or a
        # decode abort, over the base abort's reason: schedulers that conclude
        # a failed sender call abort() before failure_exception(). Also wakes a
        # transfer worker waiting on this room's metadata.
        with self.kv_mgr.condition:
            with self.kv_mgr.failure_lock:
                reason = self.kv_mgr.failure_records.get(self.bootstrap_room)
            super().abort()
            if reason is not None:
                self.kv_mgr.record_failure(self.bootstrap_room, reason)
            self.kv_mgr.condition.notify_all()


class UniflowKVReceiver(CommonKVReceiver):
    def init(self, prefill_dp_rank: int):
        super().init(prefill_dp_rank)
        if self.conclude_state == KVPoll.Failed:
            return
        if self.required_dst_info_num != 1 or self.required_prefill_response_num != 1:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                "UniFlow disaggregation currently supports homogeneous "
                "one-to-one TP/CP/PP routing only. "
                f"required_dst_info_num={self.required_dst_info_num}, "
                f"required_prefill_response_num={self.required_prefill_response_num}.",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            self.conclude_state = KVPoll.Failed
            # Tell prefill now rather than let it hold the KV until its
            # bootstrap timeout.
            self.ensure_abort_notified()

    def abort(self) -> None:
        # Keep a cause already recorded for this room, such as a prefill
        # Failed status or a waiting timeout, over the base abort's reason.
        with self.kv_mgr.failure_lock:
            reason = self.kv_mgr.failure_records.get(self.bootstrap_room)
        super().abort()
        if reason is not None:
            self.kv_mgr.record_failure(self.bootstrap_room, reason)

    def _send_abort_notification(self, *, force_arm: bool = False) -> None:
        # The base abort() and waiting-timeout paths call this after recording
        # the failure; UniFlow prefill expects its own GUARD-framed ABORT with
        # the reason and this rank's address for the deferred-release ack.
        # Arm drain-ack accounting before the ABORT goes out, under the same
        # condition as the base notification, so a racing ack is counted.
        if self.kv_mgr.enable_deferred_decode_kv_release and (
            force_arm or self.init_time is not None
        ):
            self._abort_generation = self.kv_mgr.register_deferred_abort_room(
                self.bootstrap_room
            )
        with self.kv_mgr.failure_lock:
            reason = self.kv_mgr.failure_records.get(
                self.bootstrap_room, DEFAULT_ABORT_REASON
            )
        for bootstrap_info in self.bootstrap_infos:
            try:
                sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
                with lock:
                    sock.send_multipart(
                        [
                            GUARD,
                            ABORT_MSG,
                            str(self.bootstrap_room).encode("ascii"),
                            reason.encode("utf-8"),
                            self.kv_mgr.local_ip.encode("ascii"),
                            str(self.kv_mgr.rank_port).encode("ascii"),
                        ]
                        + (
                            []
                            if self._abort_generation is None
                            else [str(self._abort_generation).encode("ascii")]
                        )
                    )
            except Exception:
                # Best-effort per peer, like the base notification.
                logger.exception(
                    "Failed to notify UniFlow prefill of abort for room=%s",
                    self.bootstrap_room,
                )

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: int | None = None,
        state_indices: list | None = None,
        decode_prefix_len: int | None = None,
        destination: KVTransferDestination = KVTransferDestination.DEVICE,
        device_kv_indices: npt.NDArray[np.int32] | None = None,
    ):
        if not self.bootstrap_infos:
            # No prefill peer: fail now with a reason instead of at the waiting
            # timeout.
            self.kv_mgr.record_failure(
                self.bootstrap_room, "UniFlow: no prefill peer (empty bootstrap_infos)"
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            self.conclude_state = KVPoll.Failed
            return
        super().send_metadata(
            kv_indices,
            aux_index,
            state_indices,
            decode_prefix_len,
            destination,
            device_kv_indices,
        )


class UniflowKVBootstrapServer(CommonKVBootstrapServer):
    pass
