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
the room before the ABORT has returned, and withholds the ack if a failed put's
connection did not shut down cleanly. A decode transfer queue that honors the
flag holds the aborted request's pages until that ack or its deferred-release
timeout; one that does not releases them on Failed. A put that fails or times
out shuts its connection down before prefill reports Failed or acks, because
decode releases the pages as soon as it sees either; only the RDMA tier's
shutdown waits for in-flight puts (see DESIGN.md).

Scope is intentionally narrow: CP=1, PP=1, DCP=1, no hisparse or custom memory
pool, one-to-one prefill/decode rank routing (MLA dummy ranks included), and
mamba, SWA, or DSA state.
"""

from __future__ import annotations

import base64
import functools
import json
import logging
import queue
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from typing import Any, NoReturn

import msgspec
import numpy as np
import numpy.typing as npt
import zmq

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
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
from sglang.srt.disaggregation.utils import DisaggregationMode
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
# One per prefill connection, reconnects after a failed put included. Decode
# cannot tell a dead connection from a live one, so past the bound the oldest
# is dropped, which closes it.
MAX_ACCEPTED_CONNECTIONS = 1024
# A late abort for a cleared room is never consumed, so the oldest are evicted.
MAX_PENDING_ABORTS = 4096
# A room whose shutdown after a failed put raised may never be aborted,
# so the oldest are evicted.
MAX_FAILED_SHUTDOWN_ROOMS = 4096
DEFAULT_ABORT_REASON = "UniFlow request aborted"
UNKNOWN_FAILURE_REASON = "Failed due to an unknown reason from another rank"
SUPPORTED_STATE_TYPES = frozenset({StateType.MAMBA, StateType.SWA, StateType.DSA})
# Paged state that tolerates an index-count drift, as in the Mooncake backend.
TRUNCATABLE_STATE_TYPES = frozenset({StateType.SWA, StateType.DSA})


class UniflowPutError(RuntimeError):
    """A put() that failed or timed out; its writes may still be in flight."""


def _uniflow_value(result: Any, action: str) -> Any:
    if not result.has_value():
        error = result.error() if result.has_error() else "no value"
        raise RuntimeError(f"UniFlow {action} failed: {error}")
    return result.value()


def _encode_export_id(export_id: bytes) -> str:
    return base64.b64encode(export_id).decode("ascii")


def _decode_export_id(export_id: str) -> bytes:
    return base64.b64decode(export_id.encode("ascii"))


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


class DecodePeer(msgspec.Struct, frozen=True, kw_only=True):
    connection: Any
    remote_kv_segments: list[Any]
    remote_aux_segments: list[Any]
    remote_state_segments: list[list[Any]]


class UniflowKVManager(CommonKVManager):
    kv_status_msg_tag = GUARD
    kv_status_msg_carries_reason = True
    # The ABORT handler queues the ack behind every put already queued for the room.
    supports_deferred_decode_kv_release = True

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

        try:
            from uniflow._core import (
                MemoryType,
                Segment,
                TransferRequest,
                UniflowAgent,
                UniflowAgentConfig,
            )
        except ImportError as e:
            raise ImportError(
                "--disaggregation-transfer-backend uniflow requires the uniflow "
                "package with its compiled _core extension; importing "
                f"uniflow._core failed: {e}"
            ) from e

        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

        self.memory_type_enum = MemoryType
        self.segment_cls = Segment
        self.transfer_request_cls = TransferRequest

        config = UniflowAgentConfig(
            device_id=self.kv_args.gpu_id,
            name=f"sglang_uniflow_{disaggregation_mode.value}_{self.kv_args.engine_rank}",
            # UniFlow splits host:port on the last colon and rejects brackets,
            # so an IPv6 host stays unbracketed.
            listen_address=f"{self.local_ip}:0",
        )
        self.agent = UniflowAgent(config)
        self.agent_id = _uniflow_value(self.agent.get_unique_id(), "get unique id")

        self.registered_kv_segments: list[Any] = []
        self.registered_aux_segments: list[Any] = []
        self.registered_state_segments: list[list[Any]] = []
        self.kv_export_ids: list[str] = []
        self.aux_export_ids: list[str] = []
        self.state_export_ids: list[list[str]] = []
        # Keyed by decode agent_id. A peer is dropped after a failed put and
        # reconnected on next use; peers of departed decode instances are not
        # reclaimed.
        self.decode_peer_infos: dict[str, DecodePeerInfo] = {}
        self.decode_peers: dict[str, DecodePeer] = {}
        self.decode_peers_lock = threading.Lock()
        # Re-entrant (the default RLock) so helpers called under it can
        # re-acquire it.
        self.condition = threading.Condition()
        # Decode aborts (room -> reason) for rooms with no status. The sender's
        # __init__ consumes early ones; ones arriving after clear() age out.
        self.pending_aborts: OrderedDict[int, str] = OrderedDict()
        # Rooms whose shutdown after a failed put raised;
        # their ABORT_ACK is withheld. Only the transfer worker touches it.
        self._failed_shutdown_rooms: OrderedDict[int, None] = OrderedDict()
        self.transfer_timeout_ms = transfer_timeout_ms
        self._accepted_connections: deque[Any] = deque(maxlen=MAX_ACCEPTED_CONNECTIONS)
        self._accepted_connections_lock = threading.Lock()
        # Prefill puts run on one FIFO worker so the scheduler never blocks on
        # completion. One worker keeps per-room chunk order, confines
        # Connection use to a single thread, and runs an ABORT_ACK only after
        # every chunk queued before it.
        self._transfer_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        if self.is_mla_backend:
            # One-to-one routing gives each prefill rank one decode destination.
            self._kv_replica_factor = 1

        self._register_segments()
        self._start_accept_thread()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._start_transfer_worker()
            self._start_message_thread(self._handle_prefill_message, "prefill")
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self._start_message_thread(self._handle_decode_message, "decode")
            # Fails in-flight rooms of a prefill whose bootstrap server stops
            # answering /health, as the other backends do.
            self._start_heartbeat_checker_thread()

    def _register_segment(
        self, ptr: int, length: int, memory_type: Any, device_id: int, kind: str
    ) -> tuple[Any, str]:
        registered = _uniflow_value(
            self.agent.register_segment(
                self.segment_cls(ptr, length, memory_type, device_id)
            ),
            f"register {kind} segment",
        )
        export_id = _uniflow_value(registered.export_id(), f"export {kind} segment")
        return registered, _encode_export_id(export_id)

    def _register_segments(self) -> None:
        # No unwind on failure: a raise here aborts startup, and process exit
        # releases the registrations.
        if len(self.kv_args.kv_data_ptrs) != len(self.kv_args.kv_data_lens):
            raise ValueError(
                "UniFlow KV data ptr/len count mismatch: "
                f"ptrs={len(self.kv_args.kv_data_ptrs)} "
                f"lens={len(self.kv_args.kv_data_lens)}"
            )
        for ptr, length in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens, strict=True
        ):
            registered, export_id = self._register_segment(
                ptr, length, self.memory_type_enum.VRAM, self.kv_args.gpu_id, "KV"
            )
            self.registered_kv_segments.append(registered)
            self.kv_export_ids.append(export_id)

        if len(self.kv_args.aux_data_ptrs) != len(self.kv_args.aux_data_lens):
            raise ValueError(
                "UniFlow aux data ptr/len count mismatch: "
                f"ptrs={len(self.kv_args.aux_data_ptrs)} "
                f"lens={len(self.kv_args.aux_data_lens)}"
            )
        for ptr, length in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens, strict=True
        ):
            # Aux buffers are host memory; -1 is the UniFlow device id for DRAM.
            registered, export_id = self._register_segment(
                ptr, length, self.memory_type_enum.DRAM, -1, "aux"
            )
            self.registered_aux_segments.append(registered)
            self.aux_export_ids.append(export_id)

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
            component_segments = []
            component_export_ids = []
            for ptr, length in zip(ptrs, lens, strict=True):
                registered, export_id = self._register_segment(
                    ptr,
                    length,
                    self.memory_type_enum.VRAM,
                    self.kv_args.gpu_id,
                    "state",
                )
                component_segments.append(registered)
                component_export_ids.append(export_id)
            self.registered_state_segments.append(component_segments)
            self.state_export_ids.append(component_export_ids)

    def _start_accept_thread(self) -> None:
        def accept_loop() -> None:
            last_error: str | None = None
            while True:
                try:
                    result = self.agent.accept()
                    if result.has_error():
                        # Warn once per distinct error; a broken listener
                        # repeats the same one every retry.
                        error = str(result.error())
                        level = (
                            logging.DEBUG if error == last_error else logging.WARNING
                        )
                        logger.log(level, "UniFlow accept failed: %s", error)
                        last_error = error
                        time.sleep(0.1)
                        continue
                    last_error = None
                    # Keep accepted UniFlow connections alive, but bound retention.
                    with self._accepted_connections_lock:
                        self._accepted_connections.append(result.value())
                except Exception:
                    # Keep the accept thread alive across unexpected errors.
                    logger.exception("UniFlow accept loop iteration failed")
                    time.sleep(0.1)

        threading.Thread(target=accept_loop, daemon=True).start()

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
                        self._send_abort_ack_unless_shutdown_failed, room, target
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

    def _get_or_connect_decode_peer(self, peer_info: DecodePeerInfo) -> DecodePeer:
        with self.decode_peers_lock:
            cached = self.decode_peers.get(peer_info.agent_id)
            if cached is not None:
                return cached

            if len(peer_info.kv_export_ids) != len(self.registered_kv_segments):
                raise ValueError(
                    "UniFlow KV segment count mismatch: "
                    f"local={len(self.registered_kv_segments)} "
                    f"remote={len(peer_info.kv_export_ids)}"
                )
            if len(peer_info.aux_export_ids) != len(self.registered_aux_segments):
                raise ValueError(
                    "UniFlow aux segment count mismatch: "
                    f"local={len(self.registered_aux_segments)} "
                    f"remote={len(peer_info.aux_export_ids)}"
                )
            if len(peer_info.state_export_ids) != len(self.registered_state_segments):
                raise ValueError(
                    "UniFlow state component count mismatch: "
                    f"local={len(self.registered_state_segments)} "
                    f"remote={len(peer_info.state_export_ids)}"
                )

            remote_state_segments = []
            for local_component, remote_export_ids in zip(
                self.registered_state_segments, peer_info.state_export_ids, strict=True
            ):
                if len(local_component) != len(remote_export_ids):
                    raise ValueError(
                        "UniFlow state segment count mismatch within component: "
                        f"local={len(local_component)} remote={len(remote_export_ids)}"
                    )
                remote_state_segments.append(
                    self._import_segments(remote_export_ids, "state")
                )

            connection = _uniflow_value(
                self.agent.connect(peer_info.agent_id), "connect decode peer"
            )
            peer = DecodePeer(
                connection=connection,
                remote_kv_segments=self._import_segments(peer_info.kv_export_ids, "KV"),
                remote_aux_segments=self._import_segments(
                    peer_info.aux_export_ids, "aux"
                ),
                remote_state_segments=remote_state_segments,
            )
            self.decode_peers[peer_info.agent_id] = peer
            return peer

    def _import_segments(self, export_ids: list[str], kind: str) -> list[Any]:
        return [
            _uniflow_value(
                self.agent.import_segment(_decode_export_id(export_id)),
                f"import {kind} segment",
            )
            for export_id in export_ids
        ]

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
            peer = self.decode_peers.pop(agent_id, None)
        if peer is None:
            return True
        try:
            peer.connection.shutdown()
        except Exception:
            logger.exception("UniFlow shutdown of decode peer %s failed", agent_id)
            return False
        return True

    def _send_abort_ack_unless_shutdown_failed(
        self, room: int, target: AckTarget
    ) -> None:
        # Runs on the transfer worker, after every put queued before the abort.
        if room in self._failed_shutdown_rooms:
            logger.warning(
                "Withholding UniFlow ABORT_ACK for room=%s: shutdown after its "
                "failed put raised",
                room,
            )
            return
        self._send_abort_ack(room, target)

    def _is_room_stopped(self, room: int) -> bool:
        """Whether decode or the sender aborted the room, or the sender cleared it."""
        with self.condition:
            return self.request_status.get(room) in (None, KVPoll.Failed)

    def raise_recorded_failure(self, room: int) -> NoReturn:
        with self.failure_lock:
            reason = self.failure_records.pop(room, None)
        raise KVTransferError(
            room,
            reason or UNKNOWN_FAILURE_REASON,
            is_from_another_rank=reason is None,
        )

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
        *,
        index_slice: slice,
        is_last: bool,
        aux_index: int | None,
        state_indices: list | None,
        wait_event: Any | None,
    ) -> None:
        self._transfer_queue.put(
            functools.partial(
                self._do_transfer_request,
                room,
                kv_indices,
                index_slice=index_slice,
                is_last=is_last,
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
                if self.registered_aux_segments:
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
            # Shut down before reporting: decode frees the pages on Failed or
            # ABORT_ACK. If shutdown raised, send neither; decode's timeouts free them.
            is_shut_down = self._close_decode_peer(info.agent_id)
            if not is_shut_down:
                self._failed_shutdown_rooms[room] = None
                while len(self._failed_shutdown_rooms) > MAX_FAILED_SHUTDOWN_ROOMS:
                    self._failed_shutdown_rooms.popitem(last=False)
            self.mark_transfer_failure(info, e, should_notify_decode=is_shut_down)
            return
        except Exception as e:
            self.mark_transfer_failure(info, e)
            return

        if is_complete:
            self.mark_transfer_success(info)

    def _put_kv(
        self,
        peer: DecodePeer,
        src_indices: npt.NDArray[np.int32],
        dst_indices: npt.NDArray[np.int32],
    ) -> None:
        src_blocks, dst_blocks = group_concurrent_contiguous(src_indices, dst_indices)
        requests = []
        for local_segment, remote_segment, item_len in zip(
            self.registered_kv_segments,
            peer.remote_kv_segments,
            self.kv_args.kv_item_lens,
            strict=True,
        ):
            for src_block, dst_block in zip(src_blocks, dst_blocks, strict=True):
                length = int(item_len) * len(src_block)
                requests.append(
                    self.transfer_request_cls(
                        local_segment.span(int(src_block[0]) * item_len, length),
                        remote_segment.span(int(dst_block[0]) * item_len, length),
                    )
                )
        self._put(peer, requests, "KV put")

    def _put_aux(self, peer: DecodePeer, src_index: int, dst_index: int) -> None:
        requests = []
        for local_segment, remote_segment, item_len in zip(
            self.registered_aux_segments,
            peer.remote_aux_segments,
            self.kv_args.aux_item_lens,
            strict=True,
        ):
            requests.append(
                self.transfer_request_cls(
                    local_segment.span(src_index * item_len, item_len),
                    remote_segment.span(dst_index * item_len, item_len),
                )
            )
        self._put(peer, requests, "aux put")

    def _put_state(
        self,
        peer: DecodePeer,
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

        requests = []
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
            for local_segment, remote_segment, item_len in zip(
                self.registered_state_segments[component_id],
                peer.remote_state_segments[component_id],
                self.kv_args.state_item_lens[component_id],
                strict=True,
            ):
                for src_index, dst_index in zip(
                    src_indices[:count], dst_indices[:count], strict=True
                ):
                    requests.append(
                        self.transfer_request_cls(
                            local_segment.span(src_index * item_len, item_len),
                            remote_segment.span(dst_index * item_len, item_len),
                        )
                    )
        self._put(peer, requests, "state put")

    def _put(self, peer: DecodePeer, requests: list[Any], action: str) -> None:
        # Returns only after the batch completes, so a later raise leaves no
        # write in flight.
        if not requests:
            return
        try:
            future = peer.connection.put(requests=requests)
            if not future.wait_for(timeout_ms=self.transfer_timeout_ms):
                raise UniflowPutError(f"UniFlow {action} timed out")
            result = future.get()
            if result.has_error():
                raise UniflowPutError(f"UniFlow {action} failed: {result.error()}")
        except UniflowPutError:
            raise
        except Exception as e:
            # A binding error may leave the batch on the wire;
            # the caller shuts the connection down.
            raise UniflowPutError(f"UniFlow {action} failed: {e}") from e


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
        self._transfer_start_time: float | None = None
        # Set by the prefill scheduler before an early send under overlap.
        self._early_send_wait_event: Any | None = None
        # Starts the bootstrap timeout: a room whose decode never sends
        # METADATA must fail rather than hold its prefill KV indefinitely.
        self.init_time = time.time()
        with mgr.condition:
            reason = mgr.pending_aborts.pop(bootstrap_room, None)
            if reason is not None:
                mgr.record_failure(bootstrap_room, reason)
                mgr.update_status(bootstrap_room, KVPoll.Failed)

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: list | None = None,
        num_kv_tokens: int | None = None,
    ):
        wait_event = self._early_send_wait_event
        self._early_send_wait_event = None
        kv_indices, index_slice, is_last, should_skip = self._prepare_send_indices(
            kv_indices, state_indices
        )
        if should_skip:
            return

        if self._transfer_start_time is None and (
            len(kv_indices) > 0 or state_indices is not None
        ):
            self._transfer_start_time = time.perf_counter()

        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            np.ascontiguousarray(kv_indices, dtype=np.int32),
            index_slice=index_slice,
            is_last=is_last,
            aux_index=self.aux_index,
            state_indices=state_indices,
            wait_event=wait_event,
        )
        self._record_transfer_indices(kv_indices, state_indices)

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            if status == KVPoll.Success and self._transfer_start_time is not None:
                self._transfer_metric.transfer_latency_s = (
                    time.perf_counter() - self._transfer_start_time
                )
        elif status == KVPoll.Bootstrapping:
            timeout_result = self._check_bootstrap_timeout()
            if timeout_result is not None:
                return timeout_result
        return status

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

    def failure_exception(self) -> NoReturn:
        # Prefill clears a sender only on success, so a failed room drops its
        # local state here, as the Mooncake backend does.
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed
        self.clear()
        self.kv_mgr.raise_recorded_failure(self.bootstrap_room)


class UniflowKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: UniflowKVManager,
        bootstrap_addr: str,
        bootstrap_room: int | None = None,
    ):
        self.started_transfer = False
        super().__init__(mgr, bootstrap_addr, bootstrap_room)

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

    def _register_kv_args(self) -> bool:
        payload = json.dumps(
            {
                "agent_id": self.kv_mgr.agent_id,
                "kv_export_ids": self.kv_mgr.kv_export_ids,
                "aux_export_ids": self.kv_mgr.aux_export_ids,
                "state_export_ids": self.kv_mgr.state_export_ids,
            }
        ).encode("utf-8")
        for bootstrap_info in self.bootstrap_infos:
            try:
                sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
                with lock:
                    sock.send_multipart([GUARD, REGISTER_MSG, payload])
            except zmq.ZMQError:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    "UniFlow register to prefill "
                    f"{bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')} "
                    "failed",
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return False
        return True

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: int | None = None,
        state_indices: list | None = None,
        decode_prefix_len: int | None = None,
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

        packed_state_indices = pack_int_lists(
            flatten_optional_indices(state_indices), "i"
        )
        aux_frame = str(NO_AUX_INDEX if aux_index is None else aux_index).encode(
            "ascii"
        )
        for bootstrap_info in self.bootstrap_infos:
            is_dummy = bootstrap_info["is_dummy"]
            try:
                sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
                with lock:
                    sock.send_multipart(
                        [
                            GUARD,
                            METADATA_MSG,
                            str(self.bootstrap_room).encode("ascii"),
                            self.kv_mgr.local_ip.encode("ascii"),
                            str(self.kv_mgr.rank_port).encode("ascii"),
                            self.kv_mgr.agent_id.encode("ascii"),
                            kv_indices.tobytes() if not is_dummy else b"",
                            aux_frame if not is_dummy else b"",
                            packed_state_indices if not is_dummy else b"",
                            (
                                str(decode_prefix_len).encode("ascii")
                                if decode_prefix_len is not None
                                else b""
                            ),
                        ]
                    )
            except zmq.ZMQError:
                self.invalidate_cached_bootstrap_infos()
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    "UniFlow send_metadata to prefill "
                    f"{bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')} "
                    "failed",
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return

        self.started_transfer = True
        # Waiting timeout starts here, not in init().
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status

        if status == KVPoll.WaitingForInput:
            timeout_result = self._check_waiting_timeout()
            if timeout_result is not None:
                self.conclude_state = timeout_result
                return timeout_result

        if not self.started_transfer:
            return status

        # Prefill reports only the terminal status, so decode stays
        # WaitingForInput until then.
        return KVPoll.WaitingForInput

    def failure_exception(self) -> NoReturn:
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed
        self.clear()
        self.kv_mgr.raise_recorded_failure(self.bootstrap_room)


class UniflowKVBootstrapServer(CommonKVBootstrapServer):
    pass
