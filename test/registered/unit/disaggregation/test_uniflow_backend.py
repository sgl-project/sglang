"""UniFlow P/D KV-transfer backend tests.

Every test runs on CPU in one process. Unit tests drive the connector state
machine on bare objects; the end-to-end tests run real prefill and decode
managers over the ZMQ side channel with a fake uniflow._core whose put() copies
host memory.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import logging
import queue
import re
import socket
import struct
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Iterator
from types import ModuleType, SimpleNamespace
from typing import Any

import msgspec
import numpy as np
import pytest
import torch
import zmq

from sglang.srt.arg_groups.choices import DISAGG_TRANSFER_BACKEND_CHOICES
from sglang.srt.disaggregation.base.conn import (
    KVArgs,
    KVPoll,
    KVTransferMetric,
    StateType,
)
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.uniflow import conn as uniflow_conn
from sglang.srt.disaggregation.uniflow.conn import (
    ABORT_MSG,
    DEFAULT_ABORT_REASON,
    GUARD,
    METADATA_MSG,
    NO_AUX_INDEX,
    REGISTER_MSG,
    UNKNOWN_FAILURE_REASON,
    DecodePeer,
    DecodePeerInfo,
    KVTransferError,
    TransferInfo,
    UniflowKVBootstrapServer,
    UniflowKVManager,
    UniflowKVReceiver,
    UniflowKVSender,
    UniflowPutError,
    flatten_optional_indices,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

POLL_INTERVAL = 0.02
INIT_HANDSHAKE_TIMEOUT = 10.0

NUM_LAYERS = 2
NUM_KV_HEADS = 4
HEAD_DIM = 64
PAGE_SIZE = 16
NUM_PAGES = 8
AUX_BYTES_PER_SLOT = 64
NUM_AUX_SLOTS = 4
STATE_BYTES = 512
NUM_STATE_SLOTS = 4

LOCALHOST = "127.0.0.1"
DECODE_AGENT = "decode-agent"
DECODE_RANK_PORT = 17000
PREFILL_RANK_PORT = 18000
PREFILL_PEER = {"rank_ip": LOCALHOST, "rank_port": PREFILL_RANK_PORT, "is_dummy": False}
SINGLE_RANK_PARALLEL = dict(
    attn_tp_size=1,
    attn_tp_rank=0,
    attn_cp_size=1,
    attn_cp_rank=0,
    attn_dcp_size=1,
    attn_dcp_rank=0,
    attn_dp_size=1,
    attn_dp_rank=0,
    pp_group=None,
)


def _ascii(value: object) -> bytes:
    return str(value).encode("ascii")


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((LOCALHOST, 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_INTERVAL)
    return True


def _poll_until(
    fetch: Callable[[], int], targets: Iterable[int], timeout: float
) -> int:
    target_set = set(int(target) for target in targets)
    deadline = time.monotonic() + timeout
    last = -1
    while time.monotonic() < deadline:
        last = int(fetch())
        if last in target_set:
            return last
        time.sleep(POLL_INTERVAL)
    return last


def _peer_info() -> DecodePeerInfo:
    return DecodePeerInfo(
        agent_id=DECODE_AGENT,
        kv_export_ids=["kv0"],
        aux_export_ids=["aux0"],
        state_export_ids=[["st0"]],
    )


def _transfer_info(room: int, **overrides: Any) -> TransferInfo:
    fields: dict[str, Any] = dict(
        room=room,
        endpoint=LOCALHOST,
        dst_port=DECODE_RANK_PORT + 1,
        agent_id=DECODE_AGENT,
        dst_kv_indices=np.array([4, 5, 6], dtype=np.int32),
        dst_aux_index=1,
        dst_state_indices=[],
        decode_prefix_len=None,
        is_dummy=False,
    )
    fields.update(overrides)
    return TransferInfo(**fields)


def _bare_manager(**attrs: Any) -> UniflowKVManager:
    """A manager holding only the state the tested paths read: no sockets or threads."""
    mgr = object.__new__(UniflowKVManager)
    fields: dict[str, Any] = dict(
        condition=threading.Condition(),
        failure_lock=threading.Lock(),
        request_status={},
        failure_records={},
        transfer_infos={},
        req_to_decode_prefix_len={},
        decode_peer_infos={},
        decode_peers={},
        decode_peers_lock=threading.Lock(),
        transfer_timeout_ms=1000,
        pending_aborts=OrderedDict(),
        _failed_shutdown_rooms=OrderedDict(),
        registered_aux_segments=[],
        state_types=[],
        enable_deferred_decode_kv_release=False,
        _transfer_queue=queue.Queue(),
        _deferred_abort_ack_tracker={},
        bootstrap_timeout=0.05,
        waiting_timeout=0.05,
        is_dummy_cp_rank=False,
        enable_all_cp_ranks_for_transfer=False,
        local_ip=LOCALHOST,
        rank_port=DECODE_RANK_PORT,
        agent_id=DECODE_AGENT,
        kv_export_ids=["kv0"],
        aux_export_ids=["aux0"],
        state_export_ids=[["st0"]],
        connection_lock=threading.Lock(),
        connection_pool={},
        required_prefill_response_num_table={},
        prefill_response_tracker=defaultdict(set),
        addr_to_rooms_tracker=defaultdict(set),
        enable_staging=False,
        _staging_handler=None,
        attn_tp_rank=0,
        attn_cp_rank=0,
        attn_cp_size=1,
        pp_rank=0,
        pp_size=1,
    )
    fields.update(attrs)
    vars(mgr).update(fields)
    return mgr


def _ready_room(
    mgr: UniflowKVManager,
    room: int,
    status: int = KVPoll.WaitingForInput,
    **info_overrides: Any,
) -> None:
    """Put a room in the state METADATA and REGISTER leave it in."""
    mgr.request_status[room] = status
    mgr.transfer_infos[room] = _transfer_info(room, **info_overrides)
    mgr.decode_peer_infos[DECODE_AGENT] = _peer_info()


def _record_transfer_calls(
    mgr: UniflowKVManager,
    *,
    notify_error: Exception | None = None,
    on_connect: Callable[[], None] | None = None,
) -> SimpleNamespace:
    """Replace the peer connection, puts, and status socket with recorders.

    notified collects the KVPoll of each status frame; status_frames the frames
    and their decode endpoints.
    """
    calls = SimpleNamespace(puts=[], notified=[], status_frames=[])

    def connect(peer_info: DecodePeerInfo) -> str:
        if on_connect is not None:
            on_connect()
        return "peer"

    def send_status(endpoint: str, parts: list[bytes], is_ipv6: bool = False) -> None:
        calls.status_frames.append((endpoint, list(parts)))
        calls.notified.append(int(parts[2]))
        if notify_error is not None:
            raise notify_error

    mgr._get_or_connect_decode_peer = connect
    mgr._put_kv = lambda peer, src, dst: calls.puts.append(("kv", list(src), list(dst)))
    mgr._put_aux = lambda peer, src, dst: calls.puts.append(("aux", src, dst))
    mgr._put_state = lambda peer, src, dst: calls.puts.append(("state", src, dst))
    mgr._send_multipart_locked = send_status
    return calls


def _run_queued_tasks(mgr: UniflowKVManager) -> None:
    while not mgr._transfer_queue.empty():
        mgr._transfer_queue.get_nowait()()


def _bare_sender(
    mgr: UniflowKVManager,
    room: int,
    num_kv_indices: int = 1,
    aux_index: int | None = 0,
) -> UniflowKVSender:
    sender = object.__new__(UniflowKVSender)
    vars(sender).update(
        kv_mgr=mgr,
        bootstrap_room=room,
        num_kv_indices=num_kv_indices,
        aux_index=aux_index,
        conclude_state=None,
        init_time=time.time(),
        curr_idx=0,
        _early_send_wait_event=None,
        _transfer_start_time=None,
        _transfer_metric=KVTransferMetric(),
        _transfer_num_kv_indices=0,
        _transfer_num_state_indices=0,
    )
    return sender


class _FakeSocket:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.sent: list[list[bytes]] = []

    def send_multipart(self, parts: list[bytes]) -> None:
        if self.error is not None:
            raise self.error
        self.sent.append(list(parts))


def _bare_receiver(
    mgr: UniflowKVManager,
    room: int,
    bootstrap_infos: list[dict[str, Any]] | None,
    sock: _FakeSocket | None = None,
) -> UniflowKVReceiver:
    receiver = object.__new__(UniflowKVReceiver)
    vars(receiver).update(
        bootstrap_room=room,
        bootstrap_addr=f"{LOCALHOST}:8998",
        kv_mgr=mgr,
        conclude_state=None,
        init_time=None,
        abort_notified=False,
        _connection_pool_entries={},
        started_transfer=False,
        required_dst_info_num=1,
        required_prefill_response_num=1,
        bootstrap_infos=bootstrap_infos,
    )
    receiver._connect_to_bootstrap_server = lambda info: (sock, threading.Lock())
    return receiver


def test_flatten_optional_indices_handles_nested_state_shapes() -> None:
    assert flatten_optional_indices(None) == []
    assert flatten_optional_indices([]) == []
    assert flatten_optional_indices([np.array([1, 2], dtype=np.int64), None]) == [
        [1, 2],
        [],
    ]
    assert flatten_optional_indices([[np.array([3, 4], dtype=np.int32)], (5, 6)]) == [
        [3, 4],
        [5, 6],
    ]


def test_decode_prefix_len_is_parsed_and_consumed() -> None:
    msg = [
        GUARD,
        METADATA_MSG,
        b"123",
        b"127.0.0.1",
        b"456",
        b"agent",
        np.array([2, 3], dtype=np.int32).tobytes(),
        b"0",
        b"",
        b"7",
    ]
    assert TransferInfo.from_zmq(msg).decode_prefix_len == 7

    sender = _bare_sender(_bare_manager(req_to_decode_prefix_len={123: 7}), 123)
    assert sender.pop_decode_prefix_len() == 7
    assert sender.pop_decode_prefix_len() == 0


def test_uniflow_is_a_native_transfer_backend() -> None:
    backend = TransferBackend("uniflow")
    assert backend is TransferBackend.UNIFLOW
    expected = {
        KVClassType.KVARGS: KVArgs,
        KVClassType.MANAGER: UniflowKVManager,
        KVClassType.SENDER: UniflowKVSender,
        KVClassType.RECEIVER: UniflowKVReceiver,
        KVClassType.BOOTSTRAP_SERVER: UniflowKVBootstrapServer,
    }
    for class_type, cls in expected.items():
        assert get_kv_class(backend, class_type) is cls
    assert DISAGG_TRANSFER_BACKEND_CHOICES.count("uniflow") == 1


def test_manager_opts_into_deferred_decode_kv_release() -> None:
    # CommonKVManager enables deferred release only for backends that opt in.
    assert UniflowKVManager.supports_deferred_decode_kv_release


def test_success_status_survives_decode_notification_failure() -> None:
    room = 0xC0FFEE_201
    mgr = _bare_manager(registered_aux_segments=["aux-segment"])
    _ready_room(mgr, room)
    calls = _record_transfer_calls(mgr, notify_error=RuntimeError("decode unreachable"))

    mgr._do_transfer_request(
        room,
        np.array([1, 2, 3], dtype=np.int32),
        index_slice=slice(0, 3),
        is_last=True,
        aux_index=0,
        state_indices=None,
        wait_event=None,
    )

    assert mgr.request_status[room] == KVPoll.Success
    assert room not in mgr.transfer_infos
    assert calls.puts == [
        ("kv", [1, 2, 3], [4, 5, 6]),
        ("aux", 0, 1),
        ("state", None, []),
    ]
    assert calls.status_frames == [
        (
            f"tcp://{LOCALHOST}:{DECODE_RANK_PORT + 1}",
            [GUARD, _ascii(room), _ascii(KVPoll.Success), b"0", b""],
        )
    ]


@pytest.mark.parametrize("status", [KVPoll.Failed, None], ids=["failed", "cleared"])
def test_success_does_not_resurrect_concluded_room(status: int | None) -> None:
    room = 0xC0FFEE_202
    mgr = _bare_manager()
    if status is not None:
        mgr.request_status[room] = status
        mgr.failure_records[room] = "decode gave up"
    calls = _record_transfer_calls(mgr)

    mgr.mark_transfer_success(_transfer_info(room))

    assert mgr.request_status.get(room) == status
    if status is None:
        assert calls.status_frames == []
    else:
        # A failed room still tells decode, with the recorded reason.
        assert [frame for _, frame in calls.status_frames] == [
            [GUARD, _ascii(room), _ascii(KVPoll.Failed), b"0", b"decode gave up"]
        ]


def test_abort_while_connecting_skips_puts() -> None:
    room = 0xC0FFEE_203
    mgr = _bare_manager()
    _ready_room(mgr, room)
    calls = _record_transfer_calls(
        mgr, on_connect=lambda: mgr.update_status(room, KVPoll.Failed)
    )

    mgr._do_transfer_request(
        room,
        np.array([1], dtype=np.int32),
        index_slice=slice(0, 1),
        is_last=True,
        aux_index=0,
        state_indices=None,
        wait_event=None,
    )

    assert calls.puts == []
    assert calls.notified == []
    assert mgr.request_status[room] == KVPoll.Failed


@pytest.mark.parametrize(
    "status",
    [None, KVPoll.Failed, KVPoll.Success],
    ids=["cleared", "failed", "success"],
)
def test_queued_chunk_of_concluded_room_is_skipped(status: int | None) -> None:
    room = 0xC0FFEE_204
    mgr = _bare_manager()
    if status is not None:
        mgr.request_status[room] = status
    calls = _record_transfer_calls(mgr)
    mgr.wait_for_transfer_info = lambda _room: pytest.fail("waited on a concluded room")

    mgr._do_transfer_request(
        room,
        np.array([1], dtype=np.int32),
        index_slice=slice(0, 1),
        is_last=True,
        aux_index=0,
        state_indices=None,
        wait_event=None,
    )

    assert calls.puts == []
    assert calls.notified == []
    assert mgr.request_status.get(room) == status


@pytest.mark.parametrize(
    "aux_index, dst_aux_index",
    [(None, 1), (0, NO_AUX_INDEX)],
    ids=["no-source-aux", "no-destination-aux"],
)
def test_last_chunk_without_aux_fails_room(
    aux_index: int | None, dst_aux_index: int
) -> None:
    room = 0xC0FFEE_205
    mgr = _bare_manager(registered_aux_segments=["aux-segment"])
    _ready_room(mgr, room, dst_aux_index=dst_aux_index)
    calls = _record_transfer_calls(mgr)

    mgr._do_transfer_request(
        room,
        np.array([1, 2, 3], dtype=np.int32),
        index_slice=slice(0, 3),
        is_last=True,
        aux_index=aux_index,
        state_indices=None,
        wait_event=None,
    )

    assert mgr.request_status[room] == KVPoll.Failed
    assert mgr.failure_records[room].startswith("UniFlow transfer failed:")
    assert "requires aux_index" in mgr.failure_records[room]
    assert room not in mgr.transfer_infos
    assert calls.notified == [KVPoll.Failed]


def test_metadata_timeout_fails_room_and_sender_reports_it() -> None:
    room = 0xC0FFEE_206
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.Bootstrapping
    calls = _record_transfer_calls(mgr)

    mgr._do_transfer_request(
        room,
        np.array([1], dtype=np.int32),
        index_slice=slice(0, 1),
        is_last=True,
        aux_index=0,
        state_indices=None,
        wait_event=None,
    )

    assert "did not arrive" in mgr.failure_records[room]
    assert mgr.request_status[room] == KVPoll.Failed
    assert calls.notified == []

    sender = _bare_sender(mgr, room)
    assert sender.poll() == KVPoll.Failed
    with pytest.raises(KVTransferError, match="did not arrive") as raised:
        sender.failure_exception()
    assert not raised.value.is_from_another_rank
    assert room not in mgr.request_status
    assert room not in mgr.failure_records


@pytest.mark.parametrize("cause", ["metadata-timeout", "decode-abort", "none"])
def test_sender_abort_keeps_recorded_failure_reason(cause: str) -> None:
    # A prefill scheduler may abort a failed sender before it reads the failure.
    room = 0xC0FFEE_306
    mgr = _bare_manager()
    if cause == "metadata-timeout":
        mgr.request_status[room] = KVPoll.Bootstrapping
        _record_transfer_calls(mgr)
        mgr._do_transfer_request(
            room,
            np.array([1], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last=True,
            aux_index=0,
            state_indices=None,
            wait_event=None,
        )
        reason = mgr.failure_records[room]
        assert "did not arrive" in reason
    elif cause == "decode-abort":
        _ready_room(mgr, room)
        mgr._handle_prefill_message(
            [GUARD, ABORT_MSG, _ascii(room), b"decode timed out"]
        )
        reason = "decode timed out"
    else:
        mgr.request_status[room] = KVPoll.WaitingForInput
        reason = "Aborted by AbortReq."
    sender = _bare_sender(mgr, room)

    sender.abort()

    assert sender.poll() == KVPoll.Failed
    with pytest.raises(KVTransferError, match=re.escape(reason)) as raised:
        sender.failure_exception()
    assert not raised.value.is_from_another_rank
    assert room not in mgr.request_status
    assert room not in mgr.failure_records


def test_wait_raises_recorded_abort_reason() -> None:
    room = 0xC0FFEE_207
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.Failed
    mgr.failure_records[room] = "decode gave up"

    with pytest.raises(KVTransferError, match="decode gave up"):
        mgr.wait_for_transfer_info(room)
    del mgr.failure_records[room]
    with pytest.raises(KVTransferError, match=DEFAULT_ABORT_REASON):
        mgr.wait_for_transfer_info(room)


def test_wait_timeout_for_cleared_room_records_nothing() -> None:
    room = 0xC0FFEE_208
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.Bootstrapping

    class ClearingCondition(threading.Condition):
        """Clears the room during the wait, as a concurrent sender timeout does."""

        def wait_for(
            self, predicate: Callable[[], bool], timeout: float | None = None
        ) -> bool:
            mgr.request_status.pop(room, None)
            return False

    mgr.condition = ClearingCondition()

    with pytest.raises(KVTransferError, match="concluded before its transfer started"):
        mgr.wait_for_transfer_info(room)
    assert room not in mgr.request_status
    assert not mgr.failure_records


@pytest.mark.parametrize(
    "extra, reason",
    [([b"decode timed out"], "decode timed out"), ([], DEFAULT_ABORT_REASON)],
    ids=["with-reason", "default-reason"],
)
def test_abort_fails_active_room(extra: list[bytes], reason: str) -> None:
    room = 0xC0FFEE_209
    mgr = _bare_manager()
    _ready_room(mgr, room)
    mgr.req_to_decode_prefix_len[room] = 3

    mgr._handle_prefill_message([GUARD, ABORT_MSG, _ascii(room), *extra])

    assert mgr.request_status[room] == KVPoll.Failed
    assert mgr.failure_records[room] == reason
    assert room not in mgr.transfer_infos
    assert room not in mgr.req_to_decode_prefix_len
    assert not mgr.pending_aborts


def test_abort_before_sender_exists_is_consumed_by_sender() -> None:
    room = 0xC0FFEE_210
    mgr = _bare_manager()

    mgr._handle_prefill_message([GUARD, ABORT_MSG, _ascii(room), b"early abort"])

    assert mgr.pending_aborts == {room: "early abort"}
    assert room not in mgr.request_status
    with get_parallel().override(dp_size=1):
        sender = UniflowKVSender(mgr, f"{LOCALHOST}:8998", room, [0], 0)
    assert not mgr.pending_aborts
    assert sender.poll() == KVPoll.Failed
    with pytest.raises(KVTransferError, match="early abort"):
        sender.failure_exception()


@pytest.mark.parametrize(
    "status", [KVPoll.Success, KVPoll.Failed], ids=["success", "failed"]
)
def test_late_abort_for_concluded_room_is_ignored_but_acked(status: int) -> None:
    room = 0xC0FFEE_211
    mgr = _bare_manager(enable_deferred_decode_kv_release=True)
    mgr.request_status[room] = status
    acks: list[tuple[str, int, int]] = []
    mgr._send_abort_ack = lambda ip, port, ack_room: acks.append((ip, port, ack_room))

    mgr._handle_prefill_message(
        [GUARD, ABORT_MSG, _ascii(room), b"late", _ascii(LOCALHOST), b"17001"]
    )

    assert mgr.request_status[room] == status
    assert room not in mgr.failure_records
    assert not mgr.pending_aborts
    _run_queued_tasks(mgr)
    assert acks == [(LOCALHOST, 17001, room)]


def test_pending_aborts_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uniflow_conn, "MAX_PENDING_ABORTS", 2)
    mgr = _bare_manager()

    for room in (1, 2, 3):
        mgr._handle_prefill_message([GUARD, ABORT_MSG, _ascii(room)])

    assert list(mgr.pending_aborts) == [2, 3]


def test_abort_ack_runs_after_chunks_queued_before_it() -> None:
    room = 0xC0FFEE_212
    mgr = _bare_manager(enable_deferred_decode_kv_release=True)
    mgr.request_status[room] = KVPoll.Transferring
    order: list[str] = []
    mgr._do_transfer_request = lambda *_args, **_kwargs: order.append("chunk")
    mgr._send_abort_ack = lambda *_args: order.append("ack")

    mgr.add_transfer_request(
        room,
        np.array([1], dtype=np.int32),
        index_slice=slice(0, 1),
        is_last=False,
        aux_index=0,
        state_indices=None,
        wait_event=None,
    )
    mgr._handle_prefill_message(
        [GUARD, ABORT_MSG, _ascii(room), b"abort", _ascii(LOCALHOST), b"17001"]
    )
    _run_queued_tasks(mgr)

    assert order == ["chunk", "ack"]


@pytest.mark.parametrize(
    "enabled, extra",
    [(False, [_ascii(LOCALHOST), b"17001"]), (True, [])],
    ids=["flag-off", "no-decode-address"],
)
def test_abort_ack_requires_flag_and_decode_address(
    enabled: bool, extra: list[bytes]
) -> None:
    room = 0xC0FFEE_213
    mgr = _bare_manager(enable_deferred_decode_kv_release=enabled)
    mgr.request_status[room] = KVPoll.Transferring

    mgr._handle_prefill_message([GUARD, ABORT_MSG, _ascii(room), b"abort", *extra])

    assert mgr._transfer_queue.empty()


@pytest.mark.parametrize(
    "msg",
    [
        [],
        [b"not-uniflow", ABORT_MSG, b"1"],
        [GUARD],
        [GUARD, REGISTER_MSG],
        [GUARD, ABORT_MSG],
        [GUARD, METADATA_MSG, b"1", b"127.0.0.1"],
        [GUARD, b"unknown", b"1"],
    ],
)
def test_malformed_prefill_message_is_ignored(msg: list[bytes]) -> None:
    mgr = _bare_manager()

    mgr._handle_prefill_message(msg)

    assert not mgr.request_status
    assert not mgr.transfer_infos
    assert not mgr.decode_peer_infos
    assert not mgr.pending_aborts
    assert not mgr.failure_records


def test_register_and_metadata_make_room_ready() -> None:
    room = 0xC0FFEE_214
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.Bootstrapping
    peer = _peer_info()

    mgr._handle_prefill_message([GUARD, REGISTER_MSG, msgspec.json.encode(peer)])
    mgr._handle_prefill_message(
        [
            GUARD,
            METADATA_MSG,
            _ascii(room),
            _ascii(LOCALHOST),
            _ascii(DECODE_RANK_PORT),
            DECODE_AGENT.encode("ascii"),
            np.array([7, 6], dtype=np.int32).tobytes(),
            b"1",
            b"",
            b"",
        ]
    )

    assert mgr.decode_peer_infos == {DECODE_AGENT: peer}
    assert mgr.req_to_decode_prefix_len[room] == 0
    assert mgr.request_status[room] == KVPoll.WaitingForInput
    info, peer_info = mgr.wait_for_transfer_info(room)
    assert info is mgr.transfer_infos[room]
    assert info.dst_kv_indices.tolist() == [7, 6]
    assert peer_info == peer


def test_decode_handler_counts_acks_and_applies_status() -> None:
    room = 0xC0FFEE_215
    mgr = _bare_manager()
    mgr.register_deferred_abort_room(room)

    mgr._handle_decode_message([b"ABORT_ACK", _ascii(room), b"0"])
    mgr._handle_decode_message([b"ABORT_ACK", _ascii(room + 1), b"0"])

    assert mgr.is_abort_release_safe(room, 1)
    assert not mgr.is_abort_release_safe(room + 1, 1)

    mgr.request_status[room] = KVPoll.WaitingForInput
    mgr.required_prefill_response_num_table[room] = 1
    for msg in (
        [],
        [b"not-uniflow"],
        [GUARD, _ascii(room)],
        [GUARD, _ascii(room), b"x", b"0", b""],
    ):
        mgr._handle_decode_message(msg)
    assert mgr.request_status[room] == KVPoll.WaitingForInput

    mgr._handle_decode_message([GUARD, _ascii(room), _ascii(KVPoll.Success), b"0", b""])
    assert mgr.request_status[room] == KVPoll.Success


@pytest.mark.parametrize(
    "error", [None, RuntimeError("put failed")], ids=["success", "failure"]
)
def test_prefill_status_frame_round_trips_to_decode(
    error: Exception | None,
) -> None:
    room = 0xC0FFEE_237
    prefill = _bare_manager()
    _ready_room(prefill, room, status=KVPoll.Transferring)
    calls = _record_transfer_calls(prefill)
    info = prefill.transfer_infos[room]
    if error is None:
        prefill.mark_transfer_success(info)
    else:
        prefill.mark_transfer_failure(info, error)

    [(endpoint, frame)] = calls.status_frames
    assert endpoint == f"tcp://{LOCALHOST}:{DECODE_RANK_PORT + 1}"
    decode = _bare_manager()
    decode.request_status[room] = KVPoll.WaitingForInput
    decode.required_prefill_response_num_table[room] = 1
    decode._handle_decode_message(frame)

    if error is None:
        assert decode.request_status[room] == KVPoll.Success
        assert room not in decode.failure_records
    else:
        assert decode.request_status[room] == KVPoll.Failed
        assert decode.failure_records[room] == "UniFlow transfer failed: put failed"


def test_receiver_waiting_timeout_notifies_prefill_once() -> None:
    room = 0xC0FFEE_216
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.WaitingForInput
    sock = _FakeSocket()
    receiver = _bare_receiver(mgr, room, [PREFILL_PEER], sock)
    receiver.started_transfer = True
    receiver.init_time = time.time() - 1

    assert receiver.poll() == KVPoll.Failed

    [frame] = sock.sent
    assert frame[:3] == [GUARD, ABORT_MSG, _ascii(room)]
    assert frame[3].decode("utf-8").startswith(f"Request {room} timed out")
    assert frame[4:] == [_ascii(LOCALHOST), _ascii(DECODE_RANK_PORT)]
    mgr.request_status.pop(room)
    assert receiver.poll() == KVPoll.Failed
    assert len(sock.sent) == 1


def test_receiver_waiting_timeout_starts_at_send_metadata() -> None:
    room = 0xC0FFEE_217
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.WaitingForInput
    sock = _FakeSocket()
    receiver = _bare_receiver(mgr, room, [PREFILL_PEER], sock)

    time.sleep(mgr.waiting_timeout * 2)

    assert receiver.poll() == KVPoll.WaitingForInput
    assert receiver.conclude_state is None
    assert sock.sent == []


def test_receiver_reports_bootstrapping_until_send_metadata() -> None:
    room = 0xC0FFEE_236
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.Bootstrapping
    receiver = _bare_receiver(mgr, room, [PREFILL_PEER])

    assert receiver.poll() == KVPoll.Bootstrapping
    receiver.started_transfer = True
    assert receiver.poll() == KVPoll.WaitingForInput


def test_receiver_caches_success() -> None:
    room = 0xC0FFEE_218
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.Success
    receiver = _bare_receiver(mgr, room, [PREFILL_PEER])

    assert receiver.poll() == KVPoll.Success
    mgr.request_status.pop(room)
    assert receiver.poll() == KVPoll.Success


@pytest.mark.parametrize("bootstrap_infos", [[], None], ids=["empty", "none"])
def test_send_metadata_without_prefill_peer_fails(
    bootstrap_infos: list[dict[str, Any]] | None,
) -> None:
    room = 0xC0FFEE_219
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.WaitingForInput
    receiver = _bare_receiver(mgr, room, bootstrap_infos, _FakeSocket())

    receiver.send_metadata(np.array([1], dtype=np.int32))

    assert mgr.request_status[room] == KVPoll.Failed
    assert mgr.failure_records[room] == (
        "UniFlow: no prefill peer (empty bootstrap_infos)"
    )
    assert not receiver.started_transfer


def test_metadata_frames_round_trip_to_prefill() -> None:
    room = 0xC0FFEE_220
    decode = _bare_manager()
    decode.request_status[room] = KVPoll.WaitingForInput
    sock = _FakeSocket()
    receiver = _bare_receiver(
        decode, room, [PREFILL_PEER, {**PREFILL_PEER, "is_dummy": True}], sock
    )

    receiver.send_metadata(
        np.array([7, 6], dtype=np.int32),
        aux_index=1,
        state_indices=[[3]],
        decode_prefix_len=5,
    )

    assert receiver.started_transfer
    assert receiver.init_time is not None
    real, dummy = sock.sent
    info = TransferInfo.from_zmq(real)
    assert (info.room, info.endpoint, info.dst_port, info.agent_id) == (
        room,
        LOCALHOST,
        DECODE_RANK_PORT,
        DECODE_AGENT,
    )
    assert info.dst_kv_indices.tolist() == [7, 6]
    assert info.dst_aux_index == 1
    assert info.dst_state_indices == [[3]]
    assert info.decode_prefix_len == 5
    assert not info.is_dummy
    assert len(real) == len(dummy) == 10
    assert dummy[6:9] == [b"", b"", b""]
    assert TransferInfo.from_zmq(dummy).is_dummy

    prefill = _bare_manager()
    prefill.request_status[room] = KVPoll.Bootstrapping
    prefill._handle_prefill_message(real)
    assert prefill.transfer_infos[room].dst_kv_indices.tolist() == [7, 6]
    assert prefill.req_to_decode_prefix_len[room] == 5
    assert prefill.request_status[room] == KVPoll.WaitingForInput

    sock.sent.clear()
    receiver.send_metadata(np.array([7], dtype=np.int32))
    info = TransferInfo.from_zmq(sock.sent[0])
    assert info.dst_aux_index == NO_AUX_INDEX
    assert info.dst_state_indices == []
    assert info.decode_prefix_len is None


def test_send_metadata_zmq_error_fails_and_drops_cached_route() -> None:
    room = 0xC0FFEE_221
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.WaitingForInput
    receiver = _bare_receiver(
        mgr, room, [PREFILL_PEER], _FakeSocket(error=zmq.ZMQError())
    )
    cached_route = [PREFILL_PEER]
    mgr.connection_pool["route"] = cached_route
    receiver._connection_pool_entries["route"] = cached_route

    receiver.send_metadata(np.array([1], dtype=np.int32))

    assert "route" not in mgr.connection_pool
    assert mgr.request_status[room] == KVPoll.Failed
    assert receiver.conclude_state == KVPoll.Failed
    assert mgr.failure_records[room] == (
        f"UniFlow send_metadata to prefill {LOCALHOST}:{PREFILL_RANK_PORT} failed"
    )
    assert not receiver.started_transfer


def test_register_kv_args_sends_export_ids() -> None:
    room = 0xC0FFEE_222
    sock = _FakeSocket()
    receiver = _bare_receiver(_bare_manager(), room, [PREFILL_PEER], sock)

    assert receiver._register_kv_args()

    [frame] = sock.sent
    assert frame[:2] == [GUARD, REGISTER_MSG]
    assert DecodePeerInfo.from_json_bytes(frame[2]) == _peer_info()


def test_register_kv_args_zmq_error_fails_room() -> None:
    room = 0xC0FFEE_223
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.Bootstrapping
    receiver = _bare_receiver(
        mgr, room, [PREFILL_PEER], _FakeSocket(error=zmq.ZMQError())
    )

    assert not receiver._register_kv_args()

    assert mgr.request_status[room] == KVPoll.Failed
    assert receiver.conclude_state == KVPoll.Failed
    assert mgr.failure_records[room] == (
        f"UniFlow register to prefill {LOCALHOST}:{PREFILL_RANK_PORT} failed"
    )


def test_abort_notification_continues_past_failed_peer() -> None:
    room = 0xC0FFEE_224
    good_sock = _FakeSocket()
    socks = {
        PREFILL_RANK_PORT: _FakeSocket(error=RuntimeError("peer down")),
        PREFILL_RANK_PORT + 1: good_sock,
    }
    receiver = _bare_receiver(
        _bare_manager(),
        room,
        [PREFILL_PEER, {**PREFILL_PEER, "rank_port": PREFILL_RANK_PORT + 1}],
    )
    receiver._connect_to_bootstrap_server = lambda info: (
        socks[info["rank_port"]],
        threading.Lock(),
    )

    receiver._send_abort_notification()

    [frame] = good_sock.sent
    assert frame[3] == DEFAULT_ABORT_REASON.encode("utf-8")


def test_receiver_abort_notifies_prefill_once() -> None:
    room = 0xC0FFEE_225
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.WaitingForInput
    sock = _FakeSocket()
    receiver = _bare_receiver(mgr, room, [PREFILL_PEER], sock)

    receiver.abort()
    receiver.abort()

    [frame] = sock.sent
    assert frame[3] == b"Aborted by AbortReq."
    assert receiver.poll() == KVPoll.Failed


@pytest.mark.parametrize("cause", ["waiting-timeout", "prefill-failed"])
def test_receiver_abort_keeps_recorded_failure_reason(cause: str) -> None:
    room = 0xC0FFEE_305
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.WaitingForInput
    mgr.required_prefill_response_num_table[room] = 1
    receiver = _bare_receiver(mgr, room, [PREFILL_PEER], _FakeSocket())
    receiver.started_transfer = True
    if cause == "waiting-timeout":
        receiver.init_time = time.time() - 1
    else:
        mgr._handle_decode_message(
            [GUARD, _ascii(room), _ascii(KVPoll.Failed), b"0", b"put failed"]
        )
    assert receiver.poll() == KVPoll.Failed
    reason = mgr.failure_records[room]

    receiver.abort()

    assert mgr.failure_records[room] == reason
    with pytest.raises(KVTransferError, match=re.escape(reason)):
        receiver.failure_exception()


@pytest.mark.parametrize(
    "published, force_arm, armed",
    [(True, False, True), (False, False, False), (False, True, True)],
    ids=["published", "unpublished", "unpublished-force-arm"],
)
def test_abort_notification_arms_drain_ack_tracker(
    published: bool, force_arm: bool, armed: bool
) -> None:
    room = 0xC0FFEE_307
    mgr = _bare_manager(enable_deferred_decode_kv_release=True)
    sock = _FakeSocket()
    receiver = _bare_receiver(mgr, room, [PREFILL_PEER], sock)
    if published:
        receiver.init_time = time.time()

    receiver.ensure_abort_notified(force_arm=force_arm)

    [frame] = sock.sent
    assert frame[:2] == [GUARD, ABORT_MSG]
    assert receiver.abort_notified
    assert (room in mgr._deferred_abort_ack_tracker) == armed


@pytest.mark.parametrize(
    "dst_info_num, prefill_response_num",
    [(2, 1), (1, 2)],
    ids=["many-decode", "many-prefill"],
)
def test_receiver_rejects_non_one_to_one_routing(
    monkeypatch: pytest.MonkeyPatch, dst_info_num: int, prefill_response_num: int
) -> None:
    room = 0xC0FFEE_226

    def base_init(self, prefill_dp_rank: int) -> None:
        self.required_dst_info_num = dst_info_num
        self.required_prefill_response_num = prefill_response_num

    monkeypatch.setattr(CommonKVReceiver, "init", base_init)
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.WaitingForInput
    sock = _FakeSocket()
    receiver = _bare_receiver(mgr, room, [PREFILL_PEER], sock)

    receiver.init(0)

    assert receiver.conclude_state == KVPoll.Failed
    assert mgr.request_status[room] == KVPoll.Failed
    assert "one-to-one" in mgr.failure_records[room]
    # Prefill learns of the rejection now instead of at its bootstrap timeout.
    (frame,) = sock.sent
    assert frame[:3] == [GUARD, ABORT_MSG, _ascii(room)]
    assert b"one-to-one" in frame[3]

    # A later scheduler abort neither repeats the ABORT nor drops the reason.
    receiver.abort()
    assert len(sock.sent) == 1
    assert "one-to-one" in mgr.failure_records[room]


def test_receiver_init_keeps_base_failure_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    room = 0xC0FFEE_227

    def failing_base_init(self, prefill_dp_rank: int) -> None:
        self.kv_mgr.record_failure(self.bootstrap_room, "prefill server is down")
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        self.conclude_state = KVPoll.Failed

    monkeypatch.setattr(CommonKVReceiver, "init", failing_base_init)
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.Bootstrapping
    receiver = _bare_receiver(mgr, room, [PREFILL_PEER])

    receiver.init(0)

    assert mgr.failure_records[room] == "prefill server is down"


def test_receiver_failure_exception_consumes_reason() -> None:
    room = 0xC0FFEE_228
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.Failed
    mgr.failure_records[room] = "prefill transfer failed"
    receiver = _bare_receiver(mgr, room, [PREFILL_PEER])

    with pytest.raises(KVTransferError, match="prefill transfer failed") as raised:
        receiver.failure_exception()
    assert not raised.value.is_from_another_rank
    assert room not in mgr.request_status
    with pytest.raises(KVTransferError, match=UNKNOWN_FAILURE_REASON) as raised:
        receiver.failure_exception()
    assert raised.value.is_from_another_rank


def test_sender_bootstrap_timeout_concludes_once() -> None:
    room = 0xC0FFEE_229
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.Bootstrapping
    sender = _bare_sender(mgr, room)
    sender.init_time = time.time() - 1

    assert sender.poll() == KVPoll.Failed
    with pytest.raises(KVTransferError, match=re.escape("KVPoll.Bootstrapping")):
        sender.failure_exception()
    assert room not in mgr.request_status
    assert sender.poll() == KVPoll.Failed


def test_sender_caches_success_and_latency() -> None:
    room = 0xC0FFEE_230
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.Success
    sender = _bare_sender(mgr, room)
    sender._transfer_start_time = time.perf_counter() - 0.5

    assert sender.poll() == KVPoll.Success
    assert sender._transfer_metric.transfer_latency_s >= 0.5
    mgr.request_status.pop(room)
    assert sender.poll() == KVPoll.Success


def test_sender_has_no_bootstrap_timeout_after_metadata() -> None:
    room = 0xC0FFEE_231
    mgr = _bare_manager()
    mgr.request_status[room] = KVPoll.WaitingForInput
    sender = _bare_sender(mgr, room)
    sender.init_time = time.time() - 1

    assert sender.poll() == KVPoll.WaitingForInput
    assert room not in mgr.failure_records


def test_sender_enqueues_chunks_in_order() -> None:
    room = 0xC0FFEE_232
    mgr = _bare_manager()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    mgr.add_transfer_request = lambda *args, **kwargs: calls.append((args, kwargs))
    sender = _bare_sender(mgr, room, num_kv_indices=4, aux_index=2)
    event = object()
    sender._early_send_wait_event = event

    sender.send(np.array([1, 2], dtype=np.int64))
    sender.send(np.array([3, 5], dtype=np.int64), [[0]])

    (first_args, first_kwargs), (second_args, second_kwargs) = calls
    assert first_args[0] == room
    assert first_args[1].dtype == np.int32
    assert first_args[1].tolist() == [1, 2]
    assert first_kwargs == dict(
        index_slice=slice(0, 2),
        is_last=False,
        aux_index=2,
        state_indices=None,
        wait_event=event,
    )
    assert second_args[1].tolist() == [3, 5]
    assert second_kwargs == dict(
        index_slice=slice(2, 4),
        is_last=True,
        aux_index=2,
        state_indices=[[0]],
        wait_event=None,
    )
    assert sender._early_send_wait_event is None
    assert sender._transfer_start_time is not None
    assert sender._transfer_num_kv_indices == 4
    assert sender._transfer_num_state_indices == 1


def test_dummy_cp_rank_enqueues_nothing() -> None:
    room = 0xC0FFEE_233
    mgr = _bare_manager(is_dummy_cp_rank=True)
    mgr.request_status[room] = KVPoll.WaitingForInput
    mgr.add_transfer_request = lambda *_args, **_kwargs: pytest.fail(
        "dummy CP rank transferred"
    )
    sender = _bare_sender(mgr, room, num_kv_indices=2)

    sender.send(np.array([1], dtype=np.int32))
    assert mgr.request_status[room] == KVPoll.WaitingForInput
    sender.send(np.array([2], dtype=np.int32))
    assert sender.poll() == KVPoll.Success


def test_transfer_failure_for_cleared_room_only_notifies() -> None:
    room = 0xC0FFEE_234
    mgr = _bare_manager()
    calls = _record_transfer_calls(mgr)

    mgr.mark_transfer_failure(_transfer_info(room), RuntimeError("put failed"))

    [(_, frame)] = calls.status_frames
    assert frame == [
        GUARD,
        _ascii(room),
        _ascii(KVPoll.Failed),
        b"0",
        b"UniFlow transfer failed: put failed",
    ]
    assert not mgr.request_status
    assert not mgr.failure_records


def test_transfer_failure_records_and_clears_room() -> None:
    room = 0xC0FFEE_235
    mgr = _bare_manager()
    _ready_room(mgr, room, status=KVPoll.Transferring)
    calls = _record_transfer_calls(mgr, notify_error=RuntimeError("decode unreachable"))
    error = RuntimeError("put failed")

    mgr.mark_transfer_failure(mgr.transfer_infos[room], error)

    assert mgr.request_status[room] == KVPoll.Failed
    assert mgr.failure_records[room] == "UniFlow transfer failed: put failed"
    assert room not in mgr.transfer_infos
    assert calls.notified == [KVPoll.Failed]


def test_dummy_rank_concludes_on_last_chunk_without_transfer() -> None:
    room = 0xC0FFEE_238
    mgr = _bare_manager(registered_aux_segments=["aux-segment"])
    _ready_room(
        mgr,
        room,
        is_dummy=True,
        dst_kv_indices=np.array([], dtype=np.int32),
        dst_aux_index=NO_AUX_INDEX,
    )
    calls = _record_transfer_calls(
        mgr, on_connect=lambda: pytest.fail("dummy rank connected")
    )

    mgr._do_transfer_request(
        room,
        np.array([1], dtype=np.int32),
        index_slice=slice(0, 1),
        is_last=False,
        aux_index=0,
        state_indices=None,
        wait_event=None,
    )
    assert mgr.request_status[room] == KVPoll.WaitingForInput
    assert room in mgr.transfer_infos

    mgr._do_transfer_request(
        room,
        np.array([2], dtype=np.int32),
        index_slice=slice(1, 2),
        is_last=True,
        aux_index=0,
        state_indices=[[0]],
        wait_event=None,
    )
    assert mgr.request_status[room] == KVPoll.Success
    assert room not in mgr.transfer_infos
    assert calls.puts == []
    assert calls.status_frames == []


def test_early_send_event_is_synchronized_before_kv_put() -> None:
    room = 0xC0FFEE_239
    mgr = _bare_manager()
    _ready_room(mgr, room)
    calls = _record_transfer_calls(mgr)
    event = SimpleNamespace(synchronize=lambda: calls.puts.append(("sync",)))

    mgr._do_transfer_request(
        room,
        np.array([1, 2], dtype=np.int32),
        index_slice=slice(0, 2),
        is_last=False,
        aux_index=0,
        state_indices=None,
        wait_event=event,
    )

    assert calls.puts == [("sync",), ("kv", [1, 2], [4, 5])]
    assert mgr.request_status[room] == KVPoll.Transferring


def _manager_args(state_types: list[str], num_state_components: int) -> KVArgs:
    args = KVArgs()
    args.state_types = state_types
    args.state_data_ptrs = [[0]] * num_state_components
    return args


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cp_size: int = 1,
    pp_size: int = 1,
    dcp_size: int = 1,
    hisparse: bool = False,
) -> None:
    monkeypatch.setattr(
        uniflow_conn,
        "get_parallel",
        lambda: SimpleNamespace(
            attn_cp_size=cp_size, pp_size=pp_size, attn_dcp_size=dcp_size
        ),
    )
    monkeypatch.setattr(
        uniflow_conn,
        "get_memory",
        lambda: SimpleNamespace(enable_hisparse=hisparse),
    )


@pytest.mark.parametrize(
    "runtime, state_types, num_components, match",
    [
        ({"cp_size": 2}, [], 0, "CP=1 and PP=1"),
        ({"pp_size": 2}, [], 0, "CP=1 and PP=1"),
        ({"dcp_size": 2}, [], 0, "decode context parallelism"),
        ({"hisparse": True}, [], 0, "hisparse"),
        ({}, ["swa_ring"], 1, "does not support state types"),
        ({}, ["mamba"], 2, "count mismatch"),
    ],
    ids=["cp", "pp", "dcp", "hisparse", "state-type", "component-count"],
)
def test_manager_rejects_unsupported_configuration(
    monkeypatch: pytest.MonkeyPatch,
    runtime: dict[str, Any],
    state_types: list[str],
    num_components: int,
    match: str,
) -> None:
    _patch_runtime(monkeypatch, **runtime)

    with pytest.raises(ValueError, match=match):
        UniflowKVManager(
            _manager_args(state_types, num_components),
            DisaggregationMode.PREFILL,
            None,
        )


@pytest.mark.parametrize("timeout_ms", [0, -1])
def test_manager_rejects_non_positive_transfer_timeout(
    monkeypatch: pytest.MonkeyPatch, timeout_ms: int
) -> None:
    _patch_runtime(monkeypatch)

    with envs.SGLANG_DISAGGREGATION_UNIFLOW_TRANSFER_TIMEOUT_MS.override(timeout_ms):
        with pytest.raises(ValueError, match="must be positive"):
            UniflowKVManager(_manager_args([], 0), DisaggregationMode.PREFILL, None)


def test_manager_rejects_mooncake_custom_mem_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch)

    with envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override("INTRA_NODE_NVLINK"):
        with pytest.raises(ValueError, match="SGLANG_MOONCAKE_CUSTOM_MEM_POOL"):
            UniflowKVManager(_manager_args([], 0), DisaggregationMode.PREFILL, None)


class _BaseInitReached(Exception):
    pass


def _reach_base_init(*_args: Any, **_kwargs: Any) -> None:
    raise _BaseInitReached


@pytest.mark.parametrize("state_types", [[], ["mamba"], ["swa"], ["dsa"]])
def test_manager_accepts_supported_state_types(
    monkeypatch: pytest.MonkeyPatch, fake_uniflow: None, state_types: list[str]
) -> None:
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(CommonKVManager, "__init__", _reach_base_init)

    with pytest.raises(_BaseInitReached):
        UniflowKVManager(
            _manager_args(state_types, len(state_types)),
            DisaggregationMode.PREFILL,
            None,
        )


def test_manager_without_binding_fails_before_opening_sockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(CommonKVManager, "__init__", _reach_base_init)
    monkeypatch.setitem(sys.modules, "uniflow._core", None)

    with pytest.raises(ImportError, match="--disaggregation-transfer-backend uniflow"):
        UniflowKVManager(_manager_args([], 0), DisaggregationMode.PREFILL, None)


class _SpanSegment:
    def span(self, offset: int, length: int) -> tuple[int, int]:
        return offset, length


def test_state_index_count_drift_truncates_only_paged_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    segment = _SpanSegment()
    mgr = _bare_manager(
        state_types=[StateType.SWA, StateType.MAMBA],
        registered_state_segments=[[segment], [segment]],
        kv_args=SimpleNamespace(state_item_lens=[[8], [16]]),
        transfer_request_cls=lambda src, dst: (src, dst),
    )
    puts: list[list[Any]] = []

    def put(requests: list[Any]) -> _ScriptedFuture:
        puts.append(requests)
        return _ScriptedFuture(is_done=True)

    peer = DecodePeer(
        connection=SimpleNamespace(put=put),
        remote_kv_segments=[],
        remote_aux_segments=[],
        remote_state_segments=[[segment], [segment]],
    )

    with caplog.at_level(logging.WARNING, logger=uniflow_conn.logger.name):
        mgr._put_state(peer, [[1, 2, 3], [4]], [[5, 6], [7]])

    assert puts == [[((8, 8), (40, 8)), ((16, 8), (48, 8)), ((64, 16), (112, 16))]]
    assert "Truncating UniFlow swa state indices" in caplog.text
    with pytest.raises(ValueError, match="type=mamba"):
        mgr._put_state(peer, [[1], [4, 5]], [[5], [7]])


class _ScriptedFuture:
    def __init__(self, *, is_done: bool, error: str | None = None) -> None:
        self.is_done = is_done
        self.error = error
        self.timeouts: list[int] = []

    def wait_for(self, timeout_ms: int) -> bool:
        self.timeouts.append(timeout_ms)
        return self.is_done

    def get(self) -> SimpleNamespace:
        assert self.is_done, "get() after a timed-out wait"
        return SimpleNamespace(
            has_error=lambda: self.error is not None, error=lambda: self.error
        )


def _put_peer(put: Callable[..., Any]) -> DecodePeer:
    return DecodePeer(
        connection=SimpleNamespace(put=put),
        remote_kv_segments=[],
        remote_aux_segments=[],
        remote_state_segments=[],
    )


def test_put_raises_put_error_on_timeout_and_failure() -> None:
    mgr = _bare_manager(transfer_timeout_ms=5)

    timed_out = _ScriptedFuture(is_done=False)
    with pytest.raises(UniflowPutError, match="KV put timed out"):
        mgr._put(_put_peer(lambda requests: timed_out), ["req"], "KV put")
    assert timed_out.timeouts == [5]
    failed = _ScriptedFuture(is_done=True, error="link down")
    with pytest.raises(UniflowPutError, match="aux put failed: link down"):
        mgr._put(_put_peer(lambda requests: failed), ["req"], "aux put")

    def refuse(requests: list[Any]) -> None:
        raise RuntimeError("connection closed")

    with pytest.raises(UniflowPutError, match="state put failed: connection closed"):
        mgr._put(_put_peer(refuse), ["req"], "state put")

    def broken_get() -> None:
        raise RuntimeError("broken promise")

    # A binding error after the put was issued is a put failure,
    # so the caller shuts the connection down.
    broken = SimpleNamespace(wait_for=lambda timeout_ms: True, get=broken_get)
    with pytest.raises(UniflowPutError, match="KV put failed: broken promise"):
        mgr._put(_put_peer(lambda requests: broken), ["req"], "KV put")

    mgr._put(
        _put_peer(lambda requests: _ScriptedFuture(is_done=True)), ["req"], "KV put"
    )
    mgr._put(_put_peer(lambda requests: pytest.fail("empty batch put")), [], "KV put")


class _PendingFuture:
    """A put still on the wire: it completes only when its connection shuts down."""

    def __init__(self) -> None:
        self.completed = threading.Event()

    def wait_for(self, timeout_ms: int) -> bool:
        return self.completed.wait(timeout_ms / 1000)


class _DrainingConnection:
    """Puts never finish on their own; shutdown() completes them.

    This models the RDMA tier, whose shutdown() waits for in-flight puts.
    """

    def __init__(
        self,
        *,
        on_put: Callable[[], None] | None = None,
        shutdown_error: Exception | None = None,
    ) -> None:
        self.on_put = on_put
        self.shutdown_error = shutdown_error
        self.futures: list[_PendingFuture] = []
        self.events: list[str] = []

    def put(self, requests: list[Any]) -> _PendingFuture:
        future = _PendingFuture()
        self.futures.append(future)
        self.events.append("put")
        if self.on_put is not None:
            self.on_put()
        return future

    def shutdown(self) -> None:
        self.events.append("shutdown")
        if self.shutdown_error is not None:
            raise self.shutdown_error
        for future in self.futures:
            future.completed.set()


def _pending_put_manager(
    room: int, connection: _DrainingConnection, **attrs: Any
) -> tuple[UniflowKVManager, list[tuple[list[bytes], bool]]]:
    """A ready room whose KV put times out; frames record whether puts had drained."""
    segment = _SpanSegment()
    mgr = _bare_manager(
        transfer_timeout_ms=20,
        registered_kv_segments=[segment],
        kv_args=SimpleNamespace(kv_item_lens=[8], aux_item_lens=[]),
        transfer_request_cls=lambda src, dst: (src, dst),
        **attrs,
    )
    _ready_room(mgr, room)
    mgr.decode_peers[DECODE_AGENT] = DecodePeer(
        connection=connection,
        remote_kv_segments=[segment],
        remote_aux_segments=[],
        remote_state_segments=[],
    )
    frames: list[tuple[list[bytes], bool]] = []

    def send(endpoint: str, parts: list[bytes], is_ipv6: bool = False) -> None:
        is_drained = all(future.completed.is_set() for future in connection.futures)
        frames.append((list(parts), is_drained))

    mgr._send_multipart_locked = send
    return mgr, frames


def test_put_timeout_drains_connection_before_reporting_failed() -> None:
    room = 0xC0FFEE_245
    connection = _DrainingConnection()
    mgr, frames = _pending_put_manager(room, connection)

    mgr._do_transfer_request(
        room,
        np.array([0, 1, 2], dtype=np.int32),
        index_slice=slice(0, 3),
        is_last=False,
        aux_index=None,
        state_indices=None,
        wait_event=None,
    )

    # Decode frees its pages on Failed, so no write may still be in flight.
    assert connection.events == ["put", "shutdown"]
    [(parts, is_drained)] = frames
    assert parts[:3] == [GUARD, _ascii(room), _ascii(KVPoll.Failed)]
    assert b"KV put timed out" in parts[4]
    assert is_drained
    assert mgr.request_status[room] == KVPoll.Failed
    # The next transfer to this decode reconnects.
    assert DECODE_AGENT not in mgr.decode_peers


def _decode_abort_frame(room: int) -> list[bytes]:
    return [GUARD, ABORT_MSG, _ascii(room), b"client gone", _ascii(LOCALHOST), b"17001"]


def test_decode_abort_during_put_is_acked_only_after_drain() -> None:
    room = 0xC0FFEE_246
    mgr_ref: list[UniflowKVManager] = []
    connection = _DrainingConnection(
        on_put=lambda: mgr_ref[0]._handle_prefill_message(_decode_abort_frame(room))
    )
    mgr, frames = _pending_put_manager(
        room, connection, enable_deferred_decode_kv_release=True
    )
    mgr_ref.append(mgr)

    mgr.add_transfer_request(
        room,
        np.array([0, 1, 2], dtype=np.int32),
        index_slice=slice(0, 3),
        is_last=True,
        aux_index=None,
        state_indices=None,
        wait_event=None,
    )
    _run_queued_tasks(mgr)

    assert [parts[0] for parts, _ in frames] == [GUARD, b"ABORT_ACK"]
    assert all(is_drained for _, is_drained in frames)
    # The decode abort stays the recorded cause, not the put timeout it caused.
    assert frames[0][0][4] == b"client gone"
    assert mgr.failure_records[room] == "client gone"


def test_failed_shutdown_after_put_failure_leaves_decode_to_its_timeouts() -> None:
    room = 0xC0FFEE_247
    connection = _DrainingConnection(shutdown_error=RuntimeError("shutdown failed"))
    mgr, frames = _pending_put_manager(room, connection)

    mgr._do_transfer_request(
        room,
        np.array([0, 1, 2], dtype=np.int32),
        index_slice=slice(0, 3),
        is_last=False,
        aux_index=None,
        state_indices=None,
        wait_event=None,
    )

    # Telling decode Failed would let it reuse pages a put may still write.
    assert frames == []
    assert mgr.request_status[room] == KVPoll.Failed
    assert "KV put timed out" in mgr.failure_records[room]
    assert DECODE_AGENT not in mgr.decode_peers


def test_decode_abort_during_put_with_failed_shutdown_is_not_acked() -> None:
    room = 0xC0FFEE_249
    mgr_ref: list[UniflowKVManager] = []
    connection = _DrainingConnection(
        on_put=lambda: mgr_ref[0]._handle_prefill_message(_decode_abort_frame(room)),
        shutdown_error=RuntimeError("shutdown failed"),
    )
    mgr, frames = _pending_put_manager(
        room, connection, enable_deferred_decode_kv_release=True
    )
    mgr_ref.append(mgr)

    mgr.add_transfer_request(
        room,
        np.array([0, 1, 2], dtype=np.int32),
        index_slice=slice(0, 3),
        is_last=True,
        aux_index=None,
        state_indices=None,
        wait_event=None,
    )
    _run_queued_tasks(mgr)

    # An ack would let decode reuse pages the put may still write.
    assert frames == []
    assert mgr.failure_records[room] == "client gone"


def test_decode_abort_after_failed_shutdown_is_not_acked() -> None:
    room = 0xC0FFEE_24A
    connection = _DrainingConnection(shutdown_error=RuntimeError("shutdown failed"))
    mgr, frames = _pending_put_manager(
        room, connection, enable_deferred_decode_kv_release=True
    )
    mgr.add_transfer_request(
        room,
        np.array([0, 1, 2], dtype=np.int32),
        index_slice=slice(0, 3),
        is_last=False,
        aux_index=None,
        state_indices=None,
        wait_event=None,
    )
    _run_queued_tasks(mgr)

    mgr._handle_prefill_message(_decode_abort_frame(room))
    _run_queued_tasks(mgr)

    assert frames == []
    assert mgr.request_status[room] == KVPoll.Failed


def test_prefill_abort_during_last_put_reports_failed_to_decode() -> None:
    room = 0xC0FFEE_248
    mgr = _bare_manager(registered_aux_segments=["aux"])
    calls = _record_transfer_calls(mgr)
    _ready_room(mgr, room)
    sender = _bare_sender(mgr, room)
    put_kv = mgr._put_kv

    def put_then_abort(peer: Any, src: Any, dst: Any) -> None:
        put_kv(peer, src, dst)
        sender.abort()

    mgr._put_kv = put_then_abort

    mgr._do_transfer_request(
        room,
        np.array([0, 1, 2], dtype=np.int32),
        index_slice=slice(0, 3),
        is_last=True,
        aux_index=0,
        state_indices=None,
        wait_event=None,
    )

    # A stopped room gets no further writes, so aux and state are skipped.
    assert [put[0] for put in calls.puts] == ["kv"]
    assert [frame for _, frame in calls.status_frames] == [
        [GUARD, _ascii(room), _ascii(KVPoll.Failed), b"0", b"Aborted by AbortReq."]
    ]
    assert mgr.request_status[room] == KVPoll.Failed


class _ClearAfterFirstGet(dict):
    """request_status whose first get() runs a hook, modelling a thread switch."""

    def __init__(self, *args: Any, on_first_get: Callable[[], None]) -> None:
        super().__init__(*args)
        self.on_first_get: Callable[[], None] | None = on_first_get

    def get(self, key: Any, default: Any = None) -> Any:
        value = super().get(key, default)
        hook, self.on_first_get = self.on_first_get, None
        if hook is not None:
            hook()
        return value


def test_abort_and_clear_after_worker_check_do_not_stall_worker() -> None:
    room = 0xC0FFEE_249
    mgr = _bare_manager(bootstrap_timeout=5.0)
    calls = _record_transfer_calls(mgr)
    _ready_room(mgr, room)
    sender = _bare_sender(mgr, room)

    def scheduler_abort_then_clear() -> None:
        sender.abort()
        sender.clear()

    mgr.request_status = _ClearAfterFirstGet(
        mgr.request_status, on_first_get=scheduler_abort_then_clear
    )
    start = time.monotonic()
    mgr._do_transfer_request(
        room,
        np.array([0, 1, 2], dtype=np.int32),
        index_slice=slice(0, 3),
        is_last=True,
        aux_index=0,
        state_indices=None,
        wait_event=None,
    )

    # The single worker must not sit out the bootstrap timeout for a room its
    # sender already concluded; every other room is queued behind it.
    assert time.monotonic() - start < 1.0
    assert calls.puts == []
    assert calls.status_frames == []


class _SignalOnGet(dict):
    """request_status that signals once the transfer worker has read it."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.was_read = threading.Event()

    def get(self, key: Any, default: Any = None) -> Any:
        self.was_read.set()
        return super().get(key, default)


@pytest.mark.parametrize(
    ("conclude", "reason"),
    [
        pytest.param(UniflowKVSender.abort, "Aborted by AbortReq.", id="abort"),
        pytest.param(
            UniflowKVSender.clear, "concluded before its transfer started", id="clear"
        ),
    ],
)
def test_sender_conclude_wakes_worker_waiting_for_metadata(
    conclude: Callable[[UniflowKVSender], None], reason: str
) -> None:
    room = 0xC0FFEE_250
    mgr = _bare_manager(bootstrap_timeout=30.0)
    mgr.request_status = _SignalOnGet({room: KVPoll.Bootstrapping})
    sender = _bare_sender(mgr, room)
    errors: list[KVTransferError] = []

    def worker() -> None:
        try:
            mgr.wait_for_transfer_info(room)
        except KVTransferError as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        assert mgr.request_status.was_read.wait(timeout=5.0)
        # The worker holds the condition from that read until wait() releases
        # it, so the sender runs only once the worker sleeps on its notify.
        conclude(sender)
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "worker slept through the sender's conclusion"
    finally:
        with mgr.condition:
            mgr.request_status.pop(room, None)
            mgr.condition.notify_all()
        thread.join()
    assert len(errors) == 1 and reason in str(errors[0]), errors


def _make_buffers(dtype_str: str, device: str) -> dict[str, Any]:
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    kv_buffers = [
        torch.zeros(
            NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device
        )
        for _ in range(NUM_LAYERS * 2)
    ]
    aux_buffer = torch.zeros(NUM_AUX_SLOTS, AUX_BYTES_PER_SLOT, dtype=torch.uint8)
    state_buffer = torch.zeros(
        NUM_STATE_SLOTS, STATE_BYTES, dtype=torch.uint8, device=device
    )

    def ptrs_lens_items(
        bufs: list[torch.Tensor],
    ) -> tuple[list[int], list[int], list[int]]:
        ptrs = [int(b.data_ptr()) for b in bufs]
        lens = [int(b.numel() * b.element_size()) for b in bufs]
        items = [int(b[0].numel() * b.element_size()) for b in bufs]
        return ptrs, lens, items

    kv_ptrs, kv_lens, kv_items = ptrs_lens_items(kv_buffers)
    aux_ptrs, aux_lens, aux_items = ptrs_lens_items([aux_buffer])
    state_ptrs, state_lens, state_items = ptrs_lens_items([state_buffer])

    return {
        "kv_buffers": kv_buffers,
        "aux_buffer": aux_buffer,
        "state_buffer": state_buffer,
        "kv_ptrs": kv_ptrs,
        "kv_lens": kv_lens,
        "kv_items": kv_items,
        "aux_ptrs": aux_ptrs,
        "aux_lens": aux_lens,
        "aux_items": aux_items,
        "state_ptrs": state_ptrs,
        "state_lens": state_lens,
        "state_items": state_items,
    }


def _fill_pattern(buffers: dict[str, Any], seed: int) -> None:
    device = buffers["kv_buffers"][0].device
    generator = torch.Generator(device=device).manual_seed(seed)
    for tensor in buffers["kv_buffers"]:
        tensor.copy_(
            torch.randn(
                tensor.shape, generator=generator, dtype=torch.float32, device=device
            ).to(tensor.dtype)
        )
    buffers["aux_buffer"].copy_(
        torch.randint(
            0,
            256,
            buffers["aux_buffer"].shape,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.int32,
        ).to(torch.uint8)
    )
    buffers["state_buffer"].copy_(
        torch.randint(
            0,
            256,
            buffers["state_buffer"].shape,
            generator=generator,
            dtype=torch.int32,
            device=device,
        ).to(torch.uint8)
    )


def _hash_pages(
    buffers: dict[str, Any], page_indices: list[int], aux_idx: int, state_idx: int
) -> str:
    digest = hashlib.sha256()
    for tensor in buffers["kv_buffers"]:
        digest.update(
            tensor[page_indices].contiguous().cpu().view(torch.uint8).numpy().tobytes()
        )
    digest.update(buffers["aux_buffer"][aux_idx].numpy().tobytes())
    digest.update(buffers["state_buffer"][state_idx].cpu().numpy().tobytes())
    return digest.hexdigest()


def _build_kv_args(buffers: dict[str, Any], dtype_str: str) -> KVArgs:
    args = KVArgs()
    args.kv_cache_dtype_str = dtype_str
    args.engine_rank = 0
    args.kv_data_ptrs = buffers["kv_ptrs"]
    args.kv_data_lens = buffers["kv_lens"]
    args.kv_item_lens = buffers["kv_items"]
    args.aux_data_ptrs = buffers["aux_ptrs"]
    args.aux_data_lens = buffers["aux_lens"]
    args.aux_item_lens = buffers["aux_items"]
    args.state_data_ptrs = [buffers["state_ptrs"]]
    args.state_data_lens = [buffers["state_lens"]]
    args.state_item_lens = [buffers["state_items"]]
    args.state_types = ["mamba"]
    args.gpu_id = 0
    args.kv_head_num = NUM_KV_HEADS
    args.total_kv_head_num = NUM_KV_HEADS
    args.page_size = PAGE_SIZE
    args.pp_rank = 0
    args.system_dp_rank = 0
    args.prefill_start_layer = 0
    args.prefill_end_layer = NUM_LAYERS
    args.mla_compression_ratios = None
    return args


@pytest.fixture()
def bootstrap_server() -> Iterator[tuple[str, int]]:
    port = _free_port()
    server = UniflowKVBootstrapServer(LOCALHOST, port)
    try:
        yield LOCALHOST, port
    finally:
        with contextlib.suppress(Exception):
            server.close()


class _FakeResult:
    def __init__(self, value: Any = None) -> None:
        self._value = value

    def has_value(self) -> bool:
        return True

    def has_error(self) -> bool:
        return False

    def value(self) -> Any:
        return self._value


class _FakeSegment:
    """A segment whose export id is its host address and length."""

    def __init__(
        self,
        ptr: int,
        length: int,
        memory_type: str | None = None,
        device_id: int | None = None,
    ) -> None:
        self.ptr = ptr
        self.length = length
        self.memory_type = memory_type
        self.device_id = device_id

    def export_id(self) -> _FakeResult:
        return _FakeResult(struct.pack("<QQ", self.ptr, self.length))

    def span(self, offset: int, length: int) -> tuple[int, int]:
        assert 0 <= offset and offset + length <= self.length
        return self.ptr + offset, length


class _FakeFuture:
    def wait_for(self, timeout_ms: int) -> bool:
        return True

    def get(self) -> _FakeResult:
        return _FakeResult()


class _FakeConnection:
    def shutdown(self) -> None:
        pass

    def put(self, requests: list[Any]) -> _FakeFuture:
        for request in requests:
            (src, src_len), (dst, dst_len) = request.src, request.dst
            assert src_len == dst_len
            ctypes.memmove(dst, src, src_len)
        return _FakeFuture()


class _FakeAgent:
    def __init__(self, config: Any) -> None:
        self._agent_id = f"fake-{config.name}-{id(self)}"

    def get_unique_id(self) -> _FakeResult:
        return _FakeResult(self._agent_id)

    def register_segment(self, segment: _FakeSegment) -> _FakeResult:
        return _FakeResult(segment)

    def import_segment(self, export_id: bytes) -> _FakeResult:
        return _FakeResult(_FakeSegment(*struct.unpack("<QQ", export_id)))

    def connect(self, agent_id: str) -> _FakeResult:
        return _FakeResult(_FakeConnection())

    def accept(self) -> None:
        threading.Event().wait()


@pytest.fixture()
def fake_uniflow(monkeypatch: pytest.MonkeyPatch) -> None:
    core = ModuleType("uniflow._core")
    core.MemoryType = SimpleNamespace(VRAM="VRAM", DRAM="DRAM")
    core.Segment = _FakeSegment
    core.TransferRequest = lambda src, dst: SimpleNamespace(src=src, dst=dst)
    core.UniflowAgentConfig = lambda device_id, name, listen_address: SimpleNamespace(
        name=name
    )
    core.UniflowAgent = _FakeAgent
    monkeypatch.setitem(sys.modules, "uniflow", ModuleType("uniflow"))
    monkeypatch.setitem(sys.modules, "uniflow._core", core)
    monkeypatch.setenv("SGLANG_HOST_IP", LOCALHOST)


@pytest.fixture()
def cpu_pd(
    fake_uniflow: None, bootstrap_server: tuple[str, int]
) -> Iterator[SimpleNamespace]:
    """Prefill and decode managers on host buffers, handshaken with each other.

    Manager threads are daemons with no upstream stop contract, so each use
    leaves them running until the process exits; only the decode heartbeat
    is stopped here.
    """
    host, port = bootstrap_server
    addr = f"{host}:{port}"
    with contextlib.ExitStack() as stack:
        server_args = stack.enter_context(
            get_context().override_server_args(
                host=host,
                disaggregation_bootstrap_port=port,
                tp_size=1,
                kv_cache_dtype="bfloat16",
            )
        )
        stack.enter_context(get_parallel().override(**SINGLE_RANK_PARALLEL))
        prefill_buffers = _make_buffers("bfloat16", "cpu")
        decode_buffers = _make_buffers("bfloat16", "cpu")
        _fill_pattern(prefill_buffers, 1)
        _fill_pattern(decode_buffers, 2)
        prefill = UniflowKVManager(
            _build_kv_args(prefill_buffers, "bfloat16"),
            DisaggregationMode.PREFILL,
            server_args,
            is_mla_backend=False,
        )
        decode = UniflowKVManager(
            _build_kv_args(decode_buffers, "bfloat16"),
            DisaggregationMode.DECODE,
            server_args,
            is_mla_backend=False,
        )
        stack.callback(decode._heartbeat_shutdown.set)
        assert _wait_until(
            lambda: decode.try_ensure_parallel_info(addr), INIT_HANDSHAKE_TIMEOUT
        )
        yield SimpleNamespace(
            addr=addr,
            prefill=prefill,
            decode=decode,
            prefill_buffers=prefill_buffers,
            decode_buffers=decode_buffers,
        )


def test_cpu_managers_hold_every_field_the_bare_double_sets(
    cpu_pd: SimpleNamespace,
) -> None:
    # Keeps _bare_manager honest: unit tests must not rely on invented state.
    # The double serves both roles, so compare with the union of the two.
    real_fields = set(vars(cpu_pd.prefill)) | set(vars(cpu_pd.decode))
    invented = set(vars(_bare_manager())) - real_fields
    assert not invented, invented


def test_cpu_decode_starts_prefill_heartbeat(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    started: list[tuple[CommonKVManager, threading.Thread]] = []
    start = CommonKVManager._start_heartbeat_checker_thread

    def record_start(mgr: CommonKVManager) -> threading.Thread:
        thread = start(mgr)
        started.append((mgr, thread))
        return thread

    monkeypatch.setattr(
        CommonKVManager, "_start_heartbeat_checker_thread", record_start
    )
    cpu_pd = request.getfixturevalue("cpu_pd")

    [(mgr, thread)] = started
    assert mgr is cpu_pd.decode
    assert thread.is_alive()


def test_cpu_segments_register_kv_and_state_in_vram_aux_in_dram(
    cpu_pd: SimpleNamespace,
) -> None:
    for mgr in (cpu_pd.prefill, cpu_pd.decode):
        placements = [
            {(seg.memory_type, seg.device_id) for seg in segments}
            for segments in (
                mgr.registered_kv_segments,
                [
                    seg
                    for component in mgr.registered_state_segments
                    for seg in component
                ],
                mgr.registered_aux_segments,
            )
        ]
        assert placements == [{("VRAM", 0)}, {("VRAM", 0)}, {("DRAM", -1)}]


def test_cpu_chunked_transfer_writes_only_requested_pages(
    cpu_pd: SimpleNamespace,
) -> None:
    room = 0xC0FFEE_301
    src_pages, dst_pages = [1, 2, 3, 5], [7, 6, 0, 4]
    untouched = sorted(set(range(NUM_PAGES)) - set(dst_pages))
    before = [tensor.clone() for tensor in cpu_pd.decode_buffers["kv_buffers"]]

    receiver = UniflowKVReceiver(cpu_pd.decode, cpu_pd.addr, room)
    receiver.init(0)
    sender = UniflowKVSender(cpu_pd.prefill, cpu_pd.addr, room, [0], 0)
    sender.init(len(src_pages), aux_index=2)
    receiver.send_metadata(
        np.array(dst_pages, dtype=np.int32), aux_index=1, state_indices=[[3]]
    )
    state = _poll_until(
        sender.poll, {KVPoll.WaitingForInput, KVPoll.Failed}, timeout=5.0
    )
    assert state == KVPoll.WaitingForInput
    sender.send(np.array(src_pages[:2], dtype=np.int32))
    sender.send(np.array(src_pages[2:], dtype=np.int32), [[0]])

    terminal = {KVPoll.Success, KVPoll.Failed}
    assert _poll_until(receiver.poll, terminal, timeout=10.0) == KVPoll.Success
    assert _poll_until(sender.poll, terminal, timeout=10.0) == KVPoll.Success
    prefill, decode = cpu_pd.prefill_buffers, cpu_pd.decode_buffers
    for src, dst, old in zip(
        prefill["kv_buffers"], decode["kv_buffers"], before, strict=True
    ):
        assert torch.equal(dst[dst_pages], src[src_pages])
        assert torch.equal(dst[untouched], old[untouched])
    assert torch.equal(decode["aux_buffer"][1], prefill["aux_buffer"][2])
    assert torch.equal(decode["state_buffer"][3], prefill["state_buffer"][0])


def test_cpu_decode_abort_before_sender_fails_sender(cpu_pd: SimpleNamespace) -> None:
    room = 0xC0FFEE_302
    receiver = UniflowKVReceiver(cpu_pd.decode, cpu_pd.addr, room)
    receiver.init(0)

    receiver.abort()

    assert _wait_until(lambda: room in cpu_pd.prefill.pending_aborts, 10.0)
    sender = UniflowKVSender(cpu_pd.prefill, cpu_pd.addr, room, [0], 0)
    assert sender.poll() == KVPoll.Failed
    with pytest.raises(KVTransferError, match="Aborted by AbortReq"):
        sender.failure_exception()


def test_cpu_bootstrap_timeout_during_worker_wait_leaks_nothing(
    cpu_pd: SimpleNamespace,
) -> None:
    room = 0xC0FFEE_303
    prefill = cpu_pd.prefill
    prefill.bootstrap_timeout = 5.0
    entered, done = threading.Event(), threading.Event()
    worker_errors: list[BaseException] = []
    wait_for_transfer_info = prefill.wait_for_transfer_info

    def observed_wait(wait_room: int) -> tuple[TransferInfo, DecodePeerInfo]:
        entered.set()
        try:
            return wait_for_transfer_info(wait_room)
        except BaseException as exc:
            worker_errors.append(exc)
            raise
        finally:
            done.set()

    prefill.wait_for_transfer_info = observed_wait
    sender = UniflowKVSender(prefill, cpu_pd.addr, room, [0], 0)
    sender.init(1, aux_index=0)
    sender.send(np.array([0], dtype=np.int32))
    assert entered.wait(5.0)

    sender.init_time -= 10
    assert sender.poll() == KVPoll.Failed
    with pytest.raises(KVTransferError, match=re.escape("KVPoll.Bootstrapping")):
        sender.failure_exception()

    # clear() notifies the worker, which stops waiting long before the
    # manager's own bootstrap timeout.
    assert done.wait(2.0)
    [error] = worker_errors
    assert isinstance(error, KVTransferError)
    assert "concluded before its transfer started" in error.failure_reason
    assert room not in prefill.failure_records
    assert room not in prefill.request_status


def test_cpu_abort_ack_releases_decode_pages(cpu_pd: SimpleNamespace) -> None:
    room = 0xC0FFEE_304
    assert cpu_pd.prefill.enable_deferred_decode_kv_release
    assert cpu_pd.decode.enable_deferred_decode_kv_release
    receiver = UniflowKVReceiver(cpu_pd.decode, cpu_pd.addr, room)
    receiver.init(0)
    receiver.send_metadata(np.array([0], dtype=np.int32))
    assert _wait_until(lambda: room in cpu_pd.prefill.transfer_infos, 10.0)

    # abort() itself arms the drain-ack tracker for a published receiver.
    receiver.abort()

    assert _wait_until(lambda: cpu_pd.decode.is_abort_release_safe(room, 1), 10.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
