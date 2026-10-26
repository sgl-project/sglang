"""Deferred decode-side KV release on the Mori backend."""

import importlib
import sys
import threading
import time
import types
import unittest
from collections import defaultdict
from enum import IntEnum
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import torch


def _install_mori_stubs() -> None:
    try:
        importlib.import_module("mori.cpp")
        importlib.import_module("mori.io")
        return
    except (ImportError, OSError):
        pass

    mori = types.ModuleType("mori")
    mori.__path__ = []
    mori_cpp = types.ModuleType("mori.cpp")
    mori_io = types.ModuleType("mori.io")

    class FakeStatusCode(IntEnum):
        SUCCESS = 0
        IN_PROGRESS = 1
        FAILED = 2

    mori_cpp.TransferStatus = object
    for name in (
        "BackendType",
        "EngineDesc",
        "IOEngine",
        "IOEngineConfig",
        "MemoryDesc",
        "MemoryLocationType",
        "PollCqMode",
        "RdmaBackendConfig",
    ):
        setattr(mori_io, name, object)
    mori_io.StatusCode = FakeStatusCode
    mori.cpp = mori_cpp
    mori.io = mori_io
    sys.modules["mori"] = mori
    sys.modules["mori.cpp"] = mori_cpp
    sys.modules["mori.io"] = mori_io


_install_mori_stubs()

from test_deferred_decode_kv_release import (
    ABORT_GENERATION,
    DeferredAbortNotificationScenarios,
)

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.conn import (
    ABORT_TAG,
    AbortNotification,
    AckTarget,
)
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mori.conn import (
    FABRIC_CHUNK_ACK,
    FABRIC_CHUNK_NACK,
    FABRIC_CHUNK_READY,
    MoriKVManager,
    MoriKVReceiver,
    MoriKVSender,
    StatusCode,
    _MoriTransferSubmissionError,
    _SubmissionLocal,
)
from sglang.srt.disaggregation.mori.fabric_staging import (
    FabricSlotAcks,
    FabricStagingRoom,
    FabricStagingRx,
    default_slot_tokens,
    slot_bytes_for,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _manager(enabled: bool = True) -> MoriKVManager:
    manager = MoriKVManager.__new__(MoriKVManager)
    manager.enable_deferred_decode_kv_release = enabled
    manager.request_status = {}
    manager.transfer_lock = threading.Lock()
    manager._abort_ack_lock = threading.Lock()
    manager._deferred_ack_targets = {}
    manager._deferred_ack_poisoned_rooms = set()
    manager._deferred_abort_ack_tracker = {}
    manager._staging_outstanding = defaultdict(int)
    manager._drain_queue = Queue(maxsize=1)
    manager._submission_local = _SubmissionLocal()
    manager.req_to_decode_prefix_len = {}
    manager.transfer_infos = {}
    manager.failure_lock = threading.Lock()
    manager.failure_records = {}
    manager._sent = []
    manager._send_abort_ack = lambda *args: manager._sent.append(args)
    manager._wait_poll_ms = 0
    manager._transfer_timeout_ms = 0
    manager._mori_backend = "rdma"
    manager._fabric_staging = False
    return manager


def _abort_message(room: int = 11) -> list[bytes]:
    return AbortNotification(room, "10.0.0.3", 6000, ABORT_GENERATION).to_zmq()


def _chunk(room: int = 11) -> TransferKVChunk:
    return TransferKVChunk(
        room=room,
        prefill_kv_indices=np.array([1], dtype=np.int32),
        index_slice=slice(0, 1),
        is_last_chunk=False,
        prefill_aux_index=None,
        state_indices=None,
    )


class TestMoriAbortAck(DeferredAbortNotificationScenarios, CustomTestCase):
    room = 11
    decode_ip = "10.0.0.3"
    decode_port = 6000

    def _make_abort_manager(self, status: KVPoll | None):
        manager = _manager()
        if status is not None:
            manager.request_status[self.room] = status
        return manager

    def _dispatch_abort(self, manager) -> None:
        manager._handle_abort_message(self._abort_message())

    def _start_test_transfer(self, manager) -> None:
        manager._mark_transfer_started(_chunk(self.room))

    def _drain_test_transfer(self, manager) -> None:
        manager._mark_transfer_quiescent(_chunk(self.room))

    def test_abort_without_valid_return_address_marks_failed_without_ack(self):
        messages = (
            ("missing", [ABORT_TAG, b"11"]),
            ("malformed port", [ABORT_TAG, b"11", b"10.0.0.3", b"bad"]),
        )

        for name, message in messages:
            with self.subTest(name=name):
                manager = _manager()
                manager.request_status[11] = KVPoll.WaitingForInput

                manager._handle_abort_message(message)

                self.assertEqual(manager.request_status[11], KVPoll.Failed)
                self.assertEqual(manager._sent, [])
                self.assertEqual(manager._deferred_ack_targets, {})

    def test_feature_off_preserves_abort_without_ack(self):
        manager = _manager(enabled=False)
        manager.request_status[11] = KVPoll.WaitingForInput

        manager._handle_abort_message(_abort_message())

        self.assertEqual(manager.request_status[11], KVPoll.Failed)
        self.assertEqual(manager._sent, [])
        self.assertEqual(manager._deferred_ack_targets, {})

    def test_abort_after_submit_waits_for_terminal_status(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        status = MagicMock()
        status.InProgress.return_value = False
        manager._submit_kv_transfer = MagicMock(return_value=[status])
        wait_results = iter((StatusCode.IN_PROGRESS, StatusCode.SUCCESS))

        def wait_all(*args, **kwargs):
            result = next(wait_results)
            if result == StatusCode.IN_PROGRESS:
                manager._handle_abort_message(_abort_message())
                self.assertEqual(manager._sent, [])
            return result

        manager.engine = MagicMock()
        manager.engine.wait_all.side_effect = wait_all
        chunk = _chunk()
        manager._mark_transfer_started(chunk)

        is_quiescent = manager._process_transfer_chunk(chunk)

        self.assertTrue(is_quiescent)
        self.assertEqual(manager.engine.wait_all.call_count, 2)
        self.assertEqual(manager._sent, [])
        manager._mark_transfer_quiescent(chunk)
        self.assertEqual(
            manager._sent,
            [(11, AckTarget("10.0.0.3", 6000, ABORT_GENERATION))],
        )

    def test_sla_failure_acks_after_status_drains(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        manager._transfer_timeout_ms = 1
        manager.engine = MagicMock()
        manager.engine.wait_all.return_value = StatusCode.IN_PROGRESS
        status = MagicMock()
        status.InProgress.return_value = True
        manager._submit_kv_transfer = MagicMock(return_value=[status])
        manager.conclude_failure = MagicMock()
        chunk = _chunk()
        manager._mark_transfer_started(chunk)

        with patch(
            "sglang.srt.disaggregation.mori.conn.time.perf_counter",
            side_effect=(0.0, 0.002),
        ):
            is_quiescent = manager._process_transfer_chunk(chunk)

        self.assertFalse(is_quiescent)
        manager.engine.wait_all.assert_called_once()
        self.assertEqual(manager._sent, [])
        self.assertEqual(manager._staging_outstanding[11], 1)
        # Concluded at detection; only the ACK waits for the drain.
        manager.conclude_failure.assert_called_once_with(
            bootstrap_room=11, failure_reason="KV transfer exceeded SLA 1ms"
        )
        manager._handle_abort_message(_abort_message())
        queued_chunk, queued_statuses = manager._drain_queue.get_nowait()
        manager._drain_transfer_statuses(queued_chunk, queued_statuses)

        status.Wait.assert_called_once_with()
        self.assertEqual(
            manager._sent,
            [(11, AckTarget("10.0.0.3", 6000, ABORT_GENERATION))],
        )
        self.assertNotIn(11, manager._staging_outstanding)
        manager.conclude_failure.assert_called_once()

    def test_room_cleared_after_drain_handoff_keeps_single_owner(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        manager._transfer_timeout_ms = 1
        status = MagicMock()
        status.InProgress.return_value = True
        manager._submit_kv_transfer = MagicMock(return_value=[status])

        def wait_all(*args, **kwargs):
            manager.request_status.pop(11)
            return StatusCode.IN_PROGRESS

        manager.engine = MagicMock()
        manager.engine.wait_all.side_effect = wait_all
        manager.conclude_failure = MagicMock()
        queue = MagicMock()
        queue.get.side_effect = (_chunk(), KeyboardInterrupt())

        with patch(
            "sglang.srt.disaggregation.mori.conn.time.perf_counter",
            side_effect=(0.0, 0.002),
        ):
            with self.assertRaises(KeyboardInterrupt):
                manager._transfer_worker(queue, 0)

        self.assertEqual(manager._staging_outstanding[11], 1)
        self.assertEqual(manager._drain_queue.qsize(), 1)
        manager.conclude_failure.assert_called_once()

    def test_sla_failure_keeps_legacy_early_return_when_disabled(self):
        manager = _manager(enabled=False)
        manager._transfer_timeout_ms = 1
        manager.engine = MagicMock()
        manager.engine.wait_all.side_effect = (
            StatusCode.IN_PROGRESS,
            StatusCode.SUCCESS,
        )

        with patch(
            "sglang.srt.disaggregation.mori.conn.time.perf_counter",
            side_effect=(0.0, 0.002),
        ):
            failure = manager._wait_transfer_completion([object()])

        self.assertEqual(failure, "KV transfer exceeded SLA 1ms")
        self.assertEqual(manager.engine.wait_all.call_count, 1)

    def test_wait_event_failure_releases_outstanding_count(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        manager.conclude_failure = MagicMock()
        chunk = _chunk()
        chunk.wait_event = MagicMock()
        chunk.wait_event.synchronize.side_effect = RuntimeError("event failed")
        queue = MagicMock()
        queue.get.side_effect = (chunk, KeyboardInterrupt())

        with patch("sglang.srt.disaggregation.mori.conn.logger.exception"):
            with self.assertRaises(KeyboardInterrupt):
                manager._transfer_worker(queue, 0)

        self.assertNotIn(11, manager._staging_outstanding)
        manager.conclude_failure.assert_called_once()

    def test_wait_all_exception_keeps_status_for_draining(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        status = MagicMock()
        manager._submit_kv_transfer = MagicMock(return_value=[status])
        manager.engine = MagicMock()
        manager.engine.wait_all.side_effect = RuntimeError("wait failed")
        manager.conclude_failure = MagicMock()

        is_quiescent = manager._process_transfer_chunk(_chunk())

        self.assertFalse(is_quiescent)
        queued_chunk, queued_statuses = manager._drain_queue.get_nowait()
        self.assertEqual(queued_chunk.room, 11)
        self.assertEqual(queued_statuses, [status])
        manager.conclude_failure.assert_called_once_with(
            bootstrap_room=11,
            failure_reason="Transfer completion failed: RuntimeError('wait failed')",
        )

    def test_partial_submission_failure_transfers_ownership_to_drainer(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        status = MagicMock()

        def submit(*args, **kwargs):
            manager._record_submitted_statuses([status])
            raise RuntimeError("later target failed")

        manager._submit_kv_transfer = submit
        manager.conclude_failure = MagicMock()
        chunk = _chunk()
        manager._mark_transfer_started(chunk)

        with self.assertRaises(_MoriTransferSubmissionError) as raised:
            manager._process_transfer_chunk(chunk)
        manager._handle_submission_failure(chunk, raised.exception)

        queued_chunk, queued_statuses = manager._drain_queue.get_nowait()
        self.assertIs(queued_chunk, chunk)
        self.assertEqual(queued_statuses, [status])
        self.assertEqual(manager._staging_outstanding[11], 1)
        manager.conclude_failure.assert_called_once_with(
            bootstrap_room=11,
            failure_reason="Transfer submission failed: later target failed",
        )

    def test_drain_worker_retries_same_item_after_exception(self):
        manager = _manager()
        manager._wait_poll_ms = 0
        chunk = _chunk()
        status = MagicMock()
        manager._drain_queue = MagicMock()
        manager._drain_queue.get.side_effect = (
            (chunk, [status]),
            KeyboardInterrupt(),
        )
        manager._drain_transfer_statuses = MagicMock(
            side_effect=(RuntimeError("transient wait failure"), None)
        )

        with patch("sglang.srt.disaggregation.mori.conn.logger.exception") as log:
            with patch("sglang.srt.disaggregation.mori.conn.time.sleep"):
                with self.assertRaises(KeyboardInterrupt):
                    manager._drain_worker()

        self.assertEqual(manager._drain_transfer_statuses.call_count, 2)
        log.assert_called_once()


class TestMoriBackendOptIn(CustomTestCase):
    def test_mori_opts_into_deferred_release(self):
        # Mori sends the drain ack itself, so the default-on env var applies.
        self.assertTrue(MoriKVManager.supports_deferred_decode_kv_release)


class _FakeSlots:
    """CPU stand-in for FabricStagingSlots with a single slot."""

    def __init__(self, slot_bytes, storage=None):
        self.num_slots = 1
        self.slot_bytes = slot_bytes
        self.desc = object()
        self.storage = storage
        self.stream_mock = MagicMock()

    def offset(self, slot):
        if not 0 <= slot < self.num_slots:
            raise RuntimeError(f"Invalid Fabric staging slot {slot}")
        return 0

    def view(self, slot):
        self.offset(slot)
        return self.storage

    def stream(self, slot):
        return self.stream_mock


class TestMoriFabricStaging(CustomTestCase):
    def manager(self, local=6500, remote=5000):
        ns = types.SimpleNamespace
        manager = _manager()
        manager.kv_args = ns(
            page_size=256,
            kv_item_lens=[192, 64],
            kv_data_ptrs=[1, 2],
            kv_data_lens=[192 * 64, 64 * 64],
            state_types=[StateType.SWA_RING, StateType.DSV4_REQUEST_STATE],
            state_item_lens=[[3, 5], [48, 16]],
            state_data_ptrs=[[3, 4], [5, 6]],
            state_data_lens=[[3 * 128, 5 * 128], [48 * 8, 16 * 8]],
        )
        manager._fabric_staging = True
        manager._staging_slots = _FakeSlots(local)
        manager._staging_acks = FabricSlotAcks(1)
        manager._staging_rx = FabricStagingRx()
        manager._dcp_pack_max_tokens = None
        manager.is_dummy_cp_rank = False
        manager.enable_all_cp_ranks_for_transfer = False
        manager.attn_cp_size = 1
        manager.request_status[11] = KVPoll.WaitingForInput
        manager.transfer_infos = {
            11: {
                "dummy": ns(is_dummy=True),
                "decode": ns(
                    is_dummy=False, dst_state_indices=[np.arange(128), np.array([5, 6])]
                ),
            }
        }
        manager.decode_kv_args_table = {
            "decode": ns(
                dst_staging_desc=object(),
                dst_staging_slots=1,
                dst_staging_slot_bytes=remote,
                requires_dcp_relayout=False,
            )
        }
        manager._stopped = False
        return manager

    def sender(self, manager, pages):
        sender = MoriKVSender.__new__(MoriKVSender)
        sender.kv_mgr, sender.bootstrap_room, sender.curr_idx = manager, 11, 0
        sender._fabric_transfer_tokens = None
        sender._transfer_num_kv_indices = sender._transfer_num_state_indices = 0
        sender.init(pages, 3)
        return sender

    def test_slot_size_follows_chunk_tokens(self):
        kv_args = self.manager().kv_args
        mib = 1 << 20
        tokens_env = envs.SGLANG_MORI_FABRIC_STAGING_TOKENS
        state_env = envs.SGLANG_MORI_FABRIC_STAGING_STATE_MB
        with patch.object(state_env, "get", return_value=3):
            # 2048 tokens span 8 pages; one more covers a mid-page start.
            self.assertEqual(slot_bytes_for(kv_args, 2048), 3 * mib + mib)
            kv_args.kv_item_lens = [mib, mib]
            self.assertEqual(slot_bytes_for(kv_args, 2048), (9 * 2 + 3) * mib)
        server_args = types.SimpleNamespace(chunked_prefill_size=2048)
        with patch.object(tokens_env, "get", return_value=0):
            self.assertEqual(default_slot_tokens(server_args, 256), 2048)
            server_args.chunked_prefill_size = 1000
            self.assertEqual(default_slot_tokens(server_args, 256), 768)
            server_args.chunked_prefill_size = -1
            self.assertEqual(default_slot_tokens(server_args, 256), 8192)
        with patch.object(tokens_env, "get", return_value=512):
            self.assertEqual(default_slot_tokens(server_args, 256), 512)

    def test_cached_prefix_is_bounded_and_state_is_terminal(self):
        """A cached prefix larger than three slots must retain every page and state row."""
        ns = types.SimpleNamespace
        for local, remote in ((6500, 5000), (5000, 6500)):
            for early in (False, True):
                with self.subTest(local=local, remote=remote, early=early):
                    manager = self.manager(local, remote)
                    sender = self.sender(manager, 0)
                    cap = sender.get_max_transfer_tokens()
                    self.assertEqual(cap, 15 * 256)
                    cached, total = 4 * cap, 4 * cap + 519
                    sender.init((total + 255) // 256, 3)
                    manager.add_transfer_request = MagicMock()
                    pool = ns(
                        unified_swa_ring_size=128,
                        unified_swa_window=128,
                        request_state_transfer_indices=lambda *_: np.array([5, 6]),
                    )
                    req = ns(
                        rid="prefix",
                        kv=ns(req_pool_idx=0),
                        origin_input_ids=range(total),
                        extend_end=total,
                        start_send_idx=0,
                        disagg_decode_prefix_len=0,
                        disagg_kv_sender=sender,
                        prefix_len=cached,
                        host_hit_length=0,
                        pending_bootstrap=False,
                    )
                    scheduler = ns(
                        enable_staging=False,
                        enable_overlap=False,
                        token_to_kv_pool_allocator=ns(
                            page_size=256,
                            translate_kv_indices_for_transfer=lambda x: x,
                            get_kvcache=lambda: pool,
                        ),
                        req_to_token_pool=ns(
                            req_to_token=torch.arange(total).view(1, -1)
                        ),
                        disagg_metadata_buffers=MagicMock(),
                        kv_checksum_computer=None,
                        disagg_prefill_bootstrap_queue=ns(kv_manager=manager),
                        disagg_prefill_pending_chunk_rids=set(),
                    )
                    for name in ("send_kv_chunk", "_send_kv_chunk"):
                        setattr(
                            scheduler,
                            name,
                            types.MethodType(
                                getattr(SchedulerDisaggregationPrefillMixin, name),
                                scheduler,
                            ),
                        )
                    with patch.object(
                        envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX,
                        "get",
                        return_value=early,
                    ):
                        SchedulerDisaggregationPrefillMixin.maybe_send_cached_prefix_chunk(
                            scheduler, req
                        )
                    scheduler._send_kv_chunk(req, last_chunk=True)
                    calls = manager.add_transfer_request.call_args_list
                    self.assertEqual(len(calls), 5)
                    self.assertEqual(
                        sum(c.kwargs["num_kv_tokens"] for c in calls), total
                    )
                    np.testing.assert_array_equal(
                        np.concatenate([c.args[1] for c in calls]),
                        np.arange(sender.num_kv_indices),
                    )
                    offset = 0
                    for i, call in enumerate(calls):
                        pages = len(call.args[1])
                        self.assertEqual(call.args[2], slice(offset, offset + pages))
                        self.assertEqual(call.args[3], i == len(calls) - 1)
                        state = call.kwargs.get("state_indices")
                        self.assertEqual(state is not None, i == len(calls) - 1)
                        nbytes = pages * 256 + (1152 if state is not None else 0)
                        self.assertLessEqual(nbytes, min(local, remote))
                        self.assertLessEqual(call.kwargs["num_kv_tokens"], cap)
                        offset += pages
                    self.assertEqual([len(x) for x in state], [128, 2])
                    self.assertEqual(req.start_send_idx, total)

    def test_capacity_errors_and_non_fabric_fallback(self):
        for local, remote in ((0, 5000), (5000, 0), (1407, 6500)):
            with self.subTest(local=local, remote=remote):
                with self.assertRaises(ValueError):
                    self.sender(
                        self.manager(local, remote), 1
                    ).get_max_transfer_tokens()
        manager = self.manager()
        manager.decode_kv_args_table["decode"].dst_staging_desc = None
        with self.assertRaisesRegex(ValueError, "Missing decode staging slots"):
            self.sender(manager, 1).get_max_transfer_tokens()
        manager = self.manager()
        manager.transfer_infos[11]["decode"].dst_state_indices = []
        with self.assertRaisesRegex(ValueError, "state metadata mismatch"):
            self.sender(manager, 1).get_max_transfer_tokens()
        manager._fabric_staging = False
        self.assertIsNone(self.sender(manager, 1).get_max_transfer_tokens())
        manager._dcp_pack_max_tokens = 256
        manager.decode_kv_args_table["decode"].requires_dcp_relayout = True
        self.assertEqual(self.sender(manager, 1).get_max_transfer_tokens(), 256)

    def test_bad_pack_fails_before_launch_with_guards_intact(self):
        for rows, state, local, remote, error in (
            (np.arange(26), None, 6500, 7000, "overflow before gather"),
            (np.arange(20), None, 6500, 5000, "too small"),
            (np.array([-1]), None, 6500, 5000, "negative row"),
            (np.array([64]), None, 6500, 5000, "source overflow"),
            (
                np.array([0]),
                [np.array([128]), np.array([5, 6])],
                6500,
                5000,
                "source overflow",
            ),
        ):
            with self.subTest(error=error, rows=rows):
                manager = self.manager(local, remote)
                manager.transfer_infos[11].pop("dummy")
                storage = torch.full((local + 32,), 165, dtype=torch.uint8)
                manager._staging_slots = _FakeSlots(local, storage[16:-16])
                chunk = _chunk()
                chunk.prefill_kv_indices, chunk.state_indices = rows, state
                chunk.is_last_chunk = state is not None
                with patch(
                    "sglang.srt.disaggregation.mori.conn.pack_row_segments"
                ) as pack:
                    with self.assertRaisesRegex((ValueError, RuntimeError), error):
                        manager._process_fabric_staging_chunk(chunk, 0)
                    pack.assert_not_called()
                    self.assertTrue(torch.all(storage == 165))

    def ready_fixture(self):
        manager = self.manager()
        manager.attn_tp_rank = 0
        storage = torch.full((6532,), 165, dtype=torch.uint8)
        manager._staging_slots = _FakeSlots(6500, storage[16:-16])
        metadata = FabricStagingRoom(np.arange(6), [], {("10.0.0.1", 6000)})
        manager._staging_rx.rooms = {11: metadata}
        replies = []
        manager._connect_threadsafe = lambda *a, **kw: types.SimpleNamespace(
            send_multipart=replies.append
        )
        msg = [
            FABRIC_CHUNK_READY,
            b"11",
            b"0",
            b"2",
            b"0",
            b"512",
            b"0",
            b"10.0.0.1",
            b"6000",
            b"0",
            b"0",
            b"1",
            metadata.epoch,
        ]
        return manager, msg, replies, storage

    def test_ready_rejects_invalid_or_stale_identity_before_scatter(self):
        unpack_target = "sglang.srt.disaggregation.mori.conn.unpack_row_segments"
        for field, value in (
            (2, b"-1"),
            (3, b"99"),
            (5, b"6501"),
            (9, b"1"),
            (10, b"-1"),
            (11, b"0"),
            (12, b"old"),
        ):
            with self.subTest(field=field, value=value):
                manager, msg, replies, storage = self.ready_fixture()
                msg[field] = value
                with patch(unpack_target) as unpack:
                    manager._handle_fabric_chunk_ready(msg)
                self.assertEqual(
                    replies[0][:6],
                    [FABRIC_CHUNK_NACK, msg[1], msg[10], msg[11], msg[3], msg[9]],
                )
                self.assertEqual(replies[0][6], b"1")
                self.assertTrue(torch.all(storage == 165))
                unpack.assert_not_called()

        manager, msg, replies, _ = self.ready_fixture()
        with patch(unpack_target) as unpack:
            manager._handle_fabric_chunk_ready(msg)
            manager._handle_fabric_chunk_ready(msg)
            manager._staging_rx.rooms[11] = FabricStagingRoom(
                np.arange(6), [], {("10.0.0.1", 6000)}
            )
            msg[11] = b"2"
            manager._handle_fabric_chunk_ready(msg)
        self.assertEqual(
            [reply[0] for reply in replies],
            [FABRIC_CHUNK_ACK, FABRIC_CHUNK_NACK, FABRIC_CHUNK_NACK],
        )
        self.assertEqual(unpack.call_count, 1)

        manager, msg, replies, _ = self.ready_fixture()
        manager._staging_slots.stream_mock.synchronize.side_effect = RuntimeError(
            "stream failed"
        )
        with patch(unpack_target, side_effect=RuntimeError("copy failed")) as unpack:
            manager._handle_fabric_chunk_ready(msg)
            msg[11] = b"2"
            manager._handle_fabric_chunk_ready(msg)
        self.assertEqual(manager._staging_rx.poisoned, {0})
        self.assertEqual([reply[6] for reply in replies], [b"0", b"0"])
        self.assertEqual(unpack.call_count, 1)

    def test_abort_after_gather_stops_further_fabric_writes(self):
        """Cancellation must stop later writes and drain any already submitted."""
        for phase in ("gather", "between_targets"):
            with self.subTest(phase=phase):
                manager, _, _, storage = self.ready_fixture()
                info = manager.transfer_infos[11]["decode"]
                info.dst_staging_epoch = b"epoch"
                manager.transfer_infos[11] = {"decode": info, "second": info}
                peer = manager.decode_kv_args_table["decode"]
                peer.decode_tp_rank = 0
                manager.decode_kv_args_table["second"] = types.SimpleNamespace(
                    **{**vars(peer), "decode_tp_rank": 1}
                )
                status = MagicMock()
                manager.engine = MagicMock()

                def write(*args):
                    self.assertNotEqual(manager.request_status[11], KVPoll.Failed)
                    return [status]

                manager.engine.batch_write.side_effect = write

                def abort():
                    manager._handle_abort_message(_abort_message())

                def pack(*args):
                    if phase == "gather":
                        abort()
                    return 256

                def allocate_uid():
                    if manager.engine.batch_write.call_count:
                        abort()
                    return 1

                if phase != "gather":
                    manager.engine.allocate_transfer_uid.side_effect = allocate_uid
                chunk = _chunk()
                manager._mark_transfer_started(chunk)
                with (
                    patch(
                        "sglang.srt.disaggregation.mori.conn.pack_row_segments",
                        side_effect=pack,
                    ),
                    self.assertRaisesRegex(
                        _MoriTransferSubmissionError, "cancelled"
                    ) as raised,
                ):
                    manager._process_fabric_staging_chunk(chunk, 0)
                submitted = [] if phase == "gather" else [status]
                self.assertEqual(raised.exception.statuses, submitted)
                self.assertEqual(manager.engine.batch_write.call_count, len(submitted))
                self.assertEqual(manager.request_status[11], KVPoll.Failed)
                self.assertEqual(manager._sent, [])
                self.assertTrue(torch.all(storage == 165))
                with self.assertRaisesRegex(RuntimeError, "quarantined"):
                    manager._staging_acks.acquire(0)

    def test_failed_wait_quarantines_slot_until_terminal_reply(self):
        manager = self.manager()
        acks = manager._staging_acks
        key = (11, 0, 1, 2, 0)
        acks.expect(0, {key})

        def wait(deadline):
            acks.wait(
                {key},
                deadline,
                cancelled=lambda: manager._stopped or manager._should_skip_transfer(11),
            )

        with self.assertRaisesRegex(RuntimeError, "Timed out"):
            wait(time.monotonic() - 1)
        for state in ("timeout", "abort", "shutdown", "nack"):
            with self.subTest(state=state):
                manager.request_status[11] = KVPoll.Transferring
                manager._stopped = False
                if state == "abort":
                    manager.request_status[11] = KVPoll.Failed
                elif state == "shutdown":
                    manager._stopped = True
                elif state == "nack":
                    manager._handle_fabric_chunk_ack(
                        [
                            FABRIC_CHUNK_NACK,
                            b"11",
                            b"0",
                            b"1",
                            b"2",
                            b"0",
                            b"0",
                            b"scatter failed",
                        ]
                    )
                with self.assertRaisesRegex(RuntimeError, "quarantined"):
                    acks.acquire(0)
                if state != "timeout":
                    with self.assertRaises(RuntimeError):
                        wait(time.monotonic() + 300)
        manager._handle_fabric_chunk_ack(
            [FABRIC_CHUNK_ACK, b"11", b"0", b"2", b"2", b"0"]
        )
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            acks.acquire(0)
        manager._handle_fabric_chunk_ack(
            [FABRIC_CHUNK_NACK, b"11", b"0", b"1", b"2", b"0", b"1", b"rejected"]
        )
        acks.acquire(0)
        self.assertTrue(acks.is_idle())
        manager._handle_fabric_chunk_ack(
            [FABRIC_CHUNK_ACK, b"11", b"0", b"1", b"2", b"0"]
        )
        self.assertTrue(acks.is_idle())

    def test_clear_waits_for_scatter_and_rejects_late_ready(self):
        manager, msg, replies, _ = self.ready_fixture()
        manager.defer_decode_allocation = False
        manager.required_prefill_response_num_table = {}
        manager.prefill_response_tracker = {}
        manager.addr_to_rooms_tracker = defaultdict(set)
        receiver = MoriKVReceiver.__new__(MoriKVReceiver)
        receiver.kv_mgr, receiver.bootstrap_room, receiver.bootstrap_addr = (
            manager,
            11,
            "peer",
        )
        receiver._lifecycle_lock = manager._staging_rx.lock
        entered, finish, clearing, cleared = (threading.Event() for _ in range(4))

        def unpack(*args):
            entered.set()
            self.assertTrue(finish.wait(2))

        def clear():
            clearing.set()
            receiver.clear()
            cleared.set()

        with patch(
            "sglang.srt.disaggregation.mori.conn.unpack_row_segments",
            side_effect=unpack,
        ):
            worker = threading.Thread(
                target=manager._handle_fabric_chunk_ready, args=(msg,)
            )
            worker.start()
            self.assertTrue(entered.wait(1))
            cleanup = threading.Thread(target=clear)
            cleanup.start()
            try:
                self.assertTrue(clearing.wait(1))
                self.assertFalse(cleared.wait(0.02))
            finally:
                finish.set()
                worker.join(2)
                cleanup.join(2)
            self.assertTrue(cleared.is_set())
            manager._handle_fabric_chunk_ready(msg)
        self.assertNotIn(11, manager._staging_rx.rooms)
        self.assertEqual(
            [reply[0] for reply in replies], [FABRIC_CHUNK_ACK, FABRIC_CHUNK_NACK]
        )


if __name__ == "__main__":
    unittest.main()
