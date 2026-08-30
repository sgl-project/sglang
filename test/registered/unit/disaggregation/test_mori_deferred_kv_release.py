"""Deferred decode-side KV release on the Mori backend."""

import importlib
import sys
import threading
import types
import unittest
from collections import defaultdict
from enum import IntEnum
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np


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

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mori.conn import (
    _TAG_ABORT,
    _TAG_ABORT_ACK,
    MoriKVManager,
    MoriKVReceiver,
    MoriKVSender,
    StatusCode,
    _MoriTransferSubmissionError,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _manager(enabled: bool = True) -> MoriKVManager:
    manager = MoriKVManager.__new__(MoriKVManager)
    manager.enable_deferred_decode_kv_release = enabled
    manager.request_status = {}
    manager.transfer_lock = threading.Lock()
    manager._abort_ack_lock = threading.Lock()
    manager._deferred_ack_targets = {}
    manager._deferred_abort_ack_tracker = {}
    manager._staging_outstanding = defaultdict(int)
    manager._drain_queue = Queue(maxsize=1)
    manager._submission_local = threading.local()
    manager.req_to_decode_prefix_len = {}
    manager.transfer_infos = {}
    manager._room_notify_lock = threading.Lock()
    manager._room_status_notified = {}
    manager.failure_lock = threading.Lock()
    manager.failure_records = {}
    manager._sent = []
    manager._send_abort_ack = lambda ip, port, room: manager._sent.append(
        (ip, port, room)
    )
    manager._wait_poll_ms = 0
    manager._transfer_timeout_ms = 0
    return manager


def _abort_message(room: int = 11) -> list[bytes]:
    return [_TAG_ABORT, str(room).encode(), b"10.0.0.3", b"6000"]


def _chunk(room: int = 11) -> TransferKVChunk:
    return TransferKVChunk(
        room=room,
        prefill_kv_indices=np.array([1], dtype=np.int32),
        index_slice=slice(0, 1),
        is_last_chunk=False,
        prefill_aux_index=None,
        state_indices=None,
    )


class TestMoriAbortAck(unittest.TestCase):
    def test_active_write_defers_ack_until_quiescent(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        chunk = _chunk()
        manager._mark_transfer_started(chunk)

        manager._handle_abort_message(_abort_message())

        self.assertEqual(manager.request_status[11], KVPoll.Failed)
        self.assertEqual(manager._sent, [])
        self.assertEqual(manager._deferred_ack_targets[11], ("10.0.0.3", 6000))

        manager._mark_transfer_quiescent(chunk)
        self.assertEqual(manager._sent, [("10.0.0.3", 6000, 11)])
        self.assertNotIn(11, manager._deferred_ack_targets)
        self.assertNotIn(11, manager._staging_outstanding)

    def test_idle_active_room_acks_immediately(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.WaitingForInput

        manager._handle_abort_message(_abort_message())

        self.assertEqual(manager.request_status[11], KVPoll.Failed)
        self.assertEqual(manager._sent, [("10.0.0.3", 6000, 11)])

    def test_completed_room_acks_immediately(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Success

        manager._handle_abort_message(_abort_message())

        self.assertEqual(manager.request_status[11], KVPoll.Success)
        self.assertEqual(manager._sent, [("10.0.0.3", 6000, 11)])

    def test_completed_room_with_outstanding_write_defers_ack(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Success
        chunk = _chunk()
        manager._mark_transfer_started(chunk)

        manager._handle_abort_message(_abort_message())

        self.assertEqual(manager._sent, [])
        self.assertEqual(manager._deferred_ack_targets[11], ("10.0.0.3", 6000))
        manager._mark_transfer_quiescent(chunk)
        self.assertEqual(manager._sent, [("10.0.0.3", 6000, 11)])

    def test_legacy_abort_without_return_address_only_marks_failed(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.WaitingForInput

        manager._handle_abort_message([_TAG_ABORT, b"11"])

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

    def test_invalid_return_address_still_marks_room_failed(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.WaitingForInput

        manager._handle_abort_message([_TAG_ABORT, b"11", b"10.0.0.3", b"bad"])

        self.assertEqual(manager.request_status[11], KVPoll.Failed)
        self.assertEqual(manager._sent, [])

    def test_abort_after_submit_waits_for_terminal_status(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        status = MagicMock()
        status.InProgress.return_value = False
        manager._submit_kv_transfer = MagicMock(return_value=([status], None))
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
        self.assertEqual(manager._sent, [("10.0.0.3", 6000, 11)])

    def test_sla_failure_acks_after_status_drains(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        manager._transfer_timeout_ms = 1
        manager.engine = MagicMock()
        manager.engine.wait_all.return_value = StatusCode.IN_PROGRESS
        status = MagicMock()
        status.InProgress.return_value = True
        manager._submit_kv_transfer = MagicMock(return_value=([status], None))
        manager._conclude_room_failure = MagicMock()
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
        manager._conclude_room_failure.assert_not_called()
        manager._handle_abort_message(_abort_message())
        queued_chunk, queued_statuses, failure_reason = (
            manager._drain_queue.get_nowait()
        )
        manager._drain_transfer_statuses(queued_chunk, queued_statuses, failure_reason)

        status.Wait.assert_called_once_with()
        self.assertEqual(manager._sent, [("10.0.0.3", 6000, 11)])
        self.assertNotIn(11, manager._staging_outstanding)
        manager._conclude_room_failure.assert_called_once_with(
            11, "KV transfer exceeded SLA 1ms"
        )

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

    def test_sender_clear_preserves_ack_target_until_write_drains(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Failed
        chunk = _chunk()
        manager._mark_transfer_started(chunk)
        manager._handle_abort_message(_abort_message())
        sender = MoriKVSender.__new__(MoriKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = 11

        sender.clear()

        self.assertEqual(manager._deferred_ack_targets[11], ("10.0.0.3", 6000))
        manager._mark_transfer_quiescent(chunk)
        self.assertEqual(manager._sent, [("10.0.0.3", 6000, 11)])

    def test_wait_event_failure_releases_outstanding_count(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        manager._conclude_room_failure = MagicMock()
        chunk = _chunk()
        chunk.wait_event = MagicMock()
        chunk.wait_event.synchronize.side_effect = RuntimeError("event failed")
        queue = MagicMock()
        queue.get.side_effect = (chunk, KeyboardInterrupt())

        with patch("sglang.srt.disaggregation.mori.conn.logger.exception"):
            with self.assertRaises(KeyboardInterrupt):
                manager._transfer_worker(queue)

        self.assertNotIn(11, manager._staging_outstanding)
        manager._conclude_room_failure.assert_called_once()

    def test_wait_all_exception_keeps_status_for_draining(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        status = MagicMock()
        manager._submit_kv_transfer = MagicMock(return_value=([status], None))
        manager.engine = MagicMock()
        manager.engine.wait_all.side_effect = RuntimeError("wait failed")
        manager._conclude_room_failure = MagicMock()

        is_quiescent = manager._process_transfer_chunk(_chunk())

        self.assertFalse(is_quiescent)
        queued_chunk, queued_statuses, failure_reason = (
            manager._drain_queue.get_nowait()
        )
        self.assertEqual(queued_chunk.room, 11)
        self.assertEqual(queued_statuses, [status])
        self.assertEqual(
            failure_reason, "Transfer completion failed: RuntimeError('wait failed')"
        )
        manager._conclude_room_failure.assert_not_called()

    def test_partial_submission_failure_transfers_ownership_to_drainer(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        status = MagicMock()

        def submit(*args, **kwargs):
            manager._record_submitted_statuses([status])
            raise RuntimeError("later target failed")

        manager._submit_kv_transfer = submit
        manager._conclude_room_failure = MagicMock()
        chunk = _chunk()
        manager._mark_transfer_started(chunk)

        with self.assertRaises(_MoriTransferSubmissionError) as raised:
            manager._process_transfer_chunk(chunk)
        manager._handle_submission_failure(chunk, raised.exception)

        queued_chunk, queued_statuses, failure_reason = (
            manager._drain_queue.get_nowait()
        )
        self.assertIs(queued_chunk, chunk)
        self.assertEqual(queued_statuses, [status])
        self.assertEqual(
            failure_reason,
            "Transfer submission failed: later target failed",
        )
        self.assertEqual(manager._staging_outstanding[11], 1)
        manager._conclude_room_failure.assert_not_called()

    def test_drain_worker_retries_same_item_after_exception(self):
        manager = _manager()
        manager._wait_poll_ms = 0
        chunk = _chunk()
        status = MagicMock()
        manager._drain_queue = MagicMock()
        manager._drain_queue.get.side_effect = (
            (chunk, [status], None),
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

    def test_failure_notification_precedes_ownership_release(self):
        manager = _manager()
        chunk = _chunk()
        status = MagicMock()
        status.InProgress.return_value = False
        manager._conclude_room_failure = MagicMock(
            side_effect=(RuntimeError("notification failed"), None)
        )
        manager._mark_transfer_quiescent = MagicMock()

        with self.assertRaises(RuntimeError):
            manager._drain_transfer_statuses(chunk, [status], "transfer failed")
        manager._mark_transfer_quiescent.assert_not_called()

        manager._drain_transfer_statuses(chunk, [status], "transfer failed")

        self.assertEqual(manager._conclude_room_failure.call_count, 2)
        manager._mark_transfer_quiescent.assert_called_once_with(chunk)


class TestMoriDecodeAbortAck(unittest.TestCase):
    def test_ack_is_recorded_for_held_room(self):
        manager = _manager()
        manager.register_deferred_abort_room(21)

        manager._handle_abort_ack_message([_TAG_ABORT_ACK, b"21", b"3"])

        self.assertTrue(manager.is_abort_release_safe(21, required_acks=1))
        self.assertEqual(manager._deferred_abort_ack_tracker[21], {3})

    def test_receiver_arms_tracker_before_sending_abort(self):
        manager = _manager()
        manager.request_status[21] = KVPoll.WaitingForInput
        receiver = MoriKVReceiver.__new__(MoriKVReceiver)
        receiver.kv_mgr = manager
        receiver.bootstrap_room = 21
        receiver.bootstrap_infos = [{"rank": 3}]
        receiver.abort_notified = False
        receiver.conclude_state = None
        receiver.init_time = 1.0
        receiver.metadata_published = True
        receiver.clear = MagicMock()

        with patch.object(
            CommonKVReceiver,
            "_send_abort_notification",
            side_effect=lambda: manager._handle_abort_ack_message(
                [_TAG_ABORT_ACK, b"21", b"3"]
            ),
        ):
            receiver.abort()
        manager.register_deferred_abort_room(21)

        self.assertEqual(manager._deferred_abort_ack_tracker[21], {3})
        receiver.clear.assert_called_once_with()

    def test_preallocation_abort_does_not_arm_deferred_tracker(self):
        manager = _manager()
        manager.request_status[21] = KVPoll.Bootstrapping
        receiver = MoriKVReceiver.__new__(MoriKVReceiver)
        receiver.kv_mgr = manager
        receiver.bootstrap_room = 21
        receiver.bootstrap_infos = [{"rank": 3}]
        receiver.abort_notified = False
        receiver.conclude_state = None
        receiver.init_time = None
        receiver.metadata_published = False
        receiver.clear = MagicMock()

        with patch.object(CommonKVReceiver, "_send_abort_notification"):
            receiver.abort()

        self.assertNotIn(21, manager._deferred_abort_ack_tracker)

    def test_malformed_ack_is_ignored(self):
        manager = _manager()
        manager.register_deferred_abort_room(21)

        manager._handle_abort_ack_message([_TAG_ABORT_ACK, b"bad", b"3"])

        self.assertFalse(manager.is_abort_release_safe(21, required_acks=1))


if __name__ == "__main__":
    unittest.main()
