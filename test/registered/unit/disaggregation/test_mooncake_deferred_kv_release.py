"""Deferred decode-side KV release on the Mooncake backend.

The bootstrap thread records the ACK target while a transfer is outstanding.
Only the transfer worker can send that ACK after its write has drained.
"""

import threading
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
from test_deferred_decode_kv_release import (
    ABORT_GENERATION,
    DeferredAbortNotificationScenarios,
    TaggedAbortNotificationScenarios,
    WorkerFailureAbortScenarios,
)

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    ABORT_ACK_TAG,
    ABORT_TAG,
    AbortNotification,
    AckTarget,
    CommonKVManager,
)
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, TransferInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

ROOM = 106
DECODE_IP = "10.0.0.2"
DECODE_PORT = 6000


def _manager(
    *, enabled: bool = True, status: KVPoll | None = KVPoll.WaitingForInput
) -> MooncakeKVManager:
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.enable_deferred_decode_kv_release = enabled
    manager.request_status = {} if status is None else {ROOM: status}
    manager._deferred_ack_targets = {}
    manager._deferred_ack_poisoned_rooms = set()
    manager._staging_outstanding = {}
    manager.check_status = lambda room: manager.request_status[room]
    manager.update_status = lambda room, value: manager.request_status.__setitem__(
        room, value
    )
    manager._sent = []
    manager._send_abort_ack = lambda *args: manager._sent.append(args)
    manager._send_multipart_locked = MagicMock()
    return manager


def _chunk() -> TransferKVChunk:
    return TransferKVChunk(
        room=ROOM,
        prefill_kv_indices=np.array([1], dtype=np.int32),
        index_slice=slice(0, 1),
        is_last_chunk=False,
        chunk_id=0,
        prefill_aux_index=None,
        state_indices=None,
    )


def _last_chunk() -> TransferKVChunk:
    return TransferKVChunk(
        room=ROOM,
        prefill_kv_indices=np.array([1], dtype=np.int32),
        index_slice=slice(0, 1),
        is_last_chunk=True,
        chunk_id=0,
        prefill_aux_index=0,
        state_indices=[[1]],
    )


class TestMooncakeAbortNotification(
    DeferredAbortNotificationScenarios,
    TaggedAbortNotificationScenarios,
    WorkerFailureAbortScenarios,
    CustomTestCase,
):
    room = ROOM
    decode_ip = DECODE_IP
    decode_port = DECODE_PORT

    def _make_abort_manager(self, status: KVPoll | None):
        return _manager(status=status)

    def _provoke_worker_failure(self, manager) -> None:
        manager.enable_trace = False
        manager.bootstrap_port = 8998
        manager._staging_outstanding = defaultdict(int)
        manager.check_status = MagicMock(side_effect=RuntimeError("worker error"))
        queue = MagicMock()
        queue.get.return_value = _chunk()

        with self.assertRaisesRegex(RuntimeError, "Transfer thread failed"):
            manager.transfer_worker(queue, MagicMock())
        manager.check_status = lambda room: manager.request_status[room]

    def test_generationless_abort_marks_failed_without_ack_or_warning(self):
        # A decode that did not arm (prealloc abort) sends no generation; that
        # is normal and must not warn about mixed versions.
        manager = _manager()
        message = AbortNotification(ROOM, DECODE_IP, DECODE_PORT).to_zmq()

        with patch(
            "sglang.srt.disaggregation.common.conn.logger.warning_once"
        ) as warning:
            claimed = manager._handle_abort_notification(message)

        self.assertTrue(claimed)
        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        self.assertEqual(manager._deferred_ack_targets, {})
        self.assertEqual(manager._sent, [])
        warning.assert_not_called()

    def test_abort_without_valid_return_address_still_marks_room_failed(self):
        messages = (
            ("missing", [ABORT_TAG, str(ROOM).encode("ascii")]),
            (
                "malformed port",
                [
                    ABORT_TAG,
                    str(ROOM).encode("ascii"),
                    DECODE_IP.encode("ascii"),
                    b"bad",
                ],
            ),
        )

        for name, message in messages:
            with self.subTest(name=name):
                manager = _manager()

                claimed = manager._handle_abort_notification(message)

                self.assertTrue(claimed)
                self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
                self.assertEqual(manager._deferred_ack_targets, {})
                self.assertEqual(manager._sent, [])

    def _prepare_worker(self, manager, num_dsts: int = 1) -> None:
        """Wire transfer_worker to num_dsts decode ranks that all need the room."""
        manager.enable_trace = False
        manager.enable_staging = False
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.bootstrap_port = 8998
        manager.attn_tp_rank = manager.pp_rank = manager.attn_cp_rank = 0
        manager.pp_size = manager.attn_cp_size = 1
        manager._staging_outstanding = defaultdict(int)
        manager.session_lock = threading.Lock()
        manager.session_failures = defaultdict(int)
        manager.failed_sessions = set()
        manager.state_layout_rejections = {}
        manager.req_to_decode_prefix_len = {}
        manager.kv_args = MagicMock(kv_data_ptrs=[0x1000])
        manager._get_dsa_cache_transfer_skip_flags = lambda info: (False, False)
        manager.decode_kv_args_table = {}
        manager.transfer_infos = {ROOM: {}}
        for i in range(num_dsts):
            session = f"decode-session-{i}"
            manager.decode_kv_args_table[session] = MagicMock(
                requires_dcp_relayout=False
            )
            manager.transfer_infos[ROOM][(DECODE_IP, DECODE_PORT + i)] = TransferInfo(
                room=ROOM,
                endpoint=DECODE_IP,
                dst_port=DECODE_PORT + i,
                mooncake_session_id=session,
                dst_kv_indices=np.array([1], dtype=np.int32),
                dst_aux_index=0,
                dst_state_indices=[[1]],
                required_dst_info_num=num_dsts,
                is_dummy=False,
            )
        manager.send_kvcache = MagicMock(return_value=0)
        manager.maybe_send_extra = MagicMock(return_value=0)
        manager.send_aux = MagicMock(return_value=0)
        manager.conclude_failure = MagicMock(
            side_effect=lambda **kw: manager.update_status(
                kw["bootstrap_room"], KVPoll.Failed
            )
        )
        manager.conclude_transfer = MagicMock(
            side_effect=lambda **kw: manager.update_status(
                kw["bootstrap_room"], kw["status"]
            )
        )

    def _run_worker(self, manager, chunk: TransferKVChunk) -> None:
        queue = MagicMock()
        queue.get.side_effect = [chunk, None]
        manager.transfer_worker(queue, MagicMock())
        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        self.assertEqual(manager._staging_outstanding.get(ROOM, 0), 0)

    def _abort_on_call(self, manager):
        """A failed send whose ABORT lands while its writes may be in flight."""

        def send(*args, **kwargs):
            self._dispatch_abort(manager)
            return -1

        return send

    def _run_worker_with_failed_send(self, manager) -> None:
        """Run one chunk through transfer_worker with send_kvcache returning -1."""
        self._prepare_worker(manager)
        manager.send_kvcache.return_value = -1
        self._run_worker(manager, _chunk())
        manager.send_kvcache.assert_called_once()
        manager.conclude_failure.assert_called_once()

    def test_failed_send_never_acks_drained(self):
        # batch_transfer_sync can return -1 with the batch still writing, so a
        # failed send must leave decode to its release timeout, whether the
        # ABORT arrived mid-send or only after the Failed status.
        target = AckTarget(DECODE_IP, DECODE_PORT, ABORT_GENERATION)
        for name, abort_before_send in (("mid-send", True), ("after", False)):
            with self.subTest(name=name):
                manager = _manager()
                if abort_before_send:
                    manager._deferred_ack_targets[ROOM] = {
                        (target.ip, target.port): target
                    }

                self._run_worker_with_failed_send(manager)
                self.assertEqual(manager._sent, [])
                self.assertNotIn(ROOM, manager._deferred_ack_targets)

                self._dispatch_abort(manager)
                self.assertEqual(manager._sent, [])
                self.assertNotIn(ROOM, manager._deferred_ack_targets)

                # A new lifecycle of the same bootstrap_room acks again.
                manager.request_status.pop(ROOM, None)
                CommonKVManager.update_status(manager, ROOM, KVPoll.Bootstrapping)
                self._dispatch_abort(manager)
                self.assertEqual(manager._sent, [(ROOM, target)])

    def test_abort_during_failed_state_send_never_acks(self):
        """A state send that fails after an ABORT marked the room Failed must
        still withhold the drained ACK."""
        manager = _manager()
        self._prepare_worker(manager)
        manager.maybe_send_extra.side_effect = self._abort_on_call(manager)

        self._run_worker(manager, _last_chunk())

        self.assertEqual(manager._sent, [])
        self.assertIn(ROOM, manager._deferred_ack_poisoned_rooms)

    def test_failed_aux_to_one_decode_rank_never_acks(self):
        """An aux send that fails for the first decode rank must withhold the
        ACK even when the second rank stops the room before its own aux send."""
        for name, abort_during_aux in (("mid-send", True), ("dead-session", False)):
            with self.subTest(name=name):
                manager = _manager()
                self._prepare_worker(manager, num_dsts=2)
                if abort_during_aux:
                    manager.send_aux.side_effect = self._abort_on_call(manager)
                else:
                    manager.send_aux.return_value = -1
                    manager.failed_sessions.add("decode-session-1")

                self._run_worker(manager, _last_chunk())
                self._dispatch_abort(manager)

                manager.send_aux.assert_called_once()
                self.assertEqual(manager._sent, [])

    def test_feature_off_sends_legacy_ack(self):
        manager = _manager(enabled=False)

        claimed = manager._handle_abort_notification(self._abort_message())

        self.assertTrue(claimed)
        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        payload = manager._send_multipart_locked.call_args.args[1]
        self.assertEqual(payload, [ABORT_ACK_TAG, str(ROOM).encode("ascii")])


if __name__ == "__main__":
    unittest.main()
