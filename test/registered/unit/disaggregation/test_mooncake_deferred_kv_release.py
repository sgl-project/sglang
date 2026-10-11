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

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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

    def _run_worker_with_failed_send(self, manager) -> None:
        """Run one chunk through transfer_worker with send_kvcache returning -1."""
        session = "decode-session"
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
        manager.kv_args = MagicMock(kv_data_ptrs=[0x1000])
        manager.decode_kv_args_table = {session: MagicMock(requires_dcp_relayout=False)}
        manager._get_dsa_cache_transfer_skip_flags = lambda info: (False, False)
        manager.transfer_infos = {
            ROOM: {
                (DECODE_IP, DECODE_PORT): TransferInfo(
                    room=ROOM,
                    endpoint=DECODE_IP,
                    dst_port=DECODE_PORT,
                    mooncake_session_id=session,
                    dst_kv_indices=np.array([1], dtype=np.int32),
                    dst_aux_index=0,
                    dst_state_indices=[],
                    required_dst_info_num=1,
                    is_dummy=False,
                )
            }
        }
        manager.send_kvcache = MagicMock(return_value=-1)
        manager.conclude_failure = MagicMock(
            side_effect=lambda **kw: manager.update_status(
                kw["bootstrap_room"], KVPoll.Failed
            )
        )
        queue = MagicMock()
        queue.get.side_effect = [_chunk(), None]

        manager.transfer_worker(queue, MagicMock())

        manager.send_kvcache.assert_called_once()
        manager.conclude_failure.assert_called_once()
        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        self.assertEqual(manager._staging_outstanding[ROOM], 0)

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

    def test_feature_off_sends_legacy_ack(self):
        manager = _manager(enabled=False)

        claimed = manager._handle_abort_notification(self._abort_message())

        self.assertTrue(claimed)
        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        payload = manager._send_multipart_locked.call_args.args[1]
        self.assertEqual(payload, [ABORT_ACK_TAG, str(ROOM).encode("ascii")])


if __name__ == "__main__":
    unittest.main()
