"""Deferred decode-side KV release on the Mooncake backend.

The bootstrap thread records the ACK target while a transfer is outstanding.
Only the transfer worker can send that ACK after its write has drained.
"""

import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
from test_deferred_decode_kv_release import (
    DeferredAbortNotificationScenarios,
    TaggedAbortNotificationScenarios,
    WorkerFailureAbortScenarios,
)

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    ABORT_ACK_TAG,
    ABORT_TAG,
    AbortNotification,
)
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
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

    def test_generationless_abort_warns_and_falls_back_to_timeout(self):
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
        warning.assert_called_once()

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

    def test_feature_off_sends_legacy_ack(self):
        manager = _manager(enabled=False)

        claimed = manager._handle_abort_notification(self._abort_message())

        self.assertTrue(claimed)
        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        payload = manager._send_multipart_locked.call_args.args[1]
        self.assertEqual(payload, [ABORT_ACK_TAG, str(ROOM).encode("ascii")])


if __name__ == "__main__":
    unittest.main()
