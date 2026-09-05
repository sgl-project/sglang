"""Deferred decode-side KV release on the NIXL backend.

NIXL posts transfers asynchronously and the worker polls their handles. The
bootstrap thread records the ACK target while a transfer is outstanding; only
the worker can send that ACK after every handle reaches DONE.
"""

import unittest
from collections import defaultdict
from types import SimpleNamespace
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
    ABORT_TAG,
    AckTarget,
)
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOM = 11
DECODE_IP = "10.0.0.3"
DECODE_PORT = 6000


def _manager(
    *, enabled: bool = True, status: KVPoll | None = KVPoll.WaitingForInput
) -> NixlKVManager:
    manager = NixlKVManager.__new__(NixlKVManager)
    manager.enable_deferred_decode_kv_release = enabled
    manager._deferred_ack_targets = {}
    manager._deferred_ack_poisoned_rooms = set()
    manager._staging_outstanding = {}
    manager.request_status = {} if status is None else {ROOM: status}
    manager._sent = []
    manager._send_abort_ack = lambda *args: manager._sent.append(args)
    manager.record_failure = MagicMock()
    manager.update_status = MagicMock(
        side_effect=lambda room, value: manager.request_status.__setitem__(room, value)
    )
    manager.check_status = lambda room: manager.request_status[room]
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


class TestNixlAbortNotification(
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
        manager.enable_staging = False
        manager._staging_ctx = None
        manager._staging_outstanding = defaultdict(int)
        manager.transfer_infos = {ROOM: {}}
        manager.exceptions = {}
        manager.check_status = MagicMock(side_effect=RuntimeError("worker error"))
        queue = SimpleNamespace(
            get=MagicMock(side_effect=(_chunk(), KeyboardInterrupt()))
        )

        with patch("sglang.srt.disaggregation.nixl.conn.logger.exception"):
            with self.assertRaises(KeyboardInterrupt):
                manager.transfer_worker(queue)
        manager.check_status = lambda room: manager.request_status[room]

    def test_worker_skip_between_failure_and_target_registration_still_acks(self):
        manager = _manager()
        manager._staging_outstanding[ROOM] = 1
        update_status = manager.update_status.side_effect

        def fail_then_skip(room, status):
            update_status(room, status)
            # Window 1: the worker skips after Failed but before target
            # registration, so the bootstrap thread must observe zero and ACK.
            manager._staging_outstanding.pop(room, None)
            manager._maybe_ack_drained_abort(room)

        manager.update_status = MagicMock(side_effect=fail_then_skip)

        claimed = manager._handle_abort_notification(self._abort_message())

        self.assertTrue(claimed)
        self.assertEqual(
            manager._sent,
            [(ROOM, AckTarget(DECODE_IP, DECODE_PORT, ABORT_GENERATION))],
        )
        self.assertEqual(manager._deferred_ack_targets, {})

    def test_feature_off_marks_failed_without_ack(self):
        manager = _manager(enabled=False)

        claimed = manager._handle_abort_notification(self._abort_message())

        self.assertTrue(claimed)
        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        self.assertEqual(manager._deferred_ack_targets, {})
        self.assertEqual(manager._sent, [])

    def test_legacy_abort_without_return_address_is_tolerated(self):
        manager = _manager()

        claimed = manager._handle_abort_notification(
            [ABORT_TAG, str(ROOM).encode("ascii")]
        )

        self.assertTrue(claimed)
        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        self.assertEqual(manager._deferred_ack_targets, {})
        self.assertEqual(manager._sent, [])


if __name__ == "__main__":
    unittest.main()
