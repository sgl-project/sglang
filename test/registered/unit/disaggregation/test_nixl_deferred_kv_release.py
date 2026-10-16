"""Deferred decode-side KV release on the NIXL backend.

NIXL posts transfers asynchronously and the worker polls their handles. The
bootstrap thread records the ACK target while a transfer is outstanding; only
the worker can send that ACK after every handle reaches DONE.
"""

import threading
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

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

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


class _NeverSettlingAgent:
    def __init__(self, entered: threading.Event):
        self._entered = entered

    def check_xfer_state(self, handle):
        self._entered.set()
        return "PROC"

    def check_remote_metadata(self, agent_name):
        return True


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
        # A handle that never settles may still be writing, so the chunk stays
        # counted and the room is poisoned instead of notified.
        manager.conclude_failure = MagicMock()
        manager._await_handles = MagicMock(return_value=(False, True))
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

    def test_settled_worker_failure_uncounts_chunk_and_acks(self):
        # Every handle settled, so the writes are over: decode is told via
        # conclude_failure, and its ABORT, which lands while the chunk is
        # still counted, must be ACKed once the worker uncounts the chunk.
        manager = _manager()
        manager.enable_staging = False
        manager._staging_ctx = None
        manager._staging_outstanding = defaultdict(int)
        manager.transfer_infos = {ROOM: {}}
        manager.exceptions = {}
        manager.check_status = MagicMock(side_effect=RuntimeError("worker error"))

        def notify_decode(**kwargs):
            manager.check_status = lambda room: manager.request_status[room]
            self._dispatch_abort(manager)

        manager.conclude_failure = MagicMock(side_effect=notify_decode)
        queue = SimpleNamespace(
            get=MagicMock(side_effect=(_chunk(), KeyboardInterrupt()))
        )

        with patch("sglang.srt.disaggregation.nixl.conn.logger.exception"):
            with self.assertRaises(KeyboardInterrupt):
                manager.transfer_worker(queue)

        manager.conclude_failure.assert_called_once()
        self.assertEqual(manager._staging_outstanding[ROOM], 0)
        self.assertNotIn(ROOM, manager._deferred_ack_poisoned_rooms)
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

    def test_abort_during_unsettled_transfer_does_not_park_worker(self):
        """A decode abort must not park a worker on a handle that never settles."""
        manager = _manager(status=KVPoll.WaitingForInput)
        manager.enable_staging = False
        manager._staging_ctx = None
        manager._staging_outstanding = defaultdict(int)
        manager.transfer_infos = {}
        manager.exceptions = {}
        manager.conclude_failure = MagicMock()

        req = SimpleNamespace(
            room=ROOM,
            is_dummy=False,
            agent_name="decode-a",
            dst_kv_indices=np.array([1], dtype=np.int32),
        )
        dst_info = SimpleNamespace(
            decode_tp_size=1,
            requires_dcp_relayout=False,
            dst_homogeneous_mem_kind="VRAM",
            kv_xfer_segments=None,
            dst_kv_ptrs=[0x1000],
            gpu_id=0,
        )
        manager.transfer_infos = {ROOM: {"decode-a": req}}
        manager.decode_kv_args_table = {"decode-a": dst_info}
        manager.kv_args = SimpleNamespace(kv_data_ptrs=[0x2000], page_size=1)
        manager.transfer_source_rank = 0
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = False
        manager.attn_tp_size = 1
        manager.send_kvcache = MagicMock(return_value="kv-handle")

        entered_wait = threading.Event()
        manager.agent = _NeverSettlingAgent(entered_wait)
        queue = SimpleNamespace(
            get=MagicMock(side_effect=(_chunk(), KeyboardInterrupt()))
        )
        worker_result = {}

        def run_worker():
            try:
                manager.transfer_worker(queue)
            except BaseException as exc:
                worker_result["exception"] = exc

        worker = threading.Thread(target=run_worker, daemon=True)
        with (
            patch("sglang.srt.disaggregation.nixl.conn.NIXL_ERR_SETTLE_TIMEOUT_S", 0.1),
            patch("sglang.srt.disaggregation.nixl.conn.NIXL_ERR_SETTLE_POLL_S", 0.001),
        ):
            worker.start()
            self.assertTrue(entered_wait.wait(timeout=2.0))
            self._dispatch_abort(manager)
            worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive(), "worker stayed parked after abort")
        self.assertIsInstance(worker_result.get("exception"), KeyboardInterrupt)
        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        self.assertEqual(manager._staging_outstanding[ROOM], 1)
        self.assertIn(ROOM, manager._deferred_ack_poisoned_rooms)
        self.assertEqual(manager._sent, [])


if __name__ == "__main__":
    unittest.main()
