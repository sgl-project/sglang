"""Deferred decode-side KV release on the Mooncake backend.

The bootstrap thread records the ACK target while a transfer is outstanding.
Only the transfer worker can send that ACK after its write has drained.
"""

import unittest
from collections import defaultdict
from threading import Lock, RLock
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import zmq
from test_deferred_decode_kv_release import (
    DeferredAbortNotificationScenarios,
    TaggedAbortNotificationScenarios,
    WorkerFailureAbortScenarios,
)

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferDestination
from sglang.srt.disaggregation.common.conn import (
    ABORT_ACK_TAG,
    ABORT_TAG,
    AbortNotification,
    KVTransferError,
)
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    TransferInfo,
)
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

    def test_feature_off_sends_legacy_ack(self):
        manager = _manager(enabled=False)

        claimed = manager._handle_abort_notification(self._abort_message())

        self.assertTrue(claimed)
        self.assertEqual(manager.request_status[ROOM], KVPoll.Failed)
        payload = manager._send_multipart_locked.call_args.args[1]
        self.assertEqual(payload, [ABORT_ACK_TAG, str(ROOM).encode("ascii")])


class TestMooncakeHostReceiver(CustomTestCase):
    def _receiver(self):
        manager = _manager()
        manager.__dict__.update(
            attn_tp_size=2,
            dcp_size=1,
            enable_staging=False,
            local_ip=DECODE_IP,
            rank_port=DECODE_PORT,
            _deferred_abort_generation=0,
            _deferred_abort_ack_tracker={},
            failure_lock=Lock(),
            failure_records={},
            defer_decode_allocation=False,
            required_prefill_response_num_table={},
            prefill_response_tracker={},
            addr_to_rooms_tracker=defaultdict(set),
        )
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.__dict__.update(
            kv_mgr=manager,
            session_id="session",
            bootstrap_room=ROOM,
            bootstrap_addr="prefill",
            init_time=None,
            conclude_state=None,
            _host_destination_published=False,
            _abort_generation=None,
            _owns_bootstrap_room=True,
            _lifecycle_lock=RLock(),
            abort_notified=False,
            require_staging=False,
            required_dst_info_num=2,
            bootstrap_infos=[{"is_dummy": False}, {"is_dummy": True}],
            prefill_info=SimpleNamespace(
                supports_host_destination=True, attn_tp_size=2, attn_cp_size=1
            ),
        )
        receiver.socket = MagicMock()
        receiver._connect_to_bootstrap_server = lambda info: (receiver.socket, Lock())
        receiver.invalidate_cached_bootstrap_infos = MagicMock()
        return receiver

    def test_host_metadata_roundtrip_and_capability_fallback(self):
        receiver = self._receiver()
        for owner, field, value in (
            (receiver.prefill_info, "supports_host_destination", False),
            (receiver.prefill_info, "attn_tp_size", 1),
            (receiver.prefill_info, "attn_cp_size", 2),
            (receiver.kv_mgr, "dcp_size", 2),
            (receiver.kv_mgr, "enable_deferred_decode_kv_release", False),
            (receiver.kv_mgr, "supports_host_destination", False),
            (receiver, "require_staging", True),
        ):
            with self.subTest(field=field), patch.object(owner, field, value):
                self.assertFalse(receiver.supports_host_destination)
        self.assertTrue(receiver.supports_host_destination)
        receiver.send_metadata(
            np.array([3, 5], dtype=np.int32),
            7,
            [[9]],
            destination=KVTransferDestination.HOST,
        )
        for call, dummy in zip(
            receiver.socket.send_multipart.call_args_list, (False, True)
        ):
            info = TransferInfo.from_zmq(call.args[0])
            self.assertEqual(info.destination, KVTransferDestination.HOST)
            self.assertEqual(info.is_dummy, dummy)
            self.assertEqual(info.dst_state_indices, [] if dummy else [[9]])

    def test_partial_publication_keeps_early_and_late_drain_acks(self):
        for early_ack in (False, True):
            with self.subTest(early_ack=early_ack):
                receiver = self._receiver()
                manager = receiver.kv_mgr
                receiver.socket.send_multipart.side_effect = [None, zmq.ZMQError()]
                receiver.send_metadata(
                    np.array([3], dtype=np.int32),
                    7,
                    destination=KVTransferDestination.HOST,
                )
                self.assertIsNone(receiver.init_time)
                self.assertFalse(receiver.is_abort_release_safe())
                receiver.socket.send_multipart.side_effect = None
                receiver.ensure_abort_notified()
                generation = receiver._abort_generation
                self.assertIsNotNone(generation)
                manager.note_abort_ack(ROOM, 0, generation - 1)
                self.assertFalse(receiver.is_abort_release_safe())
                if early_ack:
                    manager.note_abort_ack(ROOM, 0, generation)
                with self.assertRaises(KVTransferError):
                    receiver.failure_exception()
                self.assertIn(ROOM, manager._deferred_abort_ack_tracker)
                self.assertIn(ROOM, manager.request_status)
                manager.note_abort_ack(ROOM, 1, generation)
                self.assertEqual(receiver.is_abort_release_safe(), early_ack)
                manager.note_abort_ack(ROOM, 0, generation)
                self.assertTrue(receiver.is_abort_release_safe())
                receiver.clear()
                self.assertNotIn(ROOM, manager._deferred_abort_ack_tracker)
                self.assertNotIn(ROOM, manager.request_status)


if __name__ == "__main__":
    unittest.main()
