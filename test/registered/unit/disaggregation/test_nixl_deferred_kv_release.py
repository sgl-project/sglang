"""Deferred decode-side KV release on the NIXL backend.

When a decode request is aborted while its prefill->decode transfer may still be
in flight, the decode holds its KV pages until every prefill rank acks that its
transfer drained. NIXL transfers are asynchronous (agent.transfer() posts, the
worker polls check_xfer_state), so the ack must come from the transfer worker
after its DONE barrier -- never from the bootstrap thread for an active room.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _prefill_mgr(cls=CommonKVManager, enabled=True):
    """Bare manager carrying only the prefill-side deferred-ack state."""
    mgr = cls.__new__(cls)
    mgr.enable_deferred_decode_kv_release = enabled
    mgr._deferred_ack_targets = {}
    mgr._staging_outstanding = {}
    mgr.request_status = {}
    mgr._sent = []
    # Capture acks instead of opening a socket.
    mgr._send_abort_ack = lambda ip, port, room: mgr._sent.append((ip, port, room))
    return mgr


class TestDeferredAckTargets(CustomTestCase):
    def test_ack_held_until_outstanding_drains(self):
        mgr = _prefill_mgr()
        mgr.register_deferred_ack_target(7, "10.0.0.1", 5000)

        mgr._staging_outstanding[7] = 1
        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(mgr._sent, [])  # still writing -> no ack

        mgr._staging_outstanding[7] = 0
        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(mgr._sent, [("10.0.0.1", 5000, 7)])

    def test_ack_fires_at_most_once(self):
        mgr = _prefill_mgr()
        mgr.register_deferred_ack_target(8, "10.0.0.2", 5001)
        mgr._maybe_ack_drained_abort(8)
        mgr._maybe_ack_drained_abort(8)
        self.assertEqual(len(mgr._sent), 1)
        self.assertNotIn(8, mgr._deferred_ack_targets)

    def test_unregistered_room_is_noop(self):
        mgr = _prefill_mgr()
        mgr._maybe_ack_drained_abort(999)
        self.assertEqual(mgr._sent, [])

    def test_prefill_unique_rank_matches_success_sync_formula(self):
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr.attn_tp_rank, mgr.pp_size, mgr.attn_cp_size = 2, 3, 4
        mgr.pp_rank, mgr.attn_cp_rank = 1, 3
        self.assertEqual(mgr._prefill_unique_rank(), 2 * (3 * 4) + 1 * 4 + 3)


class TestNixlAbortNotification(CustomTestCase):
    """_handle_abort_notification is the prefill bootstrap-thread entry point."""

    @staticmethod
    def _abort_msg(room=11, ip="10.0.0.3", port=6000):
        return [
            b"ABORT",
            str(room).encode("ascii"),
            ip.encode("ascii"),
            str(port).encode("ascii"),
        ]

    def _mgr(self, enabled=True, room=11, status=KVPoll.WaitingForInput):
        mgr = _prefill_mgr(NixlKVManager, enabled=enabled)
        if status is not None:
            mgr.request_status[room] = status
        mgr.record_failure = MagicMock()
        mgr.update_status = MagicMock(
            side_effect=lambda r, s: mgr.request_status.__setitem__(r, s)
        )
        mgr.check_status = lambda r: mgr.request_status[r]
        return mgr

    def test_in_flight_room_registers_target_and_does_not_ack_yet(self):
        # A counted chunk holds the ack: only the worker knows when it landed.
        mgr = self._mgr()
        mgr._staging_outstanding[11] = 1
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._deferred_ack_targets[11], ("10.0.0.3", 6000))
        self.assertEqual(mgr._sent, [])
        # Marked Failed first, so no new chunk can be enqueued for the room.
        self.assertEqual(mgr.request_status[11], KVPoll.Failed)

    def test_quiescent_active_room_acks_without_waiting_for_a_worker_visit(self):
        # Window 2: chunks already drained with none left to come, so the worker
        # never revisits the room -- acking here keeps it off the timeout path.
        mgr = self._mgr()
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._sent, [("10.0.0.3", 6000, 11)])
        self.assertEqual(mgr._deferred_ack_targets, {})

    def test_worker_skip_before_registration_still_acks(self):
        # Window 1: the worker can pass its skip point between the Failed flip
        # and registration; the ack attempt at registration covers that.
        mgr = self._mgr()
        mgr._staging_outstanding[11] = 1

        real_update = mgr.update_status.side_effect

        def failed_then_worker_skips(room, status):
            real_update(room, status)
            # Worker dequeues, sees Failed, uncounts, and finds no target yet.
            mgr._staging_outstanding.pop(room, None)
            mgr._maybe_ack_drained_abort(room)

        mgr.update_status = MagicMock(side_effect=failed_then_worker_skips)
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._sent, [("10.0.0.3", 6000, 11)])
        self.assertEqual(mgr._deferred_ack_targets, {})

    def test_concluded_room_acks_immediately(self):
        # Concluded and quiescent: ack straight away.
        mgr = self._mgr(status=None)
        mgr.check_status = lambda r: KVPoll.Success
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._sent, [("10.0.0.3", 6000, 11)])
        self.assertEqual(mgr._deferred_ack_targets, {})

    def test_cleared_room_with_outstanding_chunk_does_not_ack(self):
        # The ERR path abandons sibling handles that may still be writing and
        # leaves the chunk counted; clear() then drops the room. Acking on
        # "unknown room" alone would release decode pages under those writes.
        mgr = self._mgr(status=None)  # room absent == cleared/unknown
        mgr._staging_outstanding[11] = 1
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._sent, [])
        self.assertEqual(mgr._deferred_ack_targets, {})

    def test_feature_off_registers_nothing_and_acks_nothing(self):
        mgr = self._mgr(enabled=False)
        self.assertTrue(mgr._handle_abort_notification(self._abort_msg()))

        self.assertEqual(mgr._deferred_ack_targets, {})
        self.assertEqual(mgr._sent, [])
        # Legacy behavior preserved: the room is still failed.
        self.assertEqual(mgr.request_status[11], KVPoll.Failed)

    def test_legacy_two_frame_abort_is_tolerated(self):
        # Older peers send [ABORT, room] with no return address.
        mgr = self._mgr()
        self.assertTrue(mgr._handle_abort_notification([b"ABORT", b"11"]))
        self.assertEqual(mgr._deferred_ack_targets, {})
        self.assertEqual(mgr._sent, [])

    def test_non_abort_message_is_not_claimed(self):
        mgr = self._mgr()
        self.assertFalse(mgr._handle_abort_notification([b"STAGING_REQ", b"11"]))


class TestNixlDecodeAckIngest(CustomTestCase):
    def test_abort_ack_is_aggregated_per_rank(self):
        # Mirrors the decode listener thread's ABORT_ACK branch.
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr._deferred_abort_ack_tracker = {}
        mgr.register_deferred_abort_room(21)

        for rank in (b"0", b"1", b"1"):
            msg = [b"ABORT_ACK", b"21", rank]
            mgr.note_abort_ack(int(msg[1].decode()), int(msg[2].decode()))

        self.assertFalse(mgr.is_abort_release_safe(21, required_acks=3))
        self.assertTrue(mgr.is_abort_release_safe(21, required_acks=2))


if __name__ == "__main__":
    unittest.main()
