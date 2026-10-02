"""Deferred decode-side KV release on the NIXL backend.

When a decode request is aborted while its prefill->decode transfer may still be
in flight, the decode holds its KV pages until every prefill rank acks that its
transfer drained. NIXL transfers are asynchronous (agent.transfer() posts, the
worker polls check_xfer_state), so the ack must come from the transfer worker
after its DONE barrier -- never from the bootstrap thread for an active room.
"""

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _prefill_mgr(cls=CommonKVManager, enabled=True):
    """Bare manager carrying only the prefill-side deferred-ack state."""
    mgr = cls.__new__(cls)
    mgr.enable_deferred_decode_kv_release = enabled
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_fanout_snapshots = {}
    mgr._staging_outstanding = {}
    mgr.request_status = {}
    mgr.transfer_infos = {}
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

    def test_drain_ack_fans_out_to_every_room_peer(self):
        """With prefill TP < decode TP the room has several decode peers but the
        registry keeps only the last ABORT sender; the drain ack must reach every
        peer (dummy pairings included) or the others hold until the timeout."""
        mgr = _prefill_mgr()
        mgr.transfer_infos[7] = {
            "sess0": SimpleNamespace(endpoint="10.0.0.1", dst_port=5000),
            "sess1": SimpleNamespace(endpoint="10.0.0.2", dst_port=5001),
            # Dummy pairing: the decode rank still counts this prefill's ack.
            "sess2": SimpleNamespace(endpoint="10.0.0.3", dst_port=5002),
        }
        # Rank 1 registered last and overwrote rank 0's registration.
        mgr.register_deferred_ack_target(7, "10.0.0.2", 5001)

        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(
            sorted(mgr._sent),
            [
                ("10.0.0.1", 5000, 7),
                ("10.0.0.2", 5001, 7),
                ("10.0.0.3", 5002, 7),
            ],
        )

        # pop() semantics survive the fan-out: a second drain acks nobody.
        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(len(mgr._sent), 3)

    def test_drain_ack_fanout_survives_mid_flight_teardown(self):
        """The sender's clear() can pop transfer_infos while a chunk is still in
        flight; the peers snapshotted when the ABORT registered must still be
        acked when the worker finally drains, or they hold until the timeout."""
        mgr = _prefill_mgr()
        mgr.transfer_infos[7] = {
            "sess0": SimpleNamespace(endpoint="10.0.0.1", dst_port=5000),
            "sess1": SimpleNamespace(endpoint="10.0.0.2", dst_port=5001),
        }
        mgr._staging_outstanding[7] = 1
        mgr.register_deferred_ack_target(7, "10.0.0.2", 5001)
        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(mgr._sent, [])  # still writing -> held

        # Scheduler clears the sender mid-flight; the worker drains after.
        mgr.transfer_infos.clear()
        mgr._staging_outstanding[7] = 0
        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(
            sorted(mgr._sent),
            [("10.0.0.1", 5000, 7), ("10.0.0.2", 5001, 7)],
        )
        self.assertNotIn(7, mgr._deferred_ack_fanout_snapshots)

    def test_drain_ack_after_teardown_falls_back_to_registered_target(self):
        mgr = _prefill_mgr()
        mgr.register_deferred_ack_target(9, "10.0.0.4", 5003)
        mgr._maybe_ack_drained_abort(9)
        self.assertEqual(mgr._sent, [("10.0.0.4", 5003, 9)])

    def test_populated_room_without_registration_acks_nobody(self):
        """A normal (non-aborted) drain must not fan out spurious acks."""
        mgr = _prefill_mgr()
        mgr.transfer_infos[12] = {
            "sess0": SimpleNamespace(endpoint="10.0.0.5", dst_port=5004),
        }
        mgr._maybe_ack_drained_abort(12)
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


class _StopWorker(Exception):
    pass


class _OneChunkQueue:
    """Feeds the worker a single chunk, then unblocks it out of its loop."""

    def __init__(self, chunk):
        self._chunk = chunk
        self._served = False

    def get(self):
        if self._served:
            raise _StopWorker
        self._served = True
        return self._chunk


class _SettledFailureReq:
    """Transfer info whose first use raises a settled transport error."""

    room = 7
    is_dummy = False
    endpoint = "10.0.0.9"
    dst_port = 6009

    @property
    def agent_name(self):
        raise RuntimeError("NIXL transfer encountered ERR")


class TestNixlWorkerSettledFailureReleasesAck(CustomTestCase):
    """Bug regression: a settled transfer failure (every handle settled, decode
    told via conclude_failure) left the chunk counted in _staging_outstanding,
    so the abort ack for the room could never fire and the decode's deferred
    KV release always ran out the full timeout instead of draining."""

    def test_settled_failure_uncounts_chunk_and_releases_held_ack(self):
        mgr = _prefill_mgr(NixlKVManager)
        mgr._staging_outstanding = defaultdict(int)
        mgr.enable_staging = False
        mgr.exceptions = {}
        mgr.decode_kv_args_table = {}
        mgr.request_status[7] = KVPoll.WaitingForInput
        mgr.check_status = lambda r: mgr.request_status[r]
        mgr.update_status = lambda r, s: mgr.request_status.__setitem__(r, s)
        mgr.record_failure = MagicMock()
        mgr._await_handles = lambda handles, failure_seen=False, **_: (True, True)
        mgr.transfer_infos[7] = {"sess0": _SettledFailureReq()}
        # The decode learns of the failure and its ABORT lands while the chunk
        # is still counted -- the interleaving that held the ack forever.
        mgr.conclude_failure = MagicMock(
            side_effect=lambda **kw: mgr._handle_abort_notification(
                [b"ABORT", b"7", b"10.0.0.9", b"6009"]
            )
        )
        chunk = SimpleNamespace(room=7, staging_counted=False)

        with self.assertRaises(_StopWorker):
            mgr.transfer_worker(_OneChunkQueue(chunk))

        mgr.conclude_failure.assert_called_once()
        self.assertEqual(mgr._staging_outstanding[7], 0)
        self.assertEqual(mgr._sent, [("10.0.0.9", 6009, 7)])
        self.assertNotIn(7, mgr._deferred_ack_targets)


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
