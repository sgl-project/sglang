"""Unit tests for the deferred decode-side KV release mechanism.

When a decode request is aborted while its prefill->decode KV transfer may still
be in flight, the decode side holds its KV pages / req-slot instead of freeing
them immediately (which could let the still-in-flight write land on pages already
reused by another request). The pages are released once every prefill rank acks
that its transfer drained (CommonKVManager.is_abort_release_safe), or a timeout
fires. See DecodeTransferQueue.resolve_deferred_releases.
"""

import unittest
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    ABORT_ACK_TAG,
    ABORT_TAG,
    AbortAck,
    AbortNotification,
    AckTarget,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.fake.conn import FakeKVReceiver
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ABORT_GENERATION = 7


class AbortScenario(NamedTuple):
    name: str
    status: KVPoll | None
    outstanding: int
    expected_status: KVPoll | None


ABORT_SCENARIOS = (
    AbortScenario(
        name="active transfer",
        status=KVPoll.Transferring,
        outstanding=1,
        expected_status=KVPoll.Failed,
    ),
    # Window 2: no worker will revisit a room that is already quiescent.
    AbortScenario(
        name="active quiescent room",
        status=KVPoll.WaitingForInput,
        outstanding=0,
        expected_status=KVPoll.Failed,
    ),
    AbortScenario(
        name="completed transfer",
        status=KVPoll.Success,
        outstanding=1,
        expected_status=KVPoll.Success,
    ),
    AbortScenario(
        name="completed quiescent room",
        status=KVPoll.Success,
        outstanding=0,
        expected_status=KVPoll.Success,
    ),
    # clear() can remove request_status before a counted write drains.
    AbortScenario(
        name="untracked transfer",
        status=None,
        outstanding=1,
        expected_status=None,
    ),
    AbortScenario(
        name="untracked quiescent room",
        status=None,
        outstanding=0,
        expected_status=None,
    ),
)


def _make_manager():
    """A bare CommonKVManager carrying only the deferred-ack state the helpers
    touch (avoids the heavy real __init__)."""
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr._deferred_abort_ack_tracker = {}
    mgr._deferred_abort_generation = 0
    mgr.enable_deferred_decode_kv_release = True
    return mgr


def _make_prefill_manager():
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.enable_deferred_decode_kv_release = True
    mgr.request_status = {}
    mgr.req_to_decode_prefix_len = {}
    mgr.transfer_infos = {}
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    mgr._staging_outstanding = {}
    mgr._sent = []
    mgr._send_abort_ack = lambda *args: mgr._sent.append(args)
    return mgr


class _TestReceiver(CommonKVReceiver):
    def poll(self):
        raise NotImplementedError

    def failure_exception(self):
        raise NotImplementedError


class _TestSender(CommonKVSender):
    def poll(self):
        raise NotImplementedError

    def failure_exception(self):
        raise NotImplementedError


class DeferredAbortNotificationScenarios:
    room: int
    decode_ip: str
    decode_port: int

    def _make_abort_manager(self, status: KVPoll | None):
        raise NotImplementedError

    def _dispatch_abort(self, manager) -> None:
        claimed = manager._handle_abort_notification(self._abort_message())
        self.assertTrue(claimed)

    def _start_test_transfer(self, manager) -> None:
        manager._staging_outstanding[self.room] = 1

    def _drain_test_transfer(self, manager) -> None:
        manager._staging_outstanding[self.room] -= 1
        manager._maybe_ack_drained_abort(self.room)

    def _abort_message(self) -> list[bytes]:
        return AbortNotification(
            self.room,
            self.decode_ip,
            self.decode_port,
            ABORT_GENERATION,
        ).to_zmq()

    def test_deferred_ack_follows_room_status_and_outstanding_transfers(self):
        target = AckTarget(
            self.decode_ip,
            self.decode_port,
            ABORT_GENERATION,
        )
        for case in ABORT_SCENARIOS:
            with self.subTest(name=case.name):
                manager = self._make_abort_manager(case.status)
                if case.outstanding:
                    self._start_test_transfer(manager)

                self._dispatch_abort(manager)

                self.assertEqual(
                    manager.request_status.get(self.room), case.expected_status
                )
                if case.outstanding:
                    self.assertEqual(manager._deferred_ack_targets[self.room], target)
                    self.assertEqual(manager._sent, [])
                    self._drain_test_transfer(manager)
                self.assertNotIn(self.room, manager._deferred_ack_targets)
                self.assertEqual(manager._sent, [(self.room, target)])


class TaggedAbortNotificationScenarios:
    room: int

    def test_non_abort_message_is_not_claimed(self):
        manager = self._make_abort_manager(KVPoll.WaitingForInput)

        self.assertFalse(
            manager._handle_abort_notification(
                [b"STAGING_REQ", str(self.room).encode()]
            )
        )


class WorkerFailureAbortScenarios:
    room: int
    decode_ip: str
    decode_port: int

    def _provoke_worker_failure(self, manager) -> None:
        raise NotImplementedError

    def test_worker_exception_poison_is_reset_for_reused_room(self):
        target = AckTarget(
            self.decode_ip,
            self.decode_port,
            ABORT_GENERATION,
        )
        manager = self._make_abort_manager(KVPoll.WaitingForInput)
        manager._deferred_ack_targets[self.room] = target

        self._provoke_worker_failure(manager)

        self.assertEqual(manager._staging_outstanding[self.room], 1)
        self.assertNotIn(self.room, manager._deferred_ack_targets)

        self._dispatch_abort(manager)
        self.assertNotIn(self.room, manager._deferred_ack_targets)
        self.assertEqual(manager._sent, [])

        manager.request_status.pop(self.room, None)
        CommonKVManager.update_status(manager, self.room, KVPoll.Bootstrapping)
        self._dispatch_abort(manager)
        self.assertNotIn(self.room, manager._deferred_ack_targets)
        self.assertEqual(manager._sent, [(self.room, target)])


class TestAbortWireFormat(CustomTestCase):
    def test_abort_notification_round_trip(self):
        notification = AbortNotification(100, "10.0.0.1", 5000, 7)

        self.assertEqual(
            AbortNotification.from_zmq(notification.to_zmq()), notification
        )

    def test_legacy_abort_without_return_address(self):
        self.assertEqual(
            AbortNotification.from_zmq([ABORT_TAG, b"101"]),
            AbortNotification(room=101),
        )

    def test_malformed_abort_is_rejected(self):
        self.assertIsNone(AbortNotification.from_zmq([ABORT_TAG, b"bad-room"]))

    def test_malformed_abort_generation_is_dropped(self):
        notification = AbortNotification.from_zmq(
            [ABORT_TAG, b"101", b"10.0.0.1", b"5000", b"bad-generation"]
        )

        self.assertEqual(
            notification,
            AbortNotification(room=101, decode_ip="10.0.0.1", decode_port=5000),
        )

    def test_abort_ack_round_trip(self):
        ack = AbortAck(room=102, prefill_rank=3, generation=7)

        self.assertEqual(AbortAck.from_zmq(ack.to_zmq()), ack)


class TestCommonAbortAckDispatch(CustomTestCase):
    def test_abort_ack_message_is_aggregated(self):
        mgr = _make_manager()
        generation = mgr.register_deferred_abort_room(103)

        claimed = mgr.handle_abort_ack_message(AbortAck(103, 4, generation).to_zmq())

        self.assertTrue(claimed)
        self.assertEqual(mgr._deferred_abort_ack_tracker[103].prefill_ranks, {4})

    def test_non_ack_message_is_not_claimed(self):
        mgr = _make_manager()

        self.assertFalse(mgr.handle_abort_ack_message([b"STATUS", b"103", b"4"]))

    def test_malformed_abort_ack_is_ignored(self):
        mgr = _make_manager()
        mgr.register_deferred_abort_room(103)

        self.assertTrue(
            mgr.handle_abort_ack_message([ABORT_ACK_TAG, b"bad-room", b"4", b"1"])
        )
        self.assertFalse(mgr.is_abort_release_safe(103, required_acks=1))

    def test_generationless_abort_ack_warns_and_is_ignored(self):
        mgr = _make_manager()
        mgr.register_deferred_abort_room(103)

        with patch(
            "sglang.srt.disaggregation.common.conn.logger.warning_once"
        ) as warning:
            claimed = mgr.handle_abort_ack_message([ABORT_ACK_TAG, b"103", b"4"])

        self.assertTrue(claimed)
        self.assertFalse(mgr.is_abort_release_safe(103, required_acks=1))
        warning.assert_called_once()

    def test_stale_generation_ack_is_ignored_after_room_reuse(self):
        mgr = _make_manager()
        mgr.register_deferred_abort_room(103)
        mgr.clear_deferred_abort_state(103)
        mgr.register_deferred_abort_room(103)

        claimed = mgr.handle_abort_ack_message([ABORT_ACK_TAG, b"103", b"4", b"1"])

        self.assertTrue(claimed)
        self.assertFalse(mgr.is_abort_release_safe(103, required_acks=1))

        claimed = mgr.handle_abort_ack_message([ABORT_ACK_TAG, b"103", b"4", b"2"])

        self.assertTrue(claimed)
        self.assertTrue(mgr.is_abort_release_safe(103, required_acks=1))

    def test_abort_for_deferred_release_arms_before_abort(self):
        mgr = _make_manager()
        receiver = _TestReceiver.__new__(_TestReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = 104
        receiver.abort_notified = False
        observed = []
        receiver.abort = lambda: observed.append(104 in mgr._deferred_abort_ack_tracker)

        receiver.abort_for_deferred_release()

        self.assertEqual(observed, [True])

    def test_abort_for_deferred_release_skips_tracker_when_disabled(self):
        mgr = _make_manager()
        mgr.enable_deferred_decode_kv_release = False
        receiver = _TestReceiver.__new__(_TestReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = 105
        receiver.abort_notified = False
        receiver.abort = lambda: None

        receiver.abort_for_deferred_release()

        self.assertNotIn(105, mgr._deferred_abort_ack_tracker)

    def test_base_receiver_falls_back_to_plain_abort(self):
        receiver = FakeKVReceiver.__new__(FakeKVReceiver)
        receiver.conclude_state = None

        receiver.abort_for_deferred_release()

        self.assertEqual(receiver.conclude_state, KVPoll.Failed)


class TestDeferredAckTargets(CustomTestCase):
    def test_ack_held_until_outstanding_drains(self):
        mgr = _make_prefill_manager()
        target = AckTarget("10.0.0.1", 5000, 9)
        mgr.register_deferred_ack_target(7, target)

        mgr._staging_outstanding[7] = 1
        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(mgr._sent, [])

        mgr._staging_outstanding[7] = 0
        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(mgr._sent, [(7, target)])

    def test_ack_fires_at_most_once(self):
        mgr = _make_prefill_manager()
        target = AckTarget("10.0.0.2", 5001, 10)
        mgr.register_deferred_ack_target(8, target)

        mgr._maybe_ack_drained_abort(8)
        mgr._maybe_ack_drained_abort(8)

        self.assertEqual(mgr._sent, [(8, target)])
        self.assertNotIn(8, mgr._deferred_ack_targets)

    def test_unregistered_room_is_noop(self):
        mgr = _make_prefill_manager()

        mgr._maybe_ack_drained_abort(999)

        self.assertEqual(mgr._sent, [])

    def test_sender_clear_keeps_target_until_outstanding_transfer_drains(self):
        mgr = _make_prefill_manager()
        mgr.request_status = {7: KVPoll.Failed}
        target = AckTarget("10.0.0.1", 5000, 9)
        mgr.register_deferred_ack_target(7, target)
        mgr._staging_outstanding[7] = 1
        sender = _TestSender.__new__(_TestSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = 7

        sender.clear()

        self.assertEqual(mgr._deferred_ack_targets[7], target)
        mgr._staging_outstanding[7] = 0
        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(mgr._sent, [(7, target)])

    def test_sender_clear_discards_target_without_outstanding_transfer(self):
        mgr = _make_prefill_manager()
        mgr.request_status = {7: KVPoll.Failed}
        mgr.register_deferred_ack_target(7, AckTarget("10.0.0.1", 5000, 9))
        sender = _TestSender.__new__(_TestSender)
        sender.kv_mgr = mgr
        sender.bootstrap_room = 7

        sender.clear()

        self.assertNotIn(7, mgr._deferred_ack_targets)

    def test_prefill_unique_rank_formula(self):
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr.attn_tp_rank, mgr.pp_size, mgr.attn_cp_size = 2, 3, 4
        mgr.pp_rank, mgr.attn_cp_rank = 1, 3

        self.assertEqual(mgr._prefill_unique_rank(), 31)


class TestAbortAckAggregation(CustomTestCase):
    def test_release_safe_only_after_all_required_ranks_ack(self):
        mgr = _make_manager()
        room = 100
        generation = mgr.register_deferred_abort_room(room)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

        mgr.note_abort_ack(room, 0, generation)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

        mgr.note_abort_ack(room, 1, generation)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))

    def test_duplicate_rank_ack_does_not_over_count(self):
        mgr = _make_manager()
        room = 101
        generation = mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0, generation)
        mgr.note_abort_ack(room, 0, generation)  # same rank twice
        # Two acks arrived but from one rank: not safe for a 2-rank prefill.
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))

    def test_clear_deferred_abort_state(self):
        mgr = _make_manager()
        room = 103
        generation = mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0, generation)
        mgr.clear_deferred_abort_state(room)
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=1))

    def test_ack_before_register_is_dropped(self):
        # An ack for a room that isn't actively held must not be recorded (it
        # would otherwise pollute a later request reusing the same room).
        mgr = _make_manager()
        room = 104
        mgr.note_abort_ack(room, 0, 1)  # no register yet
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=1))

    def test_register_resets_stale_acks(self):
        mgr = _make_manager()
        room = 106
        generation = mgr.register_deferred_abort_room(room)
        mgr.note_abort_ack(room, 0, generation)
        mgr.note_abort_ack(room, 1, generation)
        self.assertTrue(mgr.is_abort_release_safe(room, required_acks=2))
        # Re-registering (a later reuse) wipes the prior acks.
        mgr.register_deferred_abort_room(room)
        self.assertFalse(mgr.is_abort_release_safe(room, required_acks=2))


class _FakeIdxAllocator:
    def __init__(self):
        self.freed = []

    def free(self, idx):
        self.freed.append(idx)


def _make_queue(timeout=30.0):
    q = DecodeTransferQueue.__new__(DecodeTransferQueue)
    q._deferred_releases = []
    q.deferred_kv_release_timeout = timeout
    q.enable_staging = False
    q.staging_handler = None
    q.tree_cache = object()
    q.metadata_buffers = SimpleNamespace(bootstrap_room={})
    q.req_to_metadata_buffer_idx_allocator = _FakeIdxAllocator()
    return q


def _make_decode_req(room, idx, mgr, n_prefill_ranks=1):
    receiver = SimpleNamespace(
        kv_mgr=mgr,
        # One entry per prefill rank the decode notified of the abort; its length
        # is the required drain-ack count (see DecodeTransferQueue._defer_release).
        bootstrap_infos=[{"rank": r} for r in range(n_prefill_ranks)],
        clear=lambda: None,
    )
    return SimpleNamespace(
        req=SimpleNamespace(bootstrap_room=room),
        kv_receiver=receiver,
        metadata_buffer_index=idx,
    )


class TestResolveDeferredReleases(CustomTestCase):
    def test_noop_when_nothing_deferred(self):
        q = _make_queue()
        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
        rel.assert_not_called()

    def test_holds_until_drained_then_releases(self):
        mgr = _make_manager()
        room, idx = 200, 7
        q = _make_queue()
        dreq = _make_decode_req(room, idx, mgr, n_prefill_ranks=2)
        # In production the room is armed in abort_request when the ABORT is
        # sent, before the scheduler defers here.
        generation = mgr.register_deferred_abort_room(room)
        q._defer_release(dreq)

        with patch.object(decode_mod, "release_kv_cache") as rel:
            # Not yet acked -> held, not released.
            q.resolve_deferred_releases()
            rel.assert_not_called()
            self.assertEqual(len(q._deferred_releases), 1)

            # One of two ranks acked -> still held.
            mgr.note_abort_ack(room, 0, generation)
            q.resolve_deferred_releases()
            rel.assert_not_called()
            self.assertEqual(len(q._deferred_releases), 1)

            # Both ranks acked -> released exactly once.
            mgr.note_abort_ack(room, 1, generation)
            q.resolve_deferred_releases()
            rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)

        # Held state fully cleaned up.
        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [idx])
        self.assertEqual(q.metadata_buffers.bootstrap_room[idx], 0)
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertIsNone(dreq.kv_receiver)

    def test_releases_on_timeout_without_ack(self):
        mgr = _make_manager()
        room, idx = 300, 3
        q = _make_queue(timeout=30.0)
        dreq = _make_decode_req(room, idx, mgr, n_prefill_ranks=1)
        # Force an already-expired deadline (no ack will ever arrive).
        q._deferred_releases.append((dreq, float("-inf"), idx, 1))

        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
            rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)

        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [idx])
        self.assertIsNone(dreq.kv_receiver)

    def test_failed_release_is_isolated_and_not_retried(self):
        # A raising _do_release must drop the entry (no double-free on retry) and
        # not brick resolve for the remaining entries or subsequent calls.
        mgr = _make_manager()
        q = _make_queue()
        good = _make_decode_req(700, 1, mgr)
        bad = _make_decode_req(701, 2, mgr)
        # Both already past deadline -> both selected for release.
        q._deferred_releases.append((bad, float("-inf"), 2, 1))
        q._deferred_releases.append((good, float("-inf"), 1, 1))

        calls = []

        def fake_release(req, tree_cache, is_insert):
            calls.append(req)
            if req is bad.req:
                raise RuntimeError("boom")

        with patch.object(decode_mod, "release_kv_cache", side_effect=fake_release):
            q.resolve_deferred_releases()  # must not raise
            # The good one still released despite the bad one throwing.
            self.assertIn(good.req, calls)
            # Nothing left held, and a second call is a clean no-op (no retry).
            self.assertEqual(q._deferred_releases, [])
            q.resolve_deferred_releases()

    def test_defer_release_records_deadline_and_idx(self):
        mgr = _make_manager()
        q = _make_queue(timeout=12.5)
        dreq = _make_decode_req(room=400, idx=9, mgr=mgr)
        q._defer_release(dreq)
        self.assertEqual(len(q._deferred_releases), 1)
        held_req, deadline, held_idx, required = q._deferred_releases[0]
        self.assertIs(held_req, dreq)
        self.assertEqual(held_idx, 9)
        self.assertIsInstance(deadline, float)


if __name__ == "__main__":
    unittest.main()
