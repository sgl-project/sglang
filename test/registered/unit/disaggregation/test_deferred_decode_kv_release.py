"""Unit tests for the deferred decode-side KV release mechanism.

When a decode request is aborted while its prefill->decode KV transfer may still
be in flight, the decode side holds its KV pages / req-slot instead of freeing
them immediately (which could let the still-in-flight write land on pages already
reused by another request). The pages are released once every prefill rank acks
that its transfer drained (CommonKVManager.is_abort_release_safe). Device
destinations may also release on timeout; host destinations require the ack.
All decode attention-TP ranks must agree on holding and releasing each request.
"""

import datetime
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock, call, patch

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation.base.conn import BaseKVManager, KVPoll
from sglang.srt.disaggregation.common.conn import (
    ABORT_ACK_TAG,
    AbortAck,
    AbortNotification,
    AckTarget,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.decode_hicache_mixin import HiCacheRestoreResult
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=37, suite="base-a-test-cpu")

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
                    self.assertEqual(
                        manager._deferred_ack_targets[self.room],
                        {(target.ip, target.port): target},
                    )
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
        manager._deferred_ack_targets[self.room] = {(target.ip, target.port): target}

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


class TestCommonAbortAckDispatch(CustomTestCase):
    def test_abort_ack_message_is_aggregated(self):
        mgr = _make_manager()
        generation = mgr.register_deferred_abort_room(103)

        claimed = mgr.handle_abort_ack_message(AbortAck(103, 4, generation).to_zmq())

        self.assertTrue(claimed)
        self.assertEqual(mgr._deferred_abort_ack_tracker[103].prefill_ranks, {4})

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

    def _make_notifying_receiver(self, mgr, room, init_time):
        receiver = _TestReceiver.__new__(_TestReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = room
        receiver.bootstrap_infos = [{"rank_ip": "10.0.0.2", "rank_port": 6000}]
        receiver.init_time = init_time
        receiver.abort_notified = False
        receiver.conclude_state = None
        receiver._abort_generation = None
        mgr.local_ip = "10.0.0.1"
        mgr.rank_port = 5000
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr.request_status = {room: KVPoll.WaitingForInput}
        sent = []

        class _Sock:
            def send_multipart(self, frames):
                sent.append(
                    (
                        room in mgr._deferred_abort_ack_tracker,
                        AbortNotification.from_zmq(frames),
                    )
                )

        receiver._connect_to_bootstrap_server = lambda info: (
            _Sock(),
            threading.Lock(),
        )
        return receiver, sent

    def test_abort_arms_tracker_before_sending(self):
        mgr = _make_manager()
        receiver, sent = self._make_notifying_receiver(mgr, 104, init_time=1.0)

        receiver.abort()

        [(armed, notification)] = sent
        self.assertTrue(armed)
        self.assertEqual(
            notification.generation,
            mgr._deferred_abort_ack_tracker[104].generation,
        )

    def test_waiting_timeout_abort_is_armed(self):
        mgr = _make_manager()
        mgr.waiting_timeout = 0
        receiver, sent = self._make_notifying_receiver(mgr, 105, init_time=1.0)
        receiver.invalidate_cached_bootstrap_infos = lambda: None

        self.assertEqual(receiver._check_waiting_timeout(), KVPoll.Failed)

        [(armed, notification)] = sent
        self.assertTrue(armed)
        self.assertIsNotNone(notification.generation)
        self.assertTrue(receiver.abort_notified)

    def test_abort_before_metadata_skips_tracker(self):
        mgr = _make_manager()
        receiver, sent = self._make_notifying_receiver(mgr, 106, init_time=None)

        receiver.abort()

        [(armed, notification)] = sent
        self.assertFalse(armed)
        self.assertIsNone(notification.generation)

    def test_abort_skips_tracker_when_disabled(self):
        mgr = _make_manager()
        mgr.enable_deferred_decode_kv_release = False
        receiver, sent = self._make_notifying_receiver(mgr, 107, init_time=1.0)

        receiver.abort()

        self.assertNotIn(107, mgr._deferred_abort_ack_tracker)
        self.assertIsNone(sent[0][1].generation)


class TestDeferredAckTargets(CustomTestCase):
    def test_target_registered_during_poison_never_acks(self):
        """A target that races past the poisoned check must not ack later."""
        mgr = _make_prefill_manager()
        mgr._staging_outstanding[7] = 1

        class _PoisonOnRead(dict):
            def get(self, *args):
                # The worker poisons between the check and the store.
                mgr.poison_deferred_ack_room(7)
                return dict.get(self, *args)

        mgr._deferred_ack_targets = _PoisonOnRead()
        mgr.register_deferred_ack_target(7, AckTarget("10.0.0.1", 5000, 9))
        mgr._staging_outstanding[7] = 0
        mgr._maybe_ack_drained_abort(7)

        self.assertEqual(mgr._sent, [])
        self.assertNotIn(7, mgr._deferred_ack_targets)

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

        self.assertEqual(
            mgr._deferred_ack_targets[7], {(target.ip, target.port): target}
        )
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

    def test_drain_ack_fans_out_to_every_aborting_decode_rank(self):
        # Prefill TP < decode TP: two decode ranks share the room and each
        # sends its own ABORT. Neither may overwrite the other's target.
        mgr = _make_prefill_manager()
        rank0 = AckTarget("10.0.0.1", 5000, 3)
        rank1 = AckTarget("10.0.0.2", 5001, 8)
        mgr._staging_outstanding[7] = 1
        mgr.register_deferred_ack_target(7, rank0)
        mgr.register_deferred_ack_target(7, rank1)

        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(mgr._sent, [])

        mgr._staging_outstanding[7] = 0
        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(sorted(mgr._sent), [(7, rank0), (7, rank1)])
        mgr._maybe_ack_drained_abort(7)
        self.assertEqual(len(mgr._sent), 2)


class _BareReceiver(CommonKVReceiver):
    """Concrete shell: the ABC check blocks CommonKVReceiver.__new__."""

    def poll(self):
        raise NotImplementedError

    def failure_exception(self):
        raise NotImplementedError


class TestAbortArmsTrackerBeforeSend(CustomTestCase):
    """Prefill can ack the moment the ABORT lands (already-drained room), and a
    peer rank's earlier abort of the same room can fan an ack out even sooner;
    an ack arriving before the tracker is armed is dropped and the rank waits
    out the full release timeout. So the receiver must arm BEFORE sending."""

    def _abort_receiver(self, mgr, init_time):
        recv = _BareReceiver.__new__(_BareReceiver)
        recv.kv_mgr = mgr
        recv.bootstrap_room = 500
        recv.init_time = init_time
        recv.abort_notified = False
        recv._abort_generation = None
        recv.bootstrap_infos = [{"rank_ip": "10.0.0.9", "rank_port": 7000}]
        armed_at_send = []
        sock = SimpleNamespace(
            send_multipart=lambda parts: armed_at_send.append(
                500 in mgr._deferred_abort_ack_tracker
            )
        )
        recv._connect_to_bootstrap_server = lambda info: (sock, threading.Lock())
        return recv, armed_at_send

    def _make_decode_manager(self, enabled=True):
        mgr = _make_manager()
        mgr.enable_deferred_decode_kv_release = enabled
        mgr.local_ip, mgr.rank_port = "10.0.0.1", 6000
        return mgr

    def test_force_arm_arms_unpublished_receiver_before_send(self):
        # Bug regression: a send_metadata that failed partway leaves init_time
        # None while earlier ranks already hold destinations, so the transfer
        # queue's notify-and-defer must still arm (before the send) -- else
        # every drain ack is dropped and the hold runs out the full timeout.
        mgr = self._make_decode_manager()
        recv, armed_at_send = self._abort_receiver(mgr, init_time=None)
        recv.ensure_abort_notified(force_arm=True)
        self.assertEqual(armed_at_send, [True])

    def test_ensure_abort_notified_sends_once_and_keeps_failure_records(self):
        # For failures decode did not initiate the true root cause is already
        # recorded; notifying must not overwrite it the way abort() does.
        mgr = self._make_decode_manager()
        mgr.record_failure = lambda *a, **k: self.fail(
            "ensure_abort_notified must not touch failure records"
        )
        recv, armed_at_send = self._abort_receiver(mgr, init_time=123.0)
        recv.ensure_abort_notified()
        recv.ensure_abort_notified()
        self.assertEqual(armed_at_send, [True])  # one send, tracker armed first
        self.assertTrue(recv.abort_notified)


class _FakeIdxAllocator:
    def __init__(self):
        self.freed = []

    def free(self, idx):
        self.freed.append(idx)


def _make_queue(timeout=30.0, enable_metrics=False):
    q = DecodeTransferQueue.__new__(DecodeTransferQueue)
    q._deferred_releases = []
    q.queue = []
    q.gloo_group = dist.group.WORLD
    q.tp_rank = 0
    q.enable_deferred_kv_release = True
    q.deferred_kv_release_timeout = timeout
    q.enable_host_receive = False
    q.enable_staging = False
    q.staging_handler = None
    q.tree_cache = MagicMock()
    q.metadata_buffers = SimpleNamespace(bootstrap_room={})
    q.req_to_metadata_buffer_idx_allocator = _FakeIdxAllocator()
    q.scheduler = SimpleNamespace(
        enable_decode_hicache=False,
        enable_hisparse=False,
        output_streamer=MagicMock(),
        metrics_reporter=SimpleNamespace(enable_metrics=enable_metrics),
        metrics_collector=SimpleNamespace(
            observe_decode_deferred_kv_release=MagicMock()
        ),
    )
    return q


def _make_decode_req(room, idx, mgr, n_prefill_ranks=1):
    receiver = SimpleNamespace(
        kv_mgr=mgr,
        abort_notified=True,
        ensure_abort_notified=MagicMock(),
        failure_exception=lambda: None,
        poll=lambda: KVPoll.Failed,
        # One entry per prefill rank the decode notified of the abort; its length
        # is the required drain-ack count (see DecodeTransferQueue._defer_release).
        bootstrap_infos=[{"rank": r} for r in range(n_prefill_ranks)],
        is_abort_release_safe=lambda: mgr.is_abort_release_safe(room, n_prefill_ranks),
        clear=lambda: None,
    )
    return SimpleNamespace(
        req=SimpleNamespace(bootstrap_room=room, rid=str(room), return_logprob=False),
        host_staged=False,
        hicache_restore_status=HiCacheRestoreResult.READY,
        prefix_match=None,
        hicache_restore_lock=None,
        kv_receiver=receiver,
        metadata_buffer_index=idx,
    )


class TestResolveDeferredReleases(CustomTestCase):
    def setUp(self):
        super().setUp()
        world_size = patch.object(dist, "get_world_size", return_value=1)
        world_size.start()
        self.addCleanup(world_size.stop)

    def test_host_release_requires_drain_on_every_rank_even_after_timeout(self):
        mgr = _make_manager()
        q = _make_queue(timeout=-1)
        q.enable_host_receive = True
        q.gloo_group = object()
        entries = [_make_decode_req(room, room, mgr) for room in (1, 2)]
        generations = {}
        for entry in entries:
            entry.host_staged = True
            room = entry.req.bootstrap_room
            generations[room] = mgr.register_deferred_abort_room(room)
            q._defer_release(entry)
        with (
            patch.object(decode_mod, "discard_kv_cache_backup") as discard,
            patch.object(decode_mod, "release_kv_cache") as device_release,
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_reduce") as reduce,
        ):
            q.resolve_deferred_releases()
            discard.assert_not_called()
            mgr.note_abort_ack(2, 0, generations[2])
            reduce.side_effect = lambda ready, **_: ready.zero_()
            q.resolve_deferred_releases()
            discard.assert_not_called()
            reduce.side_effect = None
            q.resolve_deferred_releases()
            discard.assert_called_once_with(entries[1].req, q.tree_cache, "host_pool")
            self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [2])
            mgr.note_abort_ack(1, 0, generations[1])
            q.resolve_deferred_releases()
            self.assertEqual(discard.call_count, 2)
            self.assertEqual(q._deferred_releases, [])
            device_release.assert_not_called()

    def test_holds_until_drained_then_releases(self):
        mgr = _make_manager()
        room, idx = 200, 7
        q = _make_queue(enable_metrics=True)
        dreq = _make_decode_req(room, idx, mgr, n_prefill_ranks=2)
        # In production the receiver arms the room just before it sends the
        # ABORT (_send_abort_notification), before the scheduler defers here.
        generation = mgr.register_deferred_abort_room(room)

        with patch.object(
            decode_mod.time, "monotonic", side_effect=[10.0, 11.0, 12.0, 13.0]
        ):
            q._defer_release(dreq)
            with patch.object(decode_mod, "release_kv_cache") as rel:
                # Not yet acked -> held, not released.
                q.resolve_deferred_releases()
                rel.assert_not_called()
                self.assertEqual(len(q._deferred_releases), 1)
                q.scheduler.metrics_collector.observe_decode_deferred_kv_release.assert_not_called()

                # One of two ranks acked -> still held.
                mgr.note_abort_ack(room, 0, generation)
                q.resolve_deferred_releases()
                rel.assert_not_called()
                self.assertEqual(len(q._deferred_releases), 1)

                # Both ranks acked -> released exactly once.
                mgr.note_abort_ack(room, 1, generation)
                q.resolve_deferred_releases()
                rel.assert_called_once_with(dreq.req, q.tree_cache, checkpoint=False)

        # Held state fully cleaned up.
        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(q.num_pending_deferred_releases(), 0)
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [idx])
        self.assertEqual(q.metadata_buffers.bootstrap_room[idx], 0)
        self.assertNotIn(room, mgr._deferred_abort_ack_tracker)
        self.assertIsNone(dreq.kv_receiver)
        q.scheduler.metrics_collector.observe_decode_deferred_kv_release.assert_called_once_with(
            duration_seconds=3.0,
            outcome="drained",
        )

    def test_releases_on_timeout_without_ack(self):
        mgr = _make_manager()
        room, idx = 300, 3
        q = _make_queue(timeout=30.0, enable_metrics=True)
        dreq = _make_decode_req(room, idx, mgr, n_prefill_ranks=1)
        # Force an already-expired deadline (no ack will ever arrive).
        q._deferred_releases.append((dreq, 10.0, float("-inf"), idx, 1))

        with (
            patch.object(decode_mod.time, "monotonic", return_value=42.0),
            patch.object(decode_mod, "release_kv_cache") as rel,
        ):
            q.resolve_deferred_releases()
            rel.assert_called_once_with(dreq.req, q.tree_cache, checkpoint=False)

        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [idx])
        self.assertIsNone(dreq.kv_receiver)
        q.scheduler.metrics_collector.observe_decode_deferred_kv_release.assert_called_once_with(
            duration_seconds=32.0,
            outcome="timeout",
        )

    def test_failed_release_is_isolated_and_not_retried(self):
        # A raising _do_release must drop the entry (no double-free on retry) and
        # not brick resolve for the remaining entries or subsequent calls.
        mgr = _make_manager()
        q = _make_queue(enable_metrics=True)
        good = _make_decode_req(700, 1, mgr)
        bad = _make_decode_req(701, 2, mgr)
        # Both already past deadline -> both selected for release.
        q._deferred_releases.append((bad, 10.0, float("-inf"), 2, 1))
        q._deferred_releases.append((good, 10.0, float("-inf"), 1, 1))

        calls = []

        def fake_release(req, tree_cache, checkpoint):
            calls.append(req)
            if req is bad.req:
                raise RuntimeError("boom")

        with (
            patch.object(decode_mod.time, "monotonic", return_value=40.0),
            patch.object(decode_mod, "release_kv_cache", side_effect=fake_release),
        ):
            q.resolve_deferred_releases()  # must not raise
            # The good one still released despite the bad one throwing.
            self.assertIn(good.req, calls)
            # Nothing left held, and a second call is a clean no-op (no retry).
            self.assertEqual(q._deferred_releases, [])
            q.resolve_deferred_releases()
        q.scheduler.metrics_collector.observe_decode_deferred_kv_release.assert_has_calls(
            [
                call(duration_seconds=30.0, outcome="error"),
                call(duration_seconds=30.0, outcome="timeout"),
            ]
        )

    def test_defer_release_records_deadline_and_idx(self):
        mgr = _make_manager()
        q = _make_queue(timeout=12.5)
        dreq = _make_decode_req(room=400, idx=9, mgr=mgr)
        with patch.object(decode_mod.time, "monotonic", return_value=5.0):
            q._defer_release(dreq)
        self.assertEqual(len(q._deferred_releases), 1)
        held_req, start_time, deadline, held_idx, required = q._deferred_releases[0]
        self.assertIs(held_req, dreq)
        self.assertEqual(start_time, 5.0)
        self.assertEqual(deadline, 17.5)
        self.assertEqual(held_idx, 9)
        self.assertEqual(required, 1)

    def test_two_rank_release_consensus(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(
                _run_release_consensus,
                args=(str(Path(tmp) / "rendezvous"),),
                nprocs=2,
                join=True,
            )


def _run_release_consensus(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        observations = []
        expected = []
        for scenario, host, drained, now, first_freed in (
            ("ack", False, rank == 0, 10.0, [1]),
            ("deadline", False, False, 31.0 if rank == 0 else 29.0, [1]),
            ("mixed", False, rank == 0, 31.0, [0, 1]),
            ("host_ack", True, rank == 0, 10.0, [1]),
            ("host_timeout", True, False, 31.0, [1]),
        ):
            mgr = _make_manager()
            q = _make_queue()
            q.enable_host_receive = host
            dreq = _make_decode_req(42, 0, mgr)
            dreq.host_staged = host
            dreq.kv_receiver.is_abort_release_safe = lambda: drained
            generation = mgr.register_deferred_abort_room(42)
            if drained:
                mgr.note_abort_ack(42, 0, generation)
            # An independently ready entry must be released even while the
            # first entry waits for a peer; agreement is per request.
            ready_req = _make_decode_req(44, 1, mgr)
            ready_req.host_staged = host
            ready_req.kv_receiver.is_abort_release_safe = lambda: True
            ready_generation = mgr.register_deferred_abort_room(44)
            mgr.note_abort_ack(44, 0, ready_generation)
            with patch.object(decode_mod.time, "monotonic", return_value=0.0):
                q._defer_release(dreq)
                q._defer_release(ready_req)
            with (
                patch.object(decode_mod.time, "monotonic", return_value=now) as clock,
                patch.object(decode_mod, "release_kv_cache"),
                patch.object(decode_mod, "discard_kv_cache_backup"),
            ):
                q.resolve_deferred_releases()
                observations.append(
                    (scenario, list(q.req_to_metadata_buffer_idx_allocator.freed))
                )
                expected.append((scenario, first_freed))

                drained = True
                if scenario != "deadline":
                    mgr.note_abort_ack(42, 0, generation)
                clock.return_value = 31.0
                q.resolve_deferred_releases()
                q.resolve_deferred_releases()
                observations.append(
                    (
                        scenario + "_drained",
                        list(q.req_to_metadata_buffer_idx_allocator.freed),
                    )
                )
                expected.append(
                    (scenario + "_drained", [0, 1] if scenario == "mixed" else [1, 0])
                )

        for notification in ("some", "all", "none", "disabled"):
            mgr = _make_manager()
            q = _make_queue()
            q.enable_deferred_kv_release = notification != "disabled"
            dreq = _make_decode_req(43, 1, mgr)
            receiver = dreq.kv_receiver
            receiver.abort_notified = notification == "all" or (
                notification == "some" and rank == 0
            )
            if not receiver.abort_notified:
                receiver.bootstrap_infos = None
            generation = mgr.register_deferred_abort_room(43)
            q.queue = [dreq]
            with (
                patch.object(decode_mod, "prepare_abort"),
                patch.object(decode_mod, "release_kv_cache"),
            ):
                q.pop_transferred()
                held = len(q._deferred_releases)
                observations.append((notification, held))
                expected.append((notification, int(notification in ("some", "all"))))
                held_by_rank = [None, None]
                dist.all_gather_object(held_by_rank, held)
                # Keep a broken implementation from deadlocking the test when
                # just one rank enters the release collective.
                if held_by_rank == [1, 1]:
                    mgr.note_abort_ack(43, 0, generation)
                    q.resolve_deferred_releases()
                observations.append(
                    (
                        notification + "_freed",
                        list(q.req_to_metadata_buffer_idx_allocator.freed),
                    )
                )
                expected.append((notification + "_freed", [1]))

        # Both ranks reach the same assertion even if the old code diverges.
        results = [None, None]
        dist.all_gather_object(results, observations)
        assert results == [expected, expected], results
    finally:
        dist.destroy_process_group()


def _make_pop_queue(mgr):
    """DecodeTransferQueue shell wired for pop_transferred's Failed branch."""
    q = _make_queue()
    q.enable_host_receive = False
    q.enable_deferred_kv_release = True
    q.tp_rank = 0
    q.scheduler = SimpleNamespace(
        enable_decode_hicache=False,
        enable_hisparse=False,
        output_streamer=SimpleNamespace(stream_output=lambda reqs, logprob: None),
        metrics_reporter=SimpleNamespace(enable_metrics=False),
    )
    q._poll_with_metadata_gate = lambda: [KVPoll.Failed] * len(q.queue)
    q._clean_hicache_prefetch_resources = lambda decode_req: None
    return q


def _make_failed_req(mgr, room, idx, abort_notified, notifiable=True):
    receiver = SimpleNamespace(
        kv_mgr=mgr,
        abort_notified=abort_notified,
        bootstrap_infos=[{"rank": 0}, {"rank": 1}],
        failure_exception=lambda: None,
        clear=lambda: None,
    )

    def ensure(force_arm=False):
        # A receiver that never published metadata has nobody to notify and
        # leaves abort_notified False, like the real ensure_abort_notified.
        receiver.force_arm_calls.append(force_arm)
        if notifiable:
            receiver.abort_notified = True

    receiver.force_arm_calls = []
    receiver.ensure_abort_notified = ensure
    return SimpleNamespace(
        req=SimpleNamespace(bootstrap_room=room, rid=f"r{room}", return_logprob=False),
        kv_receiver=receiver,
        metadata_buffer_index=idx,
        hicache_restore_status=None,
        host_staged=False,
    )


class TestFailedTransfersDeferOnEveryFailure(CustomTestCase):
    """A transfer that fails WITHOUT a decode-initiated abort (prefill fault,
    transport error) can still have sibling-rank writes in flight toward the
    request's pages; releasing at t=0 re-opens the reuse race the hold exists
    to prevent. pop_transferred must notify the prefill ranks and defer on
    every failure kind, not only when abort_notified is already set."""

    def setUp(self):
        super().setUp()
        world_size = patch.object(dist, "get_world_size", return_value=1)
        world_size.start()
        self.addCleanup(world_size.stop)

    def _pop(self, q):
        released = []
        q._release_request = released.append
        with patch.object(decode_mod, "prepare_abort"):
            q.pop_transferred()
        return released

    def test_non_abort_failure_notifies_and_defers(self):
        mgr = _make_manager()
        mgr.enable_deferred_decode_kv_release = True
        q = _make_pop_queue(mgr)
        entry = _make_failed_req(mgr, room=7, idx=3, abort_notified=False)
        q.queue = [entry]

        released = self._pop(q)

        self.assertTrue(entry.kv_receiver.abort_notified)  # prefill was told
        # force_arm: the receiver may have failed mid-publish (init_time None),
        # and this deferral only drains via acks if the tracker armed.
        self.assertEqual(entry.kv_receiver.force_arm_calls, [True])
        self.assertEqual(released, [])
        self.assertEqual(len(q._deferred_releases), 1)
        # Metadata slot stays owned by the hold until resolve time.
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [])
        self.assertEqual(q.queue, [])

    def test_unnotifiable_failure_releases_immediately(self):
        # Metadata never published: no prefill holds this destination, so no
        # write can be in flight and holding would waste a full timeout.
        mgr = _make_manager()
        mgr.enable_deferred_decode_kv_release = True
        q = _make_pop_queue(mgr)
        entry = _make_failed_req(
            mgr, room=8, idx=4, abort_notified=False, notifiable=False
        )
        q.queue = [entry]

        released = self._pop(q)

        self.assertEqual(released, [entry])
        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [4])

    def test_backend_optout_keeps_immediate_release_and_sends_no_abort(self):
        mgr = _make_manager()
        mgr.enable_deferred_decode_kv_release = False
        q = _make_pop_queue(mgr)
        entry = _make_failed_req(mgr, room=9, idx=5, abort_notified=False)
        receiver = entry.kv_receiver  # the release path nulls entry.kv_receiver
        q.queue = [entry]

        released = self._pop(q)

        self.assertFalse(receiver.abort_notified)
        self.assertEqual(released, [entry])
        self.assertEqual(q._deferred_releases, [])


class TestBackendOptIn(CustomTestCase):
    """Without a prefill ack, every hold waits out the full release timeout."""

    def test_backends_without_a_drain_ack_stay_opted_out(self):
        # Inheriting CommonKVManager is not enough: a backend must send the
        # drain ack itself before it may opt in.
        self.assertFalse(BaseKVManager.supports_deferred_decode_kv_release)
        self.assertFalse(CommonKVManager.supports_deferred_decode_kv_release)


if __name__ == "__main__":
    unittest.main()
