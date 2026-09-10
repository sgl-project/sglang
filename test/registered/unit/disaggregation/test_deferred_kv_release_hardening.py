"""Unit tests hardening the deferred KV release mechanism for PD aborts.

Companion to test_deferred_decode_kv_release.py. Each test forces one of the
interleavings that previously let a page be released while a transfer for it
was still inside the transfer engine, or let the decode wait for the release
timeout because the prefill's drain ack was lost:

- prefill clear() dropping the held ack target before the worker drained
- a second decode rank's ABORT overwriting the first rank's ack target
- ABORT arriving for a room the scheduler already cleared mid-chunk
- a prefill-initiated failure releasing decode pages without an abort round trip
- the decode waiting-timeout abort never arming the drain-ack accounting
- prefill source pages released while the sender still has a chunk in flight
- per-layer transfer futures abandoned after the first failure

All fakes; no GPU, no network.
"""

import concurrent.futures
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation import prefill as prefill_mod
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_prefill_manager(deferred=True):
    """Prefill-side manager carrying only the drain-ack state; acks recorded."""
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.enable_deferred_decode_kv_release = deferred
    mgr._deferred_ack_targets = {}
    mgr._staging_outstanding = defaultdict(int)
    mgr.request_status = {}
    mgr.transfer_infos = {}
    mgr.req_to_decode_prefix_len = {}
    mgr.acks = []
    mgr._send_abort_ack = lambda ip, port, room: mgr.acks.append((ip, port, room))
    return mgr


class _Sender(CommonKVSender):
    def poll(self):
        raise NotImplementedError

    def failure_exception(self):
        raise NotImplementedError


class _Receiver(CommonKVReceiver):
    def poll(self):
        raise NotImplementedError

    def failure_exception(self):
        raise NotImplementedError


def _make_sender(mgr, room):
    sender = _Sender.__new__(_Sender)
    sender.kv_mgr = mgr
    sender.bootstrap_room = room
    return sender


class TestPrefillAckTargetSurvivesClear(CustomTestCase):
    """A: clear() must not drop the ack target while a chunk is outstanding."""

    def test_clear_keeps_target_while_chunk_outstanding_then_worker_acks(self):
        mgr = _make_prefill_manager()
        room = 10
        mgr.request_status[room] = KVPoll.Failed
        # Worker dequeued a chunk, then ABORT arrived and registered a target.
        mgr._staging_outstanding[room] = 1
        mgr.register_deferred_ack_target(room, "10.0.0.1", 5000)

        # Scheduler polls Failed and clears the room while the chunk is in flight.
        _make_sender(mgr, room).clear()
        self.assertIn(room, mgr._deferred_ack_targets)
        self.assertEqual(mgr.acks, [])

        # Worker finishes the chunk: outstanding hits zero and it acks.
        mgr._staging_outstanding[room] -= 1
        mgr._maybe_ack_drained_abort(room)
        self.assertEqual(mgr.acks, [("10.0.0.1", 5000, room)])
        self.assertNotIn(room, mgr._deferred_ack_targets)

    def test_clear_with_nothing_outstanding_acks_and_does_not_leak(self):
        mgr = _make_prefill_manager()
        room = 11
        mgr.request_status[room] = KVPoll.Failed
        mgr.register_deferred_ack_target(room, "10.0.0.1", 5000)

        _make_sender(mgr, room).clear()
        self.assertEqual(mgr.acks, [("10.0.0.1", 5000, room)])
        self.assertNotIn(room, mgr._deferred_ack_targets)

    def test_clear_without_target_is_noop(self):
        mgr = _make_prefill_manager()
        _make_sender(mgr, 12).clear()
        self.assertEqual(mgr.acks, [])
        self.assertEqual(mgr._deferred_ack_targets, {})


class TestMultipleAckTargetsPerRoom(CustomTestCase):
    """B: fan-out registers one target per decode rank; every one is acked."""

    def test_second_decode_rank_does_not_overwrite_first(self):
        mgr = _make_prefill_manager()
        room = 20
        mgr._staging_outstanding[room] = 1
        mgr.register_deferred_ack_target(room, "10.0.0.1", 5000)
        mgr.register_deferred_ack_target(room, "10.0.0.2", 5000)
        self.assertEqual(len(mgr._deferred_ack_targets[room]), 2)

        mgr._staging_outstanding[room] = 0
        mgr._maybe_ack_drained_abort(room)
        self.assertEqual(
            sorted(mgr.acks), [("10.0.0.1", 5000, room), ("10.0.0.2", 5000, room)]
        )
        self.assertNotIn(room, mgr._deferred_ack_targets)

    def test_duplicate_target_acked_once(self):
        mgr = _make_prefill_manager()
        room = 21
        mgr.register_deferred_ack_target(room, "10.0.0.1", 5000)
        mgr.register_deferred_ack_target(room, "10.0.0.1", 5000)
        mgr._maybe_ack_drained_abort(room)
        self.assertEqual(mgr.acks, [("10.0.0.1", 5000, room)])


class TestAbortForClearedRoomWithOutstandingChunk(CustomTestCase):
    """C: the ack decision depends on outstanding chunks, not on room_active."""

    def test_cleared_room_with_outstanding_chunk_defers_ack(self):
        mgr = _make_prefill_manager()
        room = 30
        # Room already cleared by the scheduler, chunk still in the engine.
        mgr._staging_outstanding[room] = 1
        mgr.handle_deferred_abort_ack(room, "10.0.0.1", 5000)
        self.assertEqual(mgr.acks, [])
        self.assertIn(room, mgr._deferred_ack_targets)

        mgr._staging_outstanding[room] -= 1
        mgr._maybe_ack_drained_abort(room)
        self.assertEqual(mgr.acks, [("10.0.0.1", 5000, room)])

    def test_active_room_with_nothing_outstanding_acks_now(self):
        mgr = _make_prefill_manager()
        room = 31
        mgr.request_status[room] = KVPoll.Failed
        mgr.handle_deferred_abort_ack(room, "10.0.0.1", 5000)
        self.assertEqual(mgr.acks, [("10.0.0.1", 5000, room)])
        self.assertNotIn(room, mgr._deferred_ack_targets)

    def test_worker_draining_between_check_and_register_still_acks(self):
        # The handler's post-register retry covers the worker popping the count
        # (and finding no target) right after the handler observed it non-zero.
        mgr = _make_prefill_manager()
        room = 32
        mgr._staging_outstanding[room] = 1
        real_register = mgr.register_deferred_ack_target

        def register_then_drain(*args):
            real_register(*args)
            mgr._staging_outstanding.pop(room, None)
            mgr._maybe_ack_drained_abort(room)

        mgr.register_deferred_ack_target = register_then_drain
        mgr.handle_deferred_abort_ack(room, "10.0.0.1", 5000)
        self.assertEqual(mgr.acks, [("10.0.0.1", 5000, room)])
        self.assertNotIn(room, mgr._deferred_ack_targets)

    def test_backend_without_chunk_accounting_acks_now(self):
        mgr = _make_prefill_manager()
        del mgr._staging_outstanding
        mgr.handle_deferred_abort_ack(33, "10.0.0.1", 5000)
        self.assertEqual(mgr.acks, [("10.0.0.1", 5000, 33)])


def _make_decode_manager(deferred=True):
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.enable_deferred_decode_kv_release = deferred
    mgr._deferred_abort_ack_tracker = {}
    mgr.request_status = {}
    mgr.required_prefill_response_num_table = {}
    mgr.prefill_response_tracker = defaultdict(set)
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr.waiting_timeout = 0.0
    return mgr


def _make_receiver(mgr, room, n_prefill_ranks=2, bootstrap_infos="default"):
    receiver = _Receiver.__new__(_Receiver)
    receiver.kv_mgr = mgr
    receiver.bootstrap_room = room
    receiver.conclude_state = None
    receiver.abort_notified = False
    receiver.init_time = 0.0
    receiver._connection_pool_entries = {}
    if bootstrap_infos == "default":
        bootstrap_infos = [{"rank": r} for r in range(n_prefill_ranks)]
    receiver.bootstrap_infos = bootstrap_infos
    receiver.abort_sends = 0

    def fake_send():
        receiver.abort_sends += 1

    receiver._send_abort_notification = fake_send
    receiver.invalidate_cached_bootstrap_infos = lambda: None
    return receiver


class TestAbortNotificationArmsDrainAcks(CustomTestCase):
    """E: every ABORT path arms drain-ack accounting through one helper."""

    def test_waiting_timeout_registers_room(self):
        mgr = _make_decode_manager()
        receiver = _make_receiver(mgr, 40)
        self.assertEqual(receiver._check_waiting_timeout(), KVPoll.Failed)
        self.assertTrue(receiver.abort_notified)
        self.assertEqual(receiver.abort_sends, 1)
        self.assertIn(40, mgr._deferred_abort_ack_tracker)
        # An ack racing back right after the ABORT is now counted.
        mgr.note_abort_ack(40, 0)
        mgr.note_abort_ack(40, 1)
        self.assertTrue(mgr.is_abort_release_safe(40, required_acks=2))

    def test_abort_registers_room(self):
        mgr = _make_decode_manager()
        receiver = _make_receiver(mgr, 41)
        receiver.abort()
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        self.assertIn(41, mgr._deferred_abort_ack_tracker)

    def test_redundant_abort_does_not_wipe_recorded_acks(self):
        mgr = _make_decode_manager()
        receiver = _make_receiver(mgr, 42)
        receiver.abort()
        mgr.note_abort_ack(42, 0)
        receiver.abort()
        self.assertEqual(receiver.abort_sends, 1)
        self.assertTrue(mgr.is_abort_release_safe(42, required_acks=1))

    def test_not_registered_when_deferred_release_disabled(self):
        mgr = _make_decode_manager(deferred=False)
        receiver = _make_receiver(mgr, 43)
        receiver.abort()
        self.assertEqual(receiver.abort_sends, 1)
        self.assertNotIn(43, mgr._deferred_abort_ack_tracker)

    def test_no_bootstrap_infos_means_not_notified(self):
        mgr = _make_decode_manager()
        receiver = _make_receiver(mgr, 44, bootstrap_infos=None)
        self.assertFalse(receiver.ensure_abort_notified())
        self.assertFalse(receiver.abort_notified)
        self.assertNotIn(44, mgr._deferred_abort_ack_tracker)


class _FakeIdxAllocator:
    def __init__(self):
        self.freed = []

    def free(self, idx):
        self.freed.append(idx)


def _make_transfer_queue(polls):
    q = DecodeTransferQueue.__new__(DecodeTransferQueue)
    q.queue = []
    q.tp_rank = 0
    q.enable_staging = False
    q.staging_handler = None
    q.enable_deferred_kv_release = True
    q.deferred_kv_release_timeout = 30.0
    q._deferred_releases = []
    q.tree_cache = object()
    q.metadata_buffers = SimpleNamespace(bootstrap_room={})
    q.req_to_metadata_buffer_idx_allocator = _FakeIdxAllocator()
    q.scheduler = SimpleNamespace(
        enable_decode_hicache=False,
        enable_hisparse=False,
        output_streamer=SimpleNamespace(stream_output=lambda reqs, logprob: None),
        metrics_reporter=SimpleNamespace(enable_metrics=False),
    )
    q._poll_with_metadata_gate = lambda: polls
    q._clean_hicache_prefetch_resources = lambda decode_req: None
    return q


def _make_decode_req(mgr, room, idx, bootstrap_infos="default"):
    receiver = _make_receiver(mgr, room, bootstrap_infos=bootstrap_infos)
    mgr.request_status[room] = KVPoll.Failed
    receiver.failure_exception = lambda: None
    return SimpleNamespace(
        req=SimpleNamespace(rid=f"r{room}", bootstrap_room=room, return_logprob=False),
        kv_receiver=receiver,
        hicache_restore_status=None,
        metadata_buffer_index=idx,
    )


class TestNonDecodeInitiatedFailureDefers(CustomTestCase):
    """D: any Failed defers when enabled, after telling every prefill rank."""

    def test_prefill_initiated_failure_sends_abort_and_defers(self):
        mgr = _make_decode_manager()
        dreq = _make_decode_req(mgr, 50, idx=3)
        q = _make_transfer_queue([KVPoll.Failed])
        q.queue = [dreq]

        with (
            patch.object(decode_mod, "release_kv_cache") as rel,
            patch.object(decode_mod, "prepare_abort"),
        ):
            self.assertEqual(q.pop_transferred(), [])

        rel.assert_not_called()
        self.assertEqual(dreq.kv_receiver.abort_sends, 1)
        self.assertIn(50, mgr._deferred_abort_ack_tracker)
        self.assertEqual(len(q._deferred_releases), 1)
        held, _deadline, held_idx, required_acks = q._deferred_releases[0]
        self.assertIs(held, dreq)
        self.assertEqual(held_idx, 3)
        self.assertEqual(required_acks, 2)
        self.assertEqual(q.queue, [])
        # Slot is held too: freed at resolve time, not here.
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [])

        # Both prefill ranks ack their drain -> released exactly once.
        mgr.note_abort_ack(50, 0)
        mgr.note_abort_ack(50, 1)
        with patch.object(decode_mod, "release_kv_cache") as rel:
            q.resolve_deferred_releases()
        rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [3])

    def test_decode_initiated_abort_does_not_resend_abort(self):
        mgr = _make_decode_manager()
        dreq = _make_decode_req(mgr, 51, idx=4)
        dreq.kv_receiver.abort()
        self.assertEqual(dreq.kv_receiver.abort_sends, 1)
        q = _make_transfer_queue([KVPoll.Failed])
        q.queue = [dreq]

        with (
            patch.object(decode_mod, "release_kv_cache") as rel,
            patch.object(decode_mod, "prepare_abort"),
        ):
            q.pop_transferred()

        rel.assert_not_called()
        self.assertEqual(dreq.kv_receiver.abort_sends, 1)
        self.assertEqual(len(q._deferred_releases), 1)

    def test_failure_without_bootstrap_infos_releases_immediately(self):
        # No prefill was ever told about these pages, so nothing can write them.
        mgr = _make_decode_manager()
        dreq = _make_decode_req(mgr, 52, idx=5, bootstrap_infos=None)
        q = _make_transfer_queue([KVPoll.Failed])
        q.queue = [dreq]

        with (
            patch.object(decode_mod, "release_kv_cache") as rel,
            patch.object(decode_mod, "prepare_abort"),
        ):
            q.pop_transferred()

        rel.assert_called_once_with(dreq.req, q.tree_cache, is_insert=False)
        self.assertEqual(q._deferred_releases, [])
        self.assertEqual(q.req_to_metadata_buffer_idx_allocator.freed, [5])

    def test_backend_without_deferred_support_releases_immediately(self):
        mgr = _make_decode_manager(deferred=False)
        dreq = _make_decode_req(mgr, 53, idx=6)
        receiver = dreq.kv_receiver
        q = _make_transfer_queue([KVPoll.Failed])
        q.queue = [dreq]

        with (
            patch.object(decode_mod, "release_kv_cache") as rel,
            patch.object(decode_mod, "prepare_abort"),
        ):
            q.pop_transferred()

        rel.assert_called_once()
        self.assertEqual(receiver.abort_sends, 0)
        self.assertIsNone(dreq.kv_receiver)
        self.assertEqual(q._deferred_releases, [])


def _make_mooncake_sender(mgr, room):
    sender = MooncakeKVSender.__new__(MooncakeKVSender)
    sender.kv_mgr = mgr
    sender.bootstrap_room = room
    sender.conclude_state = None
    sender.init_time = None
    sender.trace_ctx = SimpleNamespace(trace_req_finish=lambda: None)
    return sender


class TestPrefillSenderPollHoldsFailedWhileOutstanding(CustomTestCase):
    """F(i): Failed is held as Transferring while a chunk is still in flight."""

    def test_failed_held_until_drained(self):
        mgr = _make_prefill_manager()
        mgr.check_status = lambda room: mgr.request_status[room]
        room = 60
        mgr.request_status[room] = KVPoll.Failed
        mgr._staging_outstanding[room] = 1
        sender = _make_mooncake_sender(mgr, room)

        self.assertEqual(sender.poll(), KVPoll.Transferring)
        self.assertIsNone(sender.conclude_state)
        mgr._staging_outstanding[room] = 0
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertEqual(sender.conclude_state, KVPoll.Failed)

    def test_failed_not_held_when_deferred_release_disabled(self):
        mgr = _make_prefill_manager(deferred=False)
        mgr.check_status = lambda room: mgr.request_status[room]
        room = 61
        mgr.request_status[room] = KVPoll.Failed
        mgr._staging_outstanding[room] = 1
        self.assertEqual(_make_mooncake_sender(mgr, room).poll(), KVPoll.Failed)

    def test_success_still_held_regardless_of_flag(self):
        mgr = _make_prefill_manager(deferred=False)
        mgr.check_status = lambda room: mgr.request_status[room]
        room = 62
        mgr.request_status[room] = KVPoll.Success
        mgr._staging_outstanding[room] = 1
        self.assertEqual(_make_mooncake_sender(mgr, room).poll(), KVPoll.Transferring)


def _make_prefill_scheduler():
    return SimpleNamespace(
        tree_cache=object(),
        disagg_prefill_drain_releases=[],
        release_prefill_kv_after_drain=(
            SchedulerDisaggregationPrefillMixin.release_prefill_kv_after_drain
        ),
        resolve_prefill_drain_releases=(
            SchedulerDisaggregationPrefillMixin.resolve_prefill_drain_releases
        ),
    )


def _make_prefill_req(mgr, room):
    return SimpleNamespace(
        rid=f"p{room}",
        bootstrap_room=room,
        disagg_kv_sender=SimpleNamespace(kv_mgr=mgr, bootstrap_room=room),
        kv=SimpleNamespace(holds_kv=True, holds_mamba=False),
    )


class TestPrefillSourceReleaseWaitsForDrain(CustomTestCase):
    """F(ii): direct release paths hold source pages while a chunk is in flight."""

    def test_release_held_then_resolved_once_after_drain(self):
        mgr = _make_prefill_manager()
        sched = _make_prefill_scheduler()
        req = _make_prefill_req(mgr, 70)
        mgr._staging_outstanding[70] = 1

        with patch.object(prefill_mod, "release_kv_cache") as rel:
            self.assertFalse(
                sched.release_prefill_kv_after_drain(sched, req, is_insert=False)
            )
            rel.assert_not_called()
            self.assertEqual(sched.disagg_prefill_drain_releases, [(req, False)])

            # Still in flight: held across a resolve.
            sched.resolve_prefill_drain_releases(sched)
            rel.assert_not_called()

            # Drained: released exactly once, with the original is_insert.
            mgr._staging_outstanding[70] = 0
            sched.resolve_prefill_drain_releases(sched)
            rel.assert_called_once_with(req, sched.tree_cache, is_insert=False)
            self.assertEqual(sched.disagg_prefill_drain_releases, [])
            sched.resolve_prefill_drain_releases(sched)
            rel.assert_called_once()

    def test_release_immediate_when_nothing_outstanding(self):
        mgr = _make_prefill_manager()
        sched = _make_prefill_scheduler()
        req = _make_prefill_req(mgr, 71)
        with patch.object(prefill_mod, "release_kv_cache") as rel:
            self.assertTrue(sched.release_prefill_kv_after_drain(sched, req))
        rel.assert_called_once_with(req, sched.tree_cache, is_insert=True)
        self.assertEqual(sched.disagg_prefill_drain_releases, [])

    def test_release_immediate_when_deferred_release_disabled(self):
        mgr = _make_prefill_manager(deferred=False)
        sched = _make_prefill_scheduler()
        req = _make_prefill_req(mgr, 72)
        mgr._staging_outstanding[72] = 1
        with patch.object(prefill_mod, "release_kv_cache") as rel:
            self.assertTrue(sched.release_prefill_kv_after_drain(sched, req))
        rel.assert_called_once()

    def test_release_immediate_without_sender(self):
        sched = _make_prefill_scheduler()
        req = _make_prefill_req(None, 73)
        req.disagg_kv_sender = None
        with patch.object(prefill_mod, "release_kv_cache") as rel:
            self.assertTrue(sched.release_prefill_kv_after_drain(sched, req))
        rel.assert_called_once()

    def test_resolve_skips_request_already_released_elsewhere(self):
        mgr = _make_prefill_manager()
        sched = _make_prefill_scheduler()
        req = _make_prefill_req(mgr, 74)
        mgr._staging_outstanding[74] = 1
        with patch.object(prefill_mod, "release_kv_cache") as rel:
            sched.release_prefill_kv_after_drain(sched, req)
            req.kv.holds_kv = False
            mgr._staging_outstanding[74] = 0
            sched.resolve_prefill_drain_releases(sched)
        rel.assert_not_called()
        self.assertEqual(sched.disagg_prefill_drain_releases, [])


class TestAwaitTransferFuturesDrains(CustomTestCase):
    """G: every submitted future terminates before the call returns."""

    def setUp(self):
        self.mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def tearDown(self):
        self.executor.shutdown(wait=True)

    def test_running_sibling_is_drained_after_first_failure(self):
        release = threading.Event()
        sibling_done = threading.Event()

        def slow_ok():
            release.wait(timeout=10)
            sibling_done.set()
            return 0

        def fail_fast():
            return 7

        futures = [self.executor.submit(slow_ok), self.executor.submit(fail_fast)]
        # Let the failure land first, then unblock the running sibling from a
        # helper thread so the awaiting call has to wait for it.
        futures[1].result()
        threading.Timer(0.05, release.set).start()
        self.assertEqual(self.mgr._await_transfer_futures(futures), 7)
        self.assertTrue(sibling_done.is_set())
        self.assertTrue(all(f.done() for f in futures))

    def test_exception_is_normalized_to_failure_status(self):
        def boom():
            raise RuntimeError("rdma")

        futures = [self.executor.submit(boom), self.executor.submit(lambda: 0)]
        self.assertNotEqual(self.mgr._await_transfer_futures(futures), 0)
        self.assertTrue(all(f.done() for f in futures))

    def test_first_nonzero_status_is_returned(self):
        futures = [self.executor.submit(lambda: 0), self.executor.submit(lambda: 3)]
        concurrent.futures.wait(futures)
        self.assertEqual(self.mgr._await_transfer_futures(futures), 3)

    def test_all_success_returns_zero(self):
        futures = [self.executor.submit(lambda: 0) for _ in range(3)]
        self.assertEqual(self.mgr._await_transfer_futures(futures), 0)


if __name__ == "__main__":
    unittest.main()
