"""Tests for the KV transfer ownership barrier.

The barrier keeps a failed request's KV pages allocated until no native
transfer work can still read or write them. The properties under test are:

* pages stay owned while transfer work is in flight (safety);
* ownership is always released eventually, even if a peer never answers
  (liveness -- a wedged transport must not pin pages forever);
* the deferral only happens where the scheduler can handle it, and never
  weakens the failure diagnostics the operator sees.
"""

import concurrent.futures
import threading
import time
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import zmq

from sglang.srt.disaggregation.base.conn import (
    KVPoll,
    KVTransferBarrierEscalation,
)
from sglang.srt.disaggregation.common.staging_handler import DecodeStagingHandler
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    TransferKVChunk,
    drain_transfer_futures,
    submit_transfer_calls,
)
from sglang.srt.disaggregation.mooncake.conn import (
    ABORT_RETRY_INTERVAL_S,
    MAX_EMERGENCY_SCAN,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
    RoomTransferLifetime,
    TransferInfo,
)
from sglang.srt.disaggregation.utils import DisaggregationMode, poll_and_all_reduce
from sglang.srt.environ import TransferBarrierLevel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


ROOM = 7


def make_prefill_manager(
    quiesce_timeout=30.0, room_sweep_ttl=300.0, barrier=TransferBarrierLevel.WARN
):
    """A prefill-side manager with only its ownership-barrier state wired up."""
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.disaggregation_mode = DisaggregationMode.PREFILL
    mgr.quiesce_timeout = quiesce_timeout
    mgr.transfer_barrier = barrier
    mgr.max_unquiesced = 256
    mgr._unquiesced_rooms = set()
    mgr._unquiesced_lock = threading.Lock()
    mgr.bootstrap_timeout = 300
    mgr._room_sweep_ttl = room_sweep_ttl
    mgr._room_lifetimes = OrderedDict()
    mgr._room_lifetimes_lock = threading.Lock()
    mgr._endpoint_send_locks = {}
    mgr._endpoint_send_locks_lock = threading.Lock()
    mgr.request_status = {}
    mgr.transfer_infos = {}
    mgr.req_to_decode_prefix_len = {}
    mgr.check_status = lambda room: mgr.request_status[room]
    mgr.update_status = lambda room, status: mgr.request_status.__setitem__(
        room, status
    )
    mgr.record_failure = Mock()
    mgr.enable_trace = False
    mgr.enable_staging = False
    mgr.bootstrap_port = 0
    # Minimal KVArgs surface the transfer worker dereferences.
    mgr.kv_args = SimpleNamespace(
        kv_data_ptrs=[0x1000],
        kv_layer_ids=[0],
        state_layer_ids=[],
        state_data_ptrs=[],
    )
    mgr._get_dsa_cache_transfer_skip_flags = Mock(return_value=(False, False))
    return mgr


def make_sender(mgr, room=ROOM, status=KVPoll.WaitingForInput):
    """A prefill sender bound to *mgr*, bypassing bootstrap/registration."""
    sender = MooncakeKVSender.__new__(MooncakeKVSender)
    sender.kv_mgr = mgr
    sender.bootstrap_room = room
    sender.conclude_state = None
    sender._quiescing = False
    sender._quiesce_deadline = float("inf")
    sender._quiesce_timed_out = False
    sender.trace_ctx = SimpleNamespace(
        abort=Mock(),
        trace_req_finish=Mock(),
        copy_for_thread=Mock(return_value=Mock()),
    )
    mgr.open_room_transfers(room)
    if status is not None:
        mgr.request_status[room] = status
    return sender


def make_receiver(mgr, room=ROOM, bootstrap_infos=None, metadata_sent=True):
    """A decode receiver bound to *mgr*, bypassing bootstrap/registration."""
    receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
    receiver.kv_mgr = mgr
    receiver.bootstrap_room = room
    receiver.conclude_state = None
    receiver.require_staging = False
    receiver.bootstrap_infos = bootstrap_infos
    receiver._metadata_sent = metadata_sent
    receiver._quiescing = False
    receiver._quiesce_complete = False
    receiver._quiesce_deadline = float("inf")
    receiver._abort_targets = []
    receiver._expected_abort_acks = set()
    receiver._received_abort_acks = set()
    receiver._peer_lacks_barrier = False
    receiver._last_abort_send = float("-inf")
    receiver._owns_room = True
    receiver._abort_lock = threading.Lock()
    receiver._connect_to_bootstrap_server = Mock(
        side_effect=lambda info: (Mock(), threading.Lock())
    )
    return receiver


def make_decode_manager(
    quiesce_timeout=30.0, staging_handler=None, barrier=TransferBarrierLevel.WARN
):
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.quiesce_timeout = quiesce_timeout
    mgr._receivers = {}
    mgr._receivers_lock = threading.Lock()
    mgr._staging_handler = staging_handler
    mgr.transfer_barrier = barrier
    mgr.max_unquiesced = 256
    mgr._unquiesced_rooms = set()
    mgr._unquiesced_lock = threading.Lock()
    mgr.request_status = {}
    mgr.local_ip = "127.0.0.1"
    mgr.rank_port = 1234
    mgr.record_failure = Mock()
    mgr.update_status = Mock()
    return mgr


class TestPageReuseRace(unittest.TestCase):
    """The reproducer: a transfer must never outlive its pages' ownership.

    This is the whole bug, expressed without a GPU, a network, or a second
    process. It drives the real transfer_worker with an engine that blocks inside
    the write, then asks the real poll path what the scheduler would be told.
    Reporting a terminal state is what lets the scheduler hand the pages back to
    the allocator, so a terminal state while the engine is still writing means a
    request allocated those pages next has its KV silently overwritten.

    On a build without the barrier this asserts Failed at [T3] and fails.
    """

    def test_terminal_state_is_withheld_while_a_write_is_in_flight(self):
        entered_write = threading.Event()
        release_write = threading.Event()

        mgr = make_prefill_manager()
        mgr.request_status[ROOM] = KVPoll.WaitingForInput
        mgr.transfer_infos = {
            ROOM: {
                "session:1": SimpleNamespace(
                    room=ROOM,
                    endpoint="127.0.0.1",
                    dst_port=1,
                    mooncake_session_id="session:1",
                    dst_kv_indices=np.array([0], dtype=np.int32),
                    is_dummy=False,
                )
            }
        }
        mgr.decode_kv_args_table = {
            "session:1": SimpleNamespace(
                dst_attn_tp_size=1,
                dst_kv_ptrs=[0xDEAD0000],
                dst_kv_layer_ids=[0],
                dst_state_layer_ids=[],
            )
        }
        mgr.session_lock, mgr.failed_sessions = threading.Lock(), set()
        mgr.is_mla_backend, mgr.is_hybrid_mla_backend = True, False
        mgr.attn_tp_size = mgr.attn_cp_size = mgr.pp_size = 1
        mgr.attn_tp_rank = mgr.attn_cp_rank = mgr.pp_rank = mgr.attn_dp_rank = 0
        mgr.transfer_queues = [FastQueue()]

        def blocking_write(*_args, **_kwargs):
            entered_write.set()
            release_write.wait(5)
            return 0

        mgr.send_kvcache = blocking_write
        sender = make_sender(mgr)

        threading.Thread(
            target=mgr.transfer_worker,
            args=(mgr.transfer_queues[0], Mock()),
            daemon=True,
        ).start()
        mgr.add_transfer_request(
            ROOM, np.array([0], dtype=np.int32), slice(0, 1), False
        )

        # [T1] a worker is inside a write that targets the decode's KV pages
        self.assertTrue(entered_write.wait(5), "worker never entered the transfer")

        # [T2] the request is aborted while that write is still in flight
        sender.abort()

        # [T3] the scheduler must not be told this request is over
        self.assertEqual(
            self._poll(sender),
            KVPoll.Transferring,
            "a terminal state here lets the allocator hand these pages to another "
            "request while the transfer is still writing to them",
        )

        # [T4] once the write returns, the request may conclude
        release_write.set()
        for _ in range(500):
            if self._poll(sender) == KVPoll.Failed:
                break
            time.sleep(0.01)
        self.assertEqual(self._poll(sender), KVPoll.Failed)

    def test_a_queued_chunk_cannot_attach_to_a_recycled_room(self):
        """The other half of the race, reported on the upstream discussion.

        A chunk still sitting in the transfer queue names the *old* request's
        pages and the *old* decode's destinations. bootstrap_room is recycled, so
        matching on the room alone would let that chunk be transferred on behalf
        of whichever request holds the room now.
        """
        mgr = make_prefill_manager()
        first = make_sender(mgr)
        stale = TransferKVChunk(
            room=ROOM,
            prefill_kv_indices=np.array([0], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
            owner=mgr._room_lifetime(ROOM, create=False),
            trace_ctx=Mock(),
        )
        first.clear()  # the request concludes, the room is released

        # a new request draws the same bootstrap_room
        make_sender(mgr)
        self.assertIsNotNone(mgr._room_lifetime(ROOM, create=False))
        self.assertIsNone(
            mgr.try_lease_chunk(stale),
            "the stale chunk must not lease the new request's room",
        )

    @staticmethod
    def _poll(sender):
        with patch(
            "sglang.srt.disaggregation.utils.dist.all_reduce",
            side_effect=lambda tensor, **kw: None,
        ):
            return poll_and_all_reduce([sender], object())[0]


class TestRoomTransferLifetime(unittest.TestCase):
    """The primitive that decides when a room's pages may be reused."""

    def test_open_room_is_not_quiesced_and_admits_leases(self):
        lifetime = RoomTransferLifetime()
        self.assertFalse(lifetime.is_quiesced())
        self.assertTrue(lifetime.try_lease())
        lifetime.end_lease()
        self.assertFalse(lifetime.is_quiesced())

    def test_close_stops_admission_and_waits_for_outstanding_lease(self):
        lifetime = RoomTransferLifetime()
        self.assertTrue(lifetime.try_lease())
        lifetime.close()

        self.assertFalse(lifetime.try_lease())
        self.assertFalse(lifetime.is_quiesced())
        self.assertFalse(lifetime.wait_quiesced(timeout=0.01))

        lifetime.end_lease()
        self.assertTrue(lifetime.is_quiesced())
        self.assertTrue(lifetime.wait_quiesced(timeout=1))

    def test_waiter_wakes_when_the_last_lease_is_returned(self):
        lifetime = RoomTransferLifetime()
        self.assertTrue(lifetime.try_lease())
        lifetime.close()
        quiesced = threading.Event()

        waiter = threading.Thread(
            target=lambda: quiesced.set() if lifetime.wait_quiesced(2) else None
        )
        waiter.start()
        self.assertFalse(quiesced.wait(0.05))
        lifetime.end_lease()
        self.assertTrue(quiesced.wait(1))
        waiter.join()

    def test_abort_token_authorization(self):
        lifetime = RoomTransferLifetime()
        # A room with no registered decode metadata accepts any abort: nothing
        # can be in flight, and refusing would drop the abort entirely.
        self.assertTrue(lifetime.authorizes_abort(b"unknown"))
        lifetime.add_abort_token(b"mine")
        self.assertTrue(lifetime.authorizes_abort(b"mine"))
        self.assertFalse(lifetime.authorizes_abort(b"someone-elses"))
        # Peers that predate the token protocol send none; honour those aborts.
        self.assertTrue(lifetime.authorizes_abort(b""))


class TestExecutorDrain(unittest.TestCase):
    """Failing early must not hand pages back while siblings still run."""

    def test_failure_waits_for_a_running_sibling(self):
        sibling_running = threading.Event()
        release_sibling = threading.Event()
        drained = threading.Event()
        result = []

        def sibling():
            sibling_running.set()
            release_sibling.wait()
            return 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(lambda: -1), executor.submit(sibling)]
            self.assertTrue(sibling_running.wait(1))
            waiter = threading.Thread(
                target=lambda: (
                    result.append(drain_transfer_futures(futures)),
                    drained.set(),
                )
            )
            waiter.start()
            self.assertFalse(drained.wait(0.05), "returned while a sibling still ran")
            release_sibling.set()
            self.assertTrue(drained.wait(1))
            waiter.join()

        self.assertEqual(result, [-1], "the first failing status must be reported")

    def test_partial_submission_failure_still_drains_accepted_work(self):
        running = threading.Event()
        release = threading.Event()
        returned = threading.Event()
        errors = []

        def blocked():
            running.set()
            release.wait()
            return 0

        class FailsOnSecondSubmit:
            def __init__(self):
                self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                self.calls = 0

            def submit(self, fn, *args):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("submit failed")
                return self.pool.submit(fn, *args)

        executor = FailsOnSecondSubmit()

        def submit_all():
            try:
                submit_transfer_calls(executor, [(blocked, ()), (lambda: 0, ())])
            except Exception as error:  # noqa: BLE001 - recorded for assertion
                errors.append(error)
            finally:
                returned.set()

        waiter = threading.Thread(target=submit_all)
        waiter.start()
        self.assertTrue(running.wait(1))
        self.assertFalse(returned.wait(0.05), "raised while accepted work still ran")
        release.set()
        self.assertTrue(returned.wait(1))
        waiter.join()
        executor.pool.shutdown()
        self.assertEqual(str(errors[0]), "submit failed")

    def test_worker_exception_propagates_after_draining(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(lambda: (_ for _ in ()).throw(ValueError("boom"))),
                executor.submit(lambda: 0),
            ]
            with self.assertRaises(ValueError):
                drain_transfer_futures(futures)


class TestPrefillOwnership(unittest.TestCase):
    """The prefill's pages are the transfer source."""

    def test_transfer_worker_holds_the_lease_across_the_native_transfer(self):
        entered_transfer = threading.Event()
        release_transfer = threading.Event()

        mgr = make_prefill_manager()
        mgr.request_status[ROOM] = KVPoll.WaitingForInput
        mgr.transfer_infos[ROOM] = {
            "session:1": SimpleNamespace(
                room=ROOM,
                endpoint="127.0.0.1",
                dst_port=1,
                mooncake_session_id="session:1",
                dst_kv_indices=np.array([0], dtype=np.int32),
                is_dummy=False,
            )
        }
        mgr.decode_kv_args_table = {
            "session:1": SimpleNamespace(
                dst_attn_tp_size=1,
                dst_kv_ptrs=[1],
                dst_kv_layer_ids=[0],
                dst_state_layer_ids=[],
            )
        }
        mgr.session_lock = threading.Lock()
        mgr.failed_sessions = set()
        mgr.is_mla_backend = True
        mgr.is_hybrid_mla_backend = False
        mgr.attn_tp_size, mgr.attn_tp_rank = 1, 0
        mgr.attn_cp_size, mgr.attn_cp_rank = 1, 0
        mgr.pp_size, mgr.pp_rank = 1, 0
        mgr.attn_dp_rank = 0

        def blocking_send_kvcache(*_args, **_kwargs):
            entered_transfer.set()
            release_transfer.wait()
            return 0

        mgr.send_kvcache = blocking_send_kvcache
        lifetime = mgr._room_lifetime(ROOM, create=True)
        sender = make_sender(mgr)

        queue = FastQueue()
        threading.Thread(
            target=mgr.transfer_worker, args=(queue, Mock()), daemon=True
        ).start()
        queue.put(
            TransferKVChunk(
                room=ROOM,
                prefill_kv_indices=np.array([0], dtype=np.int32),
                index_slice=slice(0, 1),
                is_last_chunk=False,
                prefill_aux_index=None,
                state_indices=None,
                trace_ctx=Mock(),
            )
        )
        self.assertTrue(entered_transfer.wait(1))

        # An abort now must not let the scheduler reclaim the pages.
        sender._close_barrier()
        self.assertTrue(lifetime.is_closed())
        self.assertFalse(sender.advance_failure_quiescence())

        release_transfer.set()
        for _ in range(200):
            if sender.advance_failure_quiescence():
                break
            time.sleep(0.01)
        self.assertTrue(sender.advance_failure_quiescence())

    def test_closed_room_admits_no_new_chunks(self):
        mgr = make_prefill_manager()
        sender = make_sender(mgr)
        mgr.transfer_queues = [FastQueue()]
        sender._close_barrier()

        mgr.add_transfer_request(
            ROOM, np.array([0], dtype=np.int32), slice(0, 1), False
        )
        self.assertEqual(len(mgr.transfer_queues[0]._buf), 0)

    def test_released_room_cannot_be_resurrected_by_a_queued_chunk(self):
        mgr = make_prefill_manager()
        make_sender(mgr)
        mgr._forget_room_lifetime(ROOM)
        # try_lease_room deliberately refuses to recreate the lifetime, so a
        # chunk that outlived the request cannot start a transfer into pages
        # that have already been handed back to the allocator.
        self.assertIsNone(mgr.try_lease_room(ROOM))
        self.assertNotIn(ROOM, mgr._room_lifetimes)

    def test_quiescence_is_bounded_when_a_transfer_never_returns(self):
        mgr = make_prefill_manager(quiesce_timeout=0.05)
        sender = make_sender(mgr)
        lifetime = mgr._room_lifetime(ROOM, create=False)
        self.assertTrue(lifetime.try_lease())  # never returned

        with patch(
            "sglang.srt.disaggregation.mooncake.conn.TRANSFER_QUIESCE_TIMEOUTS"
        ) as metric:
            self.assertFalse(sender.advance_failure_quiescence())
            time.sleep(0.06)
            self.assertTrue(
                sender.advance_failure_quiescence(),
                "a wedged transfer must not pin the request forever",
            )
            metric.inc.assert_called_once_with()
            # The timeout is reported once, not on every subsequent poll.
            self.assertTrue(sender.advance_failure_quiescence())
            metric.inc.assert_called_once_with()

    def test_bootstrapping_sender_is_immediately_quiesced(self):
        # Nothing has been submitted, so the request can fail without any wait.
        # This is what keeps KVPoll.Transferring out of the bootstrap queues.
        mgr = make_prefill_manager()
        sender = make_sender(mgr, status=KVPoll.Bootstrapping)
        self.assertTrue(sender.advance_failure_quiescence())

    def test_clear_forgets_the_room(self):
        mgr = make_prefill_manager()
        sender = make_sender(mgr)
        sender.clear()
        self.assertNotIn(ROOM, mgr._room_lifetimes)
        self.assertNotIn(ROOM, mgr.request_status)

    def test_tracked_rooms_are_bounded_without_evicting_a_live_room(self):
        mgr = make_prefill_manager()
        live = mgr._room_lifetime(-1, create=True)
        mgr.open_room_transfers(-1)
        self.assertTrue(live.try_lease())
        with patch("sglang.srt.disaggregation.mooncake.conn.MAX_TRACKED_ROOMS", 4):
            for room in range(200):
                mgr._close_room_for_abort(room, b"")
            # ... and rooms known only from decode metadata, which no local
            # sender will ever release, must be bounded too.
            for room in range(1000, 1200):
                mgr._room_lifetime(room, create=True)
        self.assertLessEqual(len(mgr._room_lifetimes), 8)
        self.assertIn(
            -1, mgr._room_lifetimes, "a claimed, leased room must never be evicted"
        )

    def test_emergency_reclaim_cost_does_not_grow_with_the_map(self):
        # A saturated map of non-reclaimable rooms must not make every new room
        # O(tracked rooms) on the bootstrap thread.
        mgr = make_prefill_manager()
        with patch("sglang.srt.disaggregation.mooncake.conn.MAX_TRACKED_ROOMS", 8):
            for room in range(400):
                lifetime = mgr._room_lifetime(room, create=True)
                mgr.open_room_transfers(room)
                self.assertTrue(lifetime.try_lease())
            probe = Mock(side_effect=lambda: False)
            with patch.object(RoomTransferLifetime, "is_reclaimable", probe):
                mgr._room_lifetime(99999, create=True)
        self.assertLessEqual(
            probe.call_count,
            MAX_EMERGENCY_SCAN,
            "the reclaim scan must be bounded per insert",
        )

    def test_sweep_reclaims_rooms_no_sender_will_release(self):
        mgr = make_prefill_manager(room_sweep_ttl=0.0)
        for room in range(5):
            mgr._room_lifetime(room, create=True)  # decode metadata only
        mgr.open_room_transfers(3)  # this one has a local sender
        self.assertEqual(mgr._sweep_room_lifetimes(), 4)
        self.assertEqual(list(mgr._room_lifetimes), [3])

    def test_sweep_keeps_young_rooms_so_a_late_sender_still_finds_them(self):
        mgr = make_prefill_manager(room_sweep_ttl=300.0)
        mgr._room_lifetime(1, create=True)
        self.assertEqual(mgr._sweep_room_lifetimes(), 0)
        self.assertIn(1, mgr._room_lifetimes)

    def test_barrier_can_be_disabled(self):
        mgr = make_prefill_manager(barrier=TransferBarrierLevel.OFF)
        sender = make_sender(mgr)
        lifetime = mgr._room_lifetime(ROOM, create=False)
        self.assertTrue(lifetime.try_lease())
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.TRANSFER_QUIESCE_TIMEOUTS"
        ) as metric:
            self.assertTrue(sender.advance_failure_quiescence())
            metric.inc.assert_not_called()


class TestPrefillAbortProtocol(unittest.TestCase):
    """Aborts arrive out of band and may overtake this rank's own bookkeeping."""

    def _bootstrap_thread_state(self, mgr):
        mgr._start_transfer_bookkeeping()
        mgr._send_manager_message = Mock()
        mgr.session_lock = threading.Lock()
        mgr.failed_sessions = set()
        mgr.session_failures = {}
        mgr.decode_kv_args_table = {}
        mgr.resolve_kv_replica_factor = Mock()
        mgr._staging_ctx = None
        return mgr

    @staticmethod
    def _metadata_msg(room=ROOM, session=b"session:1", token=b"token", required=1):
        return [
            str(room).encode("ascii"),
            b"127.0.0.1",
            b"1",
            session,
            np.array([0], dtype=np.int32).tobytes(),
            b"0",
            b"",
            str(required).encode("ascii"),
            b"0",
            token,
        ]

    def test_metadata_arriving_before_the_sender_is_applied_directly(self):
        # Decode and prefill are separate processes with no ordering between
        # them; metadata for a room this rank has not seen yet must still take
        # effect (KV manager status is monotonic, so the later sender keeps it).
        mgr = self._bootstrap_thread_state(make_prefill_manager())
        self.assertTrue(mgr._handle_bootstrap_metadata(self._metadata_msg()))
        self.assertEqual(mgr.request_status[ROOM], KVPoll.WaitingForInput)
        self.assertIn("session:1", mgr.transfer_infos[ROOM])
        self.assertTrue(
            mgr._room_lifetime(ROOM, create=False).authorizes_abort(b"token")
        )

    def test_abort_arriving_before_the_sender_fails_the_request(self):
        mgr = self._bootstrap_thread_state(make_prefill_manager())
        mgr._handle_abort_notification(
            [b"ABORT", str(ROOM).encode("ascii"), b"127.0.0.1", b"1"]
        )
        # A sender created afterwards must not transfer: open_room_transfers is
        # what MooncakeKVSender.__init__ consults before accepting the request.
        self.assertFalse(mgr.open_room_transfers(ROOM))
        # Metadata that arrives afterwards must not revive the room either.
        self.assertFalse(mgr._handle_bootstrap_metadata(self._metadata_msg()))
        self.assertNotIn(ROOM, mgr.transfer_infos)

    def test_abort_notification_is_acknowledged_even_for_an_unknown_room(self):
        # Otherwise a decode rank waits out its whole timeout for a room this
        # prefill has never heard of.
        mgr = self._bootstrap_thread_state(make_prefill_manager())
        sent = []
        mgr._send_manager_message = lambda ip, port, parts, **kw: sent.append(parts)
        mgr._handle_abort_notification([b"ABORT", b"4242", b"127.0.0.1", b"1", b"tok"])
        for _ in range(200):
            if sent:
                break
            time.sleep(0.01)
        self.assertEqual(sent, [[b"ABORT_ACK", b"4242", b"tok"]])

    def test_a_foreign_token_cannot_close_a_live_room(self):
        mgr = self._bootstrap_thread_state(make_prefill_manager())
        mgr._handle_bootstrap_metadata(self._metadata_msg(token=b"real"))
        self.assertIsNone(mgr._close_room_for_abort(ROOM, b"stale"))
        self.assertFalse(mgr._room_lifetime(ROOM, create=False).is_closed())
        self.assertIsNotNone(mgr._close_room_for_abort(ROOM, b"real"))
        self.assertTrue(mgr._room_lifetime(ROOM, create=False).is_closed())

    def test_abort_ack_is_withheld_until_the_room_drains(self):
        mgr = make_prefill_manager()
        mgr._start_transfer_bookkeeping()
        sent = []
        mgr._send_manager_message = lambda ip, port, parts, **kw: sent.append(parts)

        lifetime = mgr._room_lifetime(ROOM, create=True)
        self.assertTrue(lifetime.try_lease())
        lifetime.close()

        mgr.request_abort_ack(lifetime, ROOM, "127.0.0.1", 1, b"token")
        mgr.request_abort_ack(lifetime, ROOM, "127.0.0.1", 1, b"token")
        time.sleep(0.05)
        self.assertEqual(sent, [], "ACK must not promise a drain that has not happened")

        lifetime.end_lease()
        for _ in range(200):
            if sent:
                break
            time.sleep(0.01)
        self.assertEqual(sent, [[b"ABORT_ACK", str(ROOM).encode("ascii"), b"token"]])

    def test_one_pump_thread_serves_every_pending_ack(self):
        mgr = make_prefill_manager()
        before = threading.active_count()
        mgr._start_transfer_bookkeeping()
        mgr._send_manager_message = Mock()
        lifetimes = []
        for room in range(64):
            lifetime = mgr._room_lifetime(room, create=True)
            self.assertTrue(lifetime.try_lease())
            lifetime.close()
            lifetimes.append(lifetime)
            mgr.request_abort_ack(lifetime, room, "127.0.0.1", 1, b"t")
        time.sleep(0.05)
        self.assertLessEqual(
            threading.active_count() - before,
            2,
            "aborts must not spawn a thread each",
        )
        for lifetime in lifetimes:
            lifetime.end_lease()


class TestNoUnboundedWaits(unittest.TestCase):
    """Nothing on the ownership path may wait on a peer without a bound.

    The barrier runs inside the scheduler loop and inside the one thread that
    honours ABORT_ACKs, so a wedged peer must degrade, never stall.
    """

    def test_abort_notification_never_blocks_the_scheduler_loop(self):
        # These sockets have no ZMQ send timeout, so a backed-up peer would
        # otherwise stall the whole decode engine indefinitely.
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])
        used_nonblocking = []

        class WouldBlock:
            def send_multipart(self, parts, flags=0):
                used_nonblocking.append(bool(flags & zmq.NOBLOCK))
                raise zmq.Again()

        receiver._connect_to_bootstrap_server = lambda info: (
            WouldBlock(),
            threading.Lock(),
        )

        started = time.monotonic()
        self.assertFalse(receiver.advance_failure_quiescence())
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(used_nonblocking, [True])
        # The dropped notification must be retried, not silently lost.
        self.assertEqual(receiver._last_abort_send, float("-inf"))

    def test_bookkeeping_thread_survives_a_faulty_entry(self):
        # It is the only thread that can honour an ABORT_ACK: if it dies, every
        # decode peer of this instance waits out its full deadline forever.
        mgr = make_prefill_manager()
        mgr._start_transfer_bookkeeping()
        thread = self._bookkeeping_thread()
        sent = []
        mgr._send_manager_message = lambda ip, port, parts, **kw: sent.append(parts)

        class Faulty(RoomTransferLifetime):
            def is_quiesced(self):
                raise RuntimeError("backend bug")

        mgr.request_abort_ack(Faulty(), 1, "127.0.0.1", 1, b"bad")
        healthy = RoomTransferLifetime()
        healthy.close()
        mgr.request_abort_ack(healthy, 2, "127.0.0.1", 1, b"good")

        for _ in range(200):
            if sent:
                break
            time.sleep(0.01)
        self.assertTrue(thread.is_alive())
        self.assertEqual(
            sent,
            [[b"ABORT_ACK", b"2", b"good"]],
            "a faulty entry must not starve the ones behind it",
        )

    def test_a_backed_up_peer_does_not_delay_other_peers(self):
        mgr = make_prefill_manager()
        mgr._start_transfer_bookkeeping()
        delivered, stuck = [], {"on": True}

        def send(ip, port, parts, nonblocking=False):
            self.assertTrue(nonblocking, "ACKs must not block on a peer")
            if parts[1] == b"1" and stuck["on"]:
                raise zmq.Again()
            delivered.append(parts[1])

        mgr._send_manager_message = send
        for room in (1, 2, 3):
            lifetime = RoomTransferLifetime()
            lifetime.close()
            mgr.request_abort_ack(lifetime, room, "127.0.0.1", 1, b"t")

        for _ in range(200):
            if len(delivered) >= 2:
                break
            time.sleep(0.01)
        self.assertEqual(sorted(delivered), [b"2", b"3"])

        stuck["on"] = False
        for _ in range(200):
            if len(delivered) >= 3:
                break
            time.sleep(0.01)
        self.assertEqual(
            sorted(delivered),
            [b"1", b"2", b"3"],
            "the ACK must be retried, not dropped",
        )

    def test_a_backend_fault_releases_the_request_instead_of_the_engine(self):
        # The barrier is a safety optimisation: a bug in it must not take down
        # the scheduler, nor pin a request's pages forever.
        poller = SimpleNamespace(
            poll=Mock(return_value=KVPoll.Failed),
            advance_failure_quiescence=Mock(side_effect=RuntimeError("backend bug")),
            is_failure_quiescing=Mock(return_value=True),
        )
        with patch(
            "sglang.srt.disaggregation.utils.dist.all_reduce",
            side_effect=lambda tensor, **kw: None,
        ):
            self.assertEqual(poll_and_all_reduce([poller], object()), [KVPoll.Failed])

    @staticmethod
    def _bookkeeping_thread():
        return next(
            t
            for t in threading.enumerate()
            if t.name == "MooncakeTransferBookkeeping" and t.is_alive()
        )


class TestStagingComposition(unittest.TestCase):
    """Staging teardown is upstream's (release_room); this PR gates when it runs.

    release_room() drains the scatter stream and frees the room's reservations on
    every teardown, which covers local CUDA work. It cannot know about a remote
    prefill still writing into those buffers over RDMA -- that is what the ACK
    gate provides, by making the terminal transition (and therefore teardown)
    unreachable until every peer has confirmed.
    """

    def _handler(self):
        allocator = SimpleNamespace(
            free=Mock(),
            get_watermark=Mock(return_value=(0, 0)),
            _scatter_stream=None,
        )
        staged = SimpleNamespace(
            chunk_staging_infos=[(11, 0, 0, 128, 1), (12, 128, 0, 256, 1)],
            prefill_info=SimpleNamespace(attn_tp_size=2),
            bootstrap_infos=[],
            _connect_to_bootstrap_server=Mock(),
        )
        decode_req = SimpleNamespace(
            kv_receiver=staged, req=SimpleNamespace(bootstrap_room=ROOM)
        )
        handler = DecodeStagingHandler(
            kv_manager=SimpleNamespace(
                _staging_ctx=SimpleNamespace(
                    room_receivers={ROOM: staged}, room_bootstrap={ROOM: []}
                )
            ),
            staging_allocator=allocator,
            kv_buffer_info={},
            decode_tp=1,
            total_kv_heads=8,
            tp_rank=0,
            scheduler=SimpleNamespace(),
        )
        handler.register_decode_req(ROOM, decode_req)
        return handler, staged, allocator

    def test_teardown_releases_reservations(self):
        # Upstream's behaviour, asserted here because this PR depends on it: the
        # barrier no longer frees staging itself.
        handler, staged, allocator = self._handler()
        handler.unregister_decode_req(ROOM)
        self.assertEqual(
            sorted(call.args[0] for call in allocator.free.call_args_list), [11, 12]
        )
        self.assertEqual(staged.chunk_staging_infos, [(-1, -1, 0, -1, 0)] * 2)

    def test_teardown_is_unreachable_until_peers_confirm(self):
        # The composition that makes release_room safe against a remote writer.
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])
        self.assertFalse(
            receiver.advance_failure_quiescence(),
            "the request must not go terminal, so unregister_decode_req cannot run",
        )
        receiver.record_abort_ack(next(iter(receiver._expected_abort_acks)))
        self.assertTrue(receiver.advance_failure_quiescence())

    def test_watermark_broadcast_never_blocks_the_scheduler_loop(self):
        # release_room broadcasts a watermark from the scheduler loop over sockets
        # with no send timeout, so it must not wait for a peer.
        handler, _staged, _allocator = self._handler()
        flags_used = []

        class WouldBlock:
            def send_multipart(self, parts, flags=0):
                flags_used.append(bool(flags & zmq.NOBLOCK))
                raise zmq.Again()

        handler.register_wm_subscriber(
            SimpleNamespace(
                bootstrap_infos=[{"rank": 0}],
                _connect_to_bootstrap_server=lambda info: (
                    WouldBlock(),
                    threading.Lock(),
                ),
            ),
            "session",
        )
        started = time.monotonic()
        handler.unregister_decode_req(ROOM)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(flags_used and all(flags_used))


class TestDecodeOwnership(unittest.TestCase):
    """The decode's pages are the transfer destination, so peers must confirm."""

    def test_receiver_without_metadata_is_immediately_quiesced(self):
        # This is what makes an abort during bootstrap/preallocation terminal in
        # one step: no prefill rank has been told where to write.
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, metadata_sent=False)
        self.assertTrue(receiver.advance_failure_quiescence())

    def test_receiver_waits_for_every_peer_then_releases(self):
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}, {"rank": 1}])

        self.assertFalse(receiver.advance_failure_quiescence())
        tokens = sorted(receiver._expected_abort_acks)
        self.assertEqual(len(tokens), 2, "one nonce per prefill rank")

        receiver.record_abort_ack(tokens[0])
        self.assertFalse(receiver.advance_failure_quiescence())
        receiver.record_abort_ack(tokens[1])
        self.assertTrue(receiver.advance_failure_quiescence())

    def test_abort_notifications_carry_a_stable_token_and_are_retried(self):
        mgr = make_decode_manager()
        sockets = {0: Mock(), 1: Mock()}
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}, {"rank": 1}])
        receiver._connect_to_bootstrap_server = lambda info: (
            sockets[info["rank"]],
            threading.Lock(),
        )

        clock = [0.0]
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            receiver._send_abort_notification()
            tokens = {
                rank: socket.send_multipart.call_args.args[0][4]
                for rank, socket in sockets.items()
            }
            clock[0] = ABORT_RETRY_INTERVAL_S / 2
            receiver._send_abort_notification()  # rate limited
            receiver.record_abort_ack(tokens[0])
            clock[0] = ABORT_RETRY_INTERVAL_S * 2
            receiver._send_abort_notification()  # retry rank 1 only

        self.assertEqual(sockets[0].send_multipart.call_count, 1)
        self.assertEqual(sockets[1].send_multipart.call_count, 2)
        self.assertEqual(
            sockets[1].send_multipart.call_args_list[1].args[0][4],
            tokens[1],
            "a retry must reuse the nonce so the ACK still matches",
        )

    def test_quiescence_is_bounded_when_a_peer_never_acknowledges(self):
        mgr = make_decode_manager(quiesce_timeout=0.05)
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])

        with patch(
            "sglang.srt.disaggregation.mooncake.conn.TRANSFER_QUIESCE_TIMEOUTS"
        ) as metric:
            self.assertFalse(receiver.advance_failure_quiescence())
            time.sleep(0.06)
            self.assertTrue(
                receiver.advance_failure_quiescence(),
                "a dead prefill must not pin decode pages forever",
            )
            metric.inc.assert_called_once_with()

    def test_legacy_prefill_ack_stops_the_wait(self):
        # A prefill that predates the barrier ACKs immediately and without a
        # nonce. It cannot promise a drain, so waiting for it is pointless;
        # degrade to the previous behaviour instead of stalling.
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])
        self.assertFalse(receiver.advance_failure_quiescence())
        receiver.record_abort_ack(None)
        self.assertTrue(receiver.advance_failure_quiescence())

    def test_stale_ack_for_another_request_is_ignored(self):
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])
        receiver.advance_failure_quiescence()
        receiver.record_abort_ack(b"nonce-from-a-previous-request")
        self.assertFalse(receiver.advance_failure_quiescence())

    def test_abort_ack_routes_to_the_registered_receiver_only(self):
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])
        receiver.record_abort_ack = Mock()
        mgr._receivers[ROOM] = receiver

        mgr._handle_abort_ack([b"ABORT_ACK", str(ROOM).encode("ascii"), b"tok"])
        receiver.record_abort_ack.assert_called_once_with(b"tok")
        # Unknown rooms must not raise on the transport thread.
        mgr._handle_abort_ack([b"ABORT_ACK", b"999", b"tok"])

    def test_only_one_receiver_owns_a_room(self):
        mgr = make_decode_manager()
        first = make_receiver(mgr)
        self.assertTrue(mgr.register_receiver(first))
        second = make_receiver(mgr)
        self.assertFalse(
            mgr.register_receiver(second),
            "colliding rooms must not let a second receiver take over",
        )
        self.assertIs(mgr._receivers[ROOM], first)

        # The loser must not evict the owner's registration on cleanup.
        mgr.unregister_receiver(second)
        self.assertIs(mgr._receivers[ROOM], first)
        mgr.unregister_receiver(first)
        self.assertTrue(mgr.register_receiver(second))

    def test_a_receiver_that_lost_the_room_touches_no_shared_state(self):
        handler = Mock()
        mgr = make_decode_manager(staging_handler=handler)
        loser = make_receiver(mgr, metadata_sent=False)
        loser._owns_room = False
        self.assertTrue(loser.advance_failure_quiescence())
        handler.begin_abort.assert_not_called()
        loser.clear = MooncakeKVReceiver.clear.__get__(loser)
        handler.forget_abort.assert_not_called()

    def test_late_transport_messages_are_dropped_while_tearing_down(self):
        mgr = make_decode_manager()
        receiver = make_receiver(mgr)
        mgr._receivers[ROOM] = receiver
        self.assertFalse(mgr._is_tearing_down(ROOM))
        receiver._close_barrier()
        self.assertTrue(mgr._is_tearing_down(ROOM))
        self.assertFalse(mgr._is_tearing_down(999))


class TestBarrierLevels(unittest.TestCase):
    """What happens when proof of quiescence never arrives.

    WARN trades a narrow, loud corruption window for liveness; STRICT refuses to
    reuse a page it cannot prove is idle, and escalates instead.
    """

    def _stuck_sender(self, barrier, max_unquiesced=256):
        mgr = make_prefill_manager(quiesce_timeout=0.01, barrier=barrier)
        mgr.max_unquiesced = max_unquiesced
        sender = make_sender(mgr)
        lifetime = mgr._room_lifetime(ROOM, create=False)
        self.assertTrue(lifetime.try_lease())  # a worker that never returns
        return mgr, sender

    @staticmethod
    def _arm_and_expire(sender):
        """Arm the deadline, then let it pass."""
        with patch("sglang.srt.disaggregation.mooncake.conn.TRANSFER_QUIESCE_TIMEOUTS"):
            sender.advance_failure_quiescence()
        time.sleep(0.02)

    def test_warn_releases_once_and_loudly(self):
        mgr, sender = self._stuck_sender(TransferBarrierLevel.WARN)
        self.assertFalse(sender.advance_failure_quiescence())
        time.sleep(0.02)
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.TRANSFER_QUIESCE_TIMEOUTS"
        ) as metric:
            self.assertTrue(sender.advance_failure_quiescence())
            self.assertTrue(sender.advance_failure_quiescence())
            metric.inc.assert_called_once_with()

    def test_strict_never_releases_without_proof(self):
        mgr, sender = self._stuck_sender(TransferBarrierLevel.STRICT)
        self._arm_and_expire(sender)
        with patch("sglang.srt.disaggregation.mooncake.conn.TRANSFER_QUIESCE_TIMEOUTS"):
            for _ in range(5):
                self.assertFalse(
                    sender.advance_failure_quiescence(),
                    "STRICT must not hand back a page it cannot prove is idle",
                )
        self.assertIn(
            ROOM, mgr._unquiesced_rooms, "the stuck room must be tracked for escalation"
        )

    def test_strict_escalates_rather_than_leaking_forever(self):
        # Withholding pages is only safe while there are few of them; past the
        # cap the worker fails so a restart reclaims them all at once.
        mgr, sender = self._stuck_sender(TransferBarrierLevel.STRICT, max_unquiesced=2)
        self._arm_and_expire(sender)
        with patch("sglang.srt.disaggregation.mooncake.conn.TRANSFER_QUIESCE_TIMEOUTS"):
            sender.advance_failure_quiescence()
        mgr._unquiesced_rooms.add(999)  # a second stuck request
        with self.assertRaises(RuntimeError) as cm:
            mgr.release_without_proof(1234, "third stuck request")
        self.assertIn("cannot be proven quiesced", str(cm.exception))

    def test_off_restores_legacy_behaviour(self):
        mgr, sender = self._stuck_sender(TransferBarrierLevel.OFF)
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.TRANSFER_QUIESCE_TIMEOUTS"
        ) as metric:
            self.assertTrue(sender.advance_failure_quiescence())
            metric.inc.assert_not_called()

    def test_a_receiver_that_lost_its_room_does_not_wait_for_unroutable_proof(self):
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])
        receiver._owns_room = False
        with patch("sglang.srt.disaggregation.mooncake.conn.TRANSFER_QUIESCE_TIMEOUTS"):
            self.assertTrue(
                receiver.advance_failure_quiescence(),
                "ACKs are routed to the owning receiver, so this one can never "
                "receive proof and must not wait for it",
            )


class TestStrictCannotBeDowngraded(unittest.TestCase):
    """STRICT must not degrade into the most permissive policy by accident."""

    def test_escalation_reaches_the_scheduler(self):
        # The poll path contains backend faults so a bug cannot kill the engine.
        # The escalation is not a bug: swallowing it would release exactly the
        # pages STRICT exists to withhold.
        poller = SimpleNamespace(
            poll=Mock(return_value=KVPoll.Failed),
            advance_failure_quiescence=Mock(
                side_effect=KVTransferBarrierEscalation("cannot prove idle")
            ),
            is_failure_quiescing=Mock(return_value=True),
        )
        with patch(
            "sglang.srt.disaggregation.utils.dist.all_reduce",
            side_effect=lambda tensor, **kw: None,
        ):
            with self.assertRaises(KVTransferBarrierEscalation):
                poll_and_all_reduce([poller], object())

    def test_an_ordinary_backend_fault_is_still_contained(self):
        poller = SimpleNamespace(
            poll=Mock(return_value=KVPoll.Failed),
            advance_failure_quiescence=Mock(side_effect=RuntimeError("backend bug")),
            is_failure_quiescing=Mock(return_value=True),
        )
        with patch(
            "sglang.srt.disaggregation.utils.dist.all_reduce",
            side_effect=lambda tensor, **kw: None,
        ):
            self.assertEqual(poll_and_all_reduce([poller], object()), [KVPoll.Failed])

    def test_strict_does_not_accept_a_legacy_ack_as_proof(self):
        # A peer that predates the barrier acknowledges before draining, so its
        # ACK proves nothing. WARN accepts it to keep a rolling upgrade working;
        # STRICT cannot, by definition.
        for level, accepted in (
            (TransferBarrierLevel.WARN, True),
            (TransferBarrierLevel.STRICT, False),
        ):
            with self.subTest(level=level):
                mgr = make_decode_manager(barrier=level)
                receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])
                receiver.advance_failure_quiescence()
                receiver.record_abort_ack(None)
                self.assertEqual(receiver._peers_quiesced(), accepted)


class TestRoomStateRetirement(unittest.TestCase):
    """Reclaiming a room must retire everything keyed by it."""

    def _room_with_metadata(self, mgr, room=ROOM):
        mgr.session_lock = threading.Lock()
        mgr.failed_sessions, mgr.session_failures = set(), {}
        mgr.decode_kv_args_table = {}
        mgr.resolve_kv_replica_factor = Mock()
        mgr._handle_bootstrap_metadata(
            [
                str(room).encode("ascii"),
                b"127.0.0.1",
                b"1",
                b"session:1",
                np.array([0], dtype=np.int32).tobytes(),
                b"0",
                b"",
                b"1",
                b"0",
                b"token",
            ]
        )

    def test_emergency_reclaim_retires_destinations_and_status_too(self):
        # Same hazard as the sweep, via the other reclaim path: a request drawing
        # a reclaimed room must not inherit the old decode's addresses.
        mgr = make_prefill_manager()
        self._room_with_metadata(mgr)
        self.assertIn(ROOM, mgr.transfer_infos)
        with patch("sglang.srt.disaggregation.mooncake.conn.MAX_TRACKED_ROOMS", 1):
            mgr._room_lifetime(ROOM + 1, create=True)
        self.assertNotIn(ROOM, mgr._room_lifetimes)
        self.assertNotIn(ROOM, mgr.transfer_infos)
        self.assertNotIn(ROOM, mgr.request_status)
        self.assertNotIn(ROOM, mgr.req_to_decode_prefix_len)

    def test_retirement_is_atomic_against_a_new_generation(self):
        # The lifetime and the rest of the room's state must go in one step: a
        # request created in between would have its own metadata deleted.
        mgr = make_prefill_manager(room_sweep_ttl=0.0)
        self._room_with_metadata(mgr)
        observed = {}
        real_retire = mgr._retire_rooms_locked

        def retire(rooms):
            observed["held"] = mgr._room_lifetimes_lock.locked()
            real_retire(rooms)

        mgr._retire_rooms_locked = retire
        self.assertEqual(mgr._sweep_room_lifetimes(), 1)
        self.assertTrue(observed["held"], "retirement must run under the lifetime lock")

    def test_sweep_retires_destinations_and_status_together(self):
        # Otherwise a later request drawing the same bootstrap_room inherits the
        # old decode's addresses and a stale status.
        mgr = make_prefill_manager(room_sweep_ttl=0.0)
        mgr.session_lock = threading.Lock()
        mgr.failed_sessions, mgr.session_failures = set(), {}
        mgr.decode_kv_args_table = {}
        mgr.resolve_kv_replica_factor = Mock()
        mgr._handle_bootstrap_metadata(
            [
                str(ROOM).encode("ascii"),
                b"127.0.0.1",
                b"1",
                b"session:1",
                np.array([0], dtype=np.int32).tobytes(),
                b"0",
                b"",
                b"1",
                b"0",
                b"token",
            ]
        )
        self.assertIn(ROOM, mgr.transfer_infos)
        self.assertIn(ROOM, mgr.request_status)

        self.assertEqual(mgr._sweep_room_lifetimes(), 1)
        self.assertNotIn(ROOM, mgr._room_lifetimes)
        self.assertNotIn(ROOM, mgr.transfer_infos)
        self.assertNotIn(ROOM, mgr.request_status)
        self.assertNotIn(ROOM, mgr.req_to_decode_prefix_len)


class TestRoomCollisionIsolation(unittest.TestCase):
    """A receiver that lost a room collision must not damage the owner."""

    def test_the_loser_does_not_tear_down_shared_state(self):
        mgr = make_decode_manager()
        owner = make_receiver(mgr)
        self.assertTrue(mgr.register_receiver(owner))
        loser = make_receiver(mgr)
        loser._owns_room = mgr.register_receiver(loser)
        self.assertFalse(loser._owns_room)

        mgr.request_status[ROOM] = KVPoll.WaitingForInput
        mgr.required_prefill_response_num_table = {ROOM: 1}
        mgr.prefill_response_tracker = {ROOM: set()}

        loser.clear()

        self.assertIn(ROOM, mgr.request_status, "the owner's status must survive")
        self.assertIn(ROOM, mgr.required_prefill_response_num_table)
        self.assertIs(mgr._receivers[ROOM], owner)


class TestWatermarkDelivery(unittest.TestCase):
    """A dropped watermark must be retried; it can gate a prefill's progress."""

    def _handler(self, socket):
        allocator = SimpleNamespace(
            free=Mock(), get_watermark=Mock(return_value=(3, 128)), _scatter_stream=None
        )
        handler = DecodeStagingHandler(
            kv_manager=SimpleNamespace(
                _staging_ctx=SimpleNamespace(room_receivers={}, room_bootstrap={})
            ),
            staging_allocator=allocator,
            kv_buffer_info={},
            decode_tp=1,
            total_kv_heads=8,
            tp_rank=0,
            scheduler=SimpleNamespace(),
        )
        handler.register_wm_subscriber(
            SimpleNamespace(
                bootstrap_infos=[{"rank": 0}],
                _connect_to_bootstrap_server=lambda info: (socket, threading.Lock()),
            ),
            "session",
        )
        return handler

    def test_a_backpressured_watermark_is_resent(self):
        state = {"blocked": True}
        sent = []

        class Socket:
            def send_multipart(self, parts, flags=0):
                if state["blocked"]:
                    raise zmq.Again()
                sent.append(parts)

        handler = self._handler(Socket())
        handler._free_and_send_watermark(
            1, SimpleNamespace(req=SimpleNamespace(bootstrap_room=ROOM))
        )
        self.assertEqual(sent, [], "the send was backpressured")
        self.assertTrue(handler._wm_retry, "the subscriber must be remembered")

        state["blocked"] = False
        handler.retry_pending_watermarks()
        self.assertEqual(len(sent), 1, "the current watermark must be resent")
        self.assertEqual(sent[0][1:3], [b"3", b"128"])
        self.assertFalse(handler._wm_retry)

    def test_retry_is_free_when_nothing_is_pending(self):
        sent = []

        class Socket:
            def send_multipart(self, parts, flags=0):
                sent.append(parts)

        handler = self._handler(Socket())
        handler.retry_pending_watermarks()
        self.assertEqual(sent, [])


class TestOwedAckBookkeeping(unittest.TestCase):
    """A peer that never drains its socket must not accumulate work forever."""

    def test_an_undeliverable_ack_is_eventually_given_up_on(self):
        mgr = make_prefill_manager()
        mgr._start_transfer_bookkeeping()
        mgr._send_manager_message = Mock(side_effect=zmq.Again())
        lifetime = RoomTransferLifetime()
        lifetime.close()
        mgr.request_abort_ack(lifetime, ROOM, "127.0.0.1", 1, b"t")
        with mgr._abort_ack_lock:
            self.assertEqual(len(mgr._abort_ack_pending), 1)
        # Expire it rather than waiting out ABORT_ACK_MAX_AGE_S.
        with patch("sglang.srt.disaggregation.mooncake.conn.ABORT_ACK_MAX_AGE_S", 0.0):
            mgr.request_abort_ack(lifetime, ROOM + 1, "127.0.0.1", 1, b"t2")
        for _ in range(300):
            with mgr._abort_ack_lock:
                if len(mgr._abort_ack_pending) < 2:
                    break
            time.sleep(0.01)
        with mgr._abort_ack_lock:
            self.assertLess(len(mgr._abort_ack_pending), 2)

    def test_owed_acks_are_capped(self):
        mgr = make_prefill_manager()
        mgr._abort_ack_queue = __import__("queue").Queue()
        mgr._abort_ack_pending = set()
        mgr._abort_ack_lock = threading.Lock()
        with patch("sglang.srt.disaggregation.mooncake.conn.MAX_PENDING_ABORT_ACKS", 4):
            for room in range(50):
                mgr.request_abort_ack(None, room, "127.0.0.1", 1, b"t")
        self.assertLessEqual(len(mgr._abort_ack_pending), 4)


class TestPollGating(unittest.TestCase):
    """How the deferral is surfaced to the scheduler."""

    @staticmethod
    def _reduce_min(tensor, **_kwargs):
        return None  # single rank: MIN over one rank is the identity

    def test_failed_is_withheld_until_quiesced_then_reported(self):
        poller = SimpleNamespace(
            poll=Mock(return_value=KVPoll.Failed),
            advance_failure_quiescence=Mock(side_effect=[False, True]),
            is_failure_quiescing=Mock(return_value=True),
        )
        with patch(
            "sglang.srt.disaggregation.utils.dist.all_reduce",
            side_effect=self._reduce_min,
        ):
            self.assertEqual(
                poll_and_all_reduce([poller], object()), [KVPoll.Transferring]
            )
            self.assertEqual(poll_and_all_reduce([poller], object()), [KVPoll.Failed])

    def test_healthy_polls_take_no_extra_collective(self):
        poller = SimpleNamespace(
            poll=Mock(return_value=KVPoll.WaitingForInput),
            advance_failure_quiescence=Mock(),
            is_failure_quiescing=Mock(return_value=False),
        )
        with patch(
            "sglang.srt.disaggregation.utils.dist.all_reduce",
            side_effect=self._reduce_min,
        ) as all_reduce:
            self.assertEqual(
                poll_and_all_reduce([poller], object()), [KVPoll.WaitingForInput]
            )
        self.assertEqual(all_reduce.call_count, 1)
        poller.advance_failure_quiescence.assert_not_called()

    def test_a_quiescing_rank_forces_the_group_to_fail(self):
        # Otherwise one rank could release pages while another kept transferring.
        poller = SimpleNamespace(
            poll=Mock(return_value=KVPoll.WaitingForInput),
            advance_failure_quiescence=Mock(return_value=True),
            is_failure_quiescing=Mock(return_value=True),
        )
        with patch(
            "sglang.srt.disaggregation.utils.dist.all_reduce",
            side_effect=self._reduce_min,
        ):
            self.assertEqual(poll_and_all_reduce([poller], object()), [KVPoll.Failed])

    def test_bootstrap_queue_tolerates_a_deferred_terminal_state(self):
        # Regression guard: pop_bootstrapped and handle_pending_bootstrap raise on
        # unexpected poll states, so the deferral must be an accepted state there.
        from sglang.srt.disaggregation.prefill import (
            SchedulerDisaggregationPrefillMixin,
        )

        handle = SchedulerDisaggregationPrefillMixin.handle_pending_bootstrap
        self.assertFalse(
            handle(SimpleNamespace(), SimpleNamespace(rid="r"), KVPoll.Transferring)
        )

    def test_hicache_wrapper_delegates_the_barrier(self):
        from sglang.srt.disaggregation.decode_hicache_mixin import (
            HiCacheRestoreGatedKVReceiver,
        )

        receiver = SimpleNamespace(
            poll=Mock(return_value=KVPoll.Failed),
            advance_failure_quiescence=Mock(return_value=False),
            is_failure_quiescing=Mock(return_value=True),
        )
        gated = HiCacheRestoreGatedKVReceiver(
            SimpleNamespace(kv_receiver=receiver, hicache_restore_status=None)
        )
        self.assertFalse(gated.advance_failure_quiescence())
        self.assertTrue(gated.is_failure_quiescing())


class TestFailureDiagnostics(unittest.TestCase):
    """Releasing pages safely must not cost the operator the real reason."""

    def test_propagated_failures_keep_their_attribution(self):
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])
        # A failure reported by the prefill (or another rank) drives quiescence,
        # but must not be relabelled as a local user abort.
        receiver.advance_failure_quiescence()
        mgr.record_failure.assert_not_called()

    def test_user_abort_still_records_its_reason(self):
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])
        receiver.abort_notified = False
        MooncakeKVReceiver.abort(receiver)
        mgr.record_failure.assert_called_once()
        self.assertIn("Aborted by AbortReq", mgr.record_failure.call_args.args[1])


class TestChunkedPrefillAbort(unittest.TestCase):
    """Aborting a chunked prefill must not free pages an earlier chunk is reading."""

    @staticmethod
    def _scheduler(req, mode):
        return SimpleNamespace(
            _pending_chunked_abort_req=req,
            chunked_req=req,
            enable_hicache_storage=False,
            tree_cache=Mock(),
            disaggregation_mode=mode,
            disagg_prefill_inflight_queue=[],
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(send_output=Mock())
            ),
        )

    @staticmethod
    def _chunked_req():
        return SimpleNamespace(
            rid="chunked",
            disagg_kv_sender=Mock(),
            time_stats=SimpleNamespace(trace_ctx=Mock()),
            to_finish=object(),
            skip_radix_cache_insert=False,
        )

    def _run(self, mode):
        from sglang.srt.managers import scheduler as scheduler_mod

        req = self._chunked_req()
        scheduler = self._scheduler(req, mode)
        with patch.object(
            scheduler_mod, "prepare_abort"
        ) as prepare_abort, patch.object(
            scheduler_mod, "release_kv_cache"
        ) as release_kv_cache:
            scheduler_mod.Scheduler.process_pending_chunked_abort(scheduler)
        return req, scheduler, prepare_abort, release_kv_cache

    def test_prefill_defers_release_to_the_inflight_queue(self):
        req, scheduler, prepare_abort, release_kv_cache = self._run(
            DisaggregationMode.PREFILL
        )
        prepare_abort.assert_called_once()
        req.disagg_kv_sender.abort.assert_called_once_with()
        release_kv_cache.assert_not_called()
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])
        self.assertTrue(
            req.skip_radix_cache_insert,
            "the aborted partial prefix must not enter the radix cache",
        )
        self.assertIsNone(scheduler.chunked_req)
        self.assertIsNone(scheduler._pending_chunked_abort_req)

    def test_non_disagg_release_is_unchanged(self):
        req, scheduler, _prepare_abort, release_kv_cache = self._run(
            DisaggregationMode.DECODE
        )
        release_kv_cache.assert_called_once()
        self.assertEqual(release_kv_cache.call_args.kwargs, {"is_insert": False})
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()

    def test_deferral_is_idempotent(self):
        from sglang.srt.managers import scheduler as scheduler_mod

        req = self._chunked_req()
        scheduler = self._scheduler(req, DisaggregationMode.PREFILL)
        with patch.object(scheduler_mod, "prepare_abort"), patch.object(
            scheduler_mod, "release_kv_cache"
        ):
            scheduler_mod.Scheduler.process_pending_chunked_abort(scheduler)
            scheduler._pending_chunked_abort_req = req
            scheduler.chunked_req = req
            scheduler_mod.Scheduler.process_pending_chunked_abort(scheduler)
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])


class TestTransferInfoWire(unittest.TestCase):
    """The abort nonce is an optional trailing frame."""

    def test_metadata_without_a_token_parses_as_legacy(self):
        msg = [
            b"7",
            b"127.0.0.1",
            b"1",
            b"session",
            np.array([0], dtype=np.int32).tobytes(),
            b"0",
            b"",
            b"1",
            b"0",
        ]
        info = TransferInfo.from_zmq(msg)
        self.assertEqual(info.room, 7)
        self.assertEqual(info.abort_token, b"")

    def test_metadata_with_a_token_parses_it(self):
        msg = [
            b"7",
            b"127.0.0.1",
            b"1",
            b"session",
            np.array([0], dtype=np.int32).tobytes(),
            b"0",
            b"",
            b"1",
            b"0",
            b"nonce",
        ]
        self.assertEqual(TransferInfo.from_zmq(msg).abort_token, b"nonce")


if __name__ == "__main__":
    unittest.main()
