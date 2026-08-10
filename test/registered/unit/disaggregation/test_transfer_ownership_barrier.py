"""Tests for the KV transfer ownership barrier.

The barrier keeps a failed request's KV pages allocated until no native
transfer work can still read or write them. The properties under test are:

* pages stay owned while transfer work is in flight (safety);
* proof that never arrives is loud and bounded: reported after a deadline,
  escalated once too many requests are stuck -- but never a silent release;
* the deferral only happens where the scheduler can handle it, and never
  weakens the failure diagnostics the operator sees.
"""

import concurrent.futures
import threading
import time
import unittest
from collections import OrderedDict, defaultdict
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
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
    RoomTransferLifetime,
    TransferInfo,
)
from sglang.srt.disaggregation.utils import DisaggregationMode, poll_and_all_reduce
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


ROOM = 7


def make_prefill_manager(room_sweep_ttl=300.0):
    """A prefill-side manager with only its ownership-barrier state wired up."""
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.disaggregation_mode = DisaggregationMode.PREFILL
    mgr._unquiesced_rooms = set()
    mgr._unquiesced_lock = threading.Lock()
    mgr.bootstrap_timeout = 300
    mgr._room_sweep_ttl = room_sweep_ttl
    mgr._room_lifetimes = OrderedDict()
    mgr._room_lifetimes_lock = threading.Lock()
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
    mgr._staging_outstanding = defaultdict(int)
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
    receiver._metadata_sent = False
    receiver._quiescing = False
    receiver._quiesce_complete = False
    receiver._quiesce_deadline = float("inf")
    receiver._abort_targets = []
    receiver._expected_abort_acks = set()
    receiver._received_abort_acks = set()
    receiver._last_abort_send = float("-inf")
    receiver._owns_room = True
    receiver._abort_lock = threading.Lock()
    receiver._connect_to_bootstrap_server = Mock(
        side_effect=lambda info: (Mock(), threading.Lock())
    )
    if metadata_sent:
        targets = receiver._abort_targets_snapshot()
        if targets:
            for _info, token in targets:
                receiver._record_metadata_exposure(token)
        else:
            receiver._metadata_sent = True
    return receiver


def make_decode_manager(staging_handler=None):
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr._receivers = {}
    mgr._receivers_lock = threading.Lock()
    mgr._staging_handler = staging_handler
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
                    dst_device_kv_indices=None,
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
                requires_dcp_relayout=False,
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
        self.assertNotIn(ROOM, mgr._room_lifetimes)
        self.assertNotIn(ROOM, mgr.request_status)

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

    def test_lease_lifecycle(self):
        lifetime = RoomTransferLifetime()
        self.assertTrue(lifetime.try_lease())
        lifetime.close()

        self.assertFalse(lifetime.try_lease(), "a closed room admits no work")
        self.assertFalse(lifetime.is_quiesced(), "a lease is still outstanding")

        lifetime.end_lease()
        self.assertTrue(lifetime.is_quiesced())

    def test_abort_token_authorization(self):
        lifetime = RoomTransferLifetime()
        # A room with no registered decode metadata accepts any abort: nothing
        # can be in flight, and refusing would drop the abort entirely.
        self.assertTrue(lifetime.authorizes_abort(b"unknown"))
        lifetime.add_abort_token(b"mine")
        self.assertTrue(lifetime.authorizes_abort(b"mine"))
        self.assertFalse(lifetime.authorizes_abort(b"someone-elses"))
        # A tokenless abort cannot be authenticated; honour it, fail-safe.
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

    def test_dcp_send_failure_waits_for_a_running_sibling(self):
        # Per-call-site guard: send_kvcache_dcp's custom-mem-pool branch must
        # route through submit_transfer_calls like the other send paths. An
        # early return on the first failed layer would end the worker's room
        # lease while a sibling layer is still writing into decode's pages.
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.enable_custom_mem_pool = True
        mgr.kv_args = SimpleNamespace(
            page_size=1, kv_data_ptrs=[0x1, 0x2], kv_layer_ids=[]
        )
        mgr.get_mla_kv_ptrs_with_pp = lambda src, dst: (src, dst, None)

        sibling_running = threading.Event()
        release_sibling = threading.Event()
        returned = threading.Event()
        result = []
        FAILING_LAYER_PTR = 0x1

        def transfer_data(session_id, blocks):
            if blocks[0][0] == FAILING_LAYER_PTR:
                # Fail only once the sibling layer is genuinely mid-write, so
                # cancelling a still-pending future cannot pass this trivially.
                sibling_running.wait(5)
                return -1
            sibling_running.set()
            release_sibling.wait(5)
            return 0

        mgr._transfer_data = transfer_data
        plan = SimpleNamespace(
            src_token_indices=np.array([0]), dst_token_indices=np.array([0])
        )
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.build_dcp_token_transfer_plan",
            return_value=plan,
        ), concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

            def send():
                result.append(
                    mgr.send_kvcache_dcp(
                        "session",
                        np.array([0], dtype=np.int32),
                        [0x11, 0x22],
                        np.array([0], dtype=np.int32),
                        dcp_token_item_lens=[16, 16],
                        dst_dcp_size=1,
                        dst_dcp_rank=0,
                        src_page_offset=0,
                        decode_prefix_len=0,
                        num_kv_tokens=1,
                        executor=executor,
                        dst_layer_ids=[],
                    )
                )
                returned.set()

            sender = threading.Thread(target=send)
            sender.start()
            self.assertTrue(sibling_running.wait(1))
            self.assertFalse(
                returned.wait(0.05), "returned while a sibling layer still ran"
            )
            release_sibling.set()
            self.assertTrue(returned.wait(1))
            sender.join()

        self.assertEqual(result, [-1], "the failing layer status must be reported")


class TestPrefillOwnership(unittest.TestCase):
    """The prefill's pages are the transfer source."""

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

    def test_bootstrapping_sender_is_immediately_quiesced(self):
        # Nothing has been submitted, so the request can fail without any wait.
        # This is what keeps KVPoll.Transferring out of the bootstrap queues.
        mgr = make_prefill_manager()
        sender = make_sender(mgr, status=KVPoll.Bootstrapping)
        self.assertTrue(sender.advance_failure_quiescence())

    def test_tracked_rooms_are_bounded_without_evicting_a_live_room(self):
        mgr = make_prefill_manager()
        live = mgr._room_lifetime(-1, create=True)
        mgr.open_room_transfers(-1)
        self.assertTrue(live.try_lease())
        with patch("sglang.srt.disaggregation.mooncake.conn.MAX_TRACKED_ROOMS", 4):
            for room in range(200):
                mgr._close_room_for_abort(room, b"").created_at = 0
            # ... and rooms known only from decode metadata, which no local
            # sender will ever release, must be bounded too.
            for room in range(1000, 1200):
                mgr._room_lifetime(room, create=True).created_at = 0
        self.assertLessEqual(len(mgr._room_lifetimes), 8)
        self.assertIn(
            -1, mgr._room_lifetimes, "a claimed, leased room must never be evicted"
        )

    def test_sweep_reclaims_abandoned_rooms_but_keeps_young_and_claimed_ones(self):
        mgr = make_prefill_manager(room_sweep_ttl=300.0)
        mgr._room_lifetime(1, create=True).created_at = 0  # abandoned: reclaim
        mgr._room_lifetime(2, create=True)  # young: a sender may still arrive
        mgr.open_room_transfers(3)  # claimed: its sender releases it
        mgr._room_lifetime(3, create=False).created_at = 0

        self.assertEqual(mgr._sweep_room_lifetimes(), 1)
        self.assertEqual(sorted(mgr._room_lifetimes), [2, 3])


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
            b"",  # device_kv_indices (HiSparse split-index transfers only)
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

    def test_failed_metadata_send_does_not_expect_unexposed_peers(self):
        # A peer whose metadata send failed never learned where to write, so
        # waiting for its ACK would withhold the pages forever.
        mgr = make_decode_manager()
        mgr.enable_staging = False
        receiver = make_receiver(
            mgr,
            bootstrap_infos=[
                {
                    "rank": rank,
                    "rank_ip": "127.0.0.1",
                    "rank_port": rank + 1,
                    "is_dummy": False,
                }
                for rank in range(3)
            ],
            metadata_sent=False,
        )
        receiver.session_id = "decode"
        receiver.required_dst_info_num = 3
        sockets = {rank: Mock() for rank in range(3)}
        sockets[1].send_multipart.side_effect = zmq.ZMQError()
        connected = []

        def connect(info):
            connected.append(info["rank"])
            return sockets[info["rank"]], threading.Lock()

        receiver._connect_to_bootstrap_server = connect
        receiver.send_metadata(np.array([123], dtype=np.int32))

        tokens = {info["rank"]: token for info, token in receiver._abort_targets}
        self.assertEqual(connected, [0, 1])
        self.assertEqual(receiver._expected_abort_acks, {tokens[0]})
        self.assertTrue(receiver._metadata_sent)
        self.assertNotIn(tokens[1], receiver._expected_abort_acks)
        self.assertNotIn(tokens[2], receiver._expected_abort_acks)

    def test_unauthorized_acks_are_not_proof(self):
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])
        self.assertFalse(receiver.advance_failure_quiescence())

        # A nonce minted for a previous occupant of this recycled room.
        receiver.record_abort_ack(b"nonce-from-a-previous-request")
        # A tokenless ACK from a peer that predates the barrier: it acknowledges
        # before its transfers drain, so it proves nothing.
        receiver.record_abort_ack(None)
        self.assertFalse(receiver.advance_failure_quiescence())

        receiver.record_abort_ack(next(iter(receiver._expected_abort_acks)))
        self.assertTrue(receiver.advance_failure_quiescence())

    def test_abort_ack_routes_to_the_registered_receiver_only(self):
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])
        receiver.record_abort_ack = Mock()
        mgr._receivers[ROOM] = receiver

        mgr._handle_abort_ack([b"ABORT_ACK", str(ROOM).encode("ascii"), b"tok"])
        receiver.record_abort_ack.assert_called_once_with(b"tok")
        # Unknown rooms must not raise on the transport thread.
        mgr._handle_abort_ack([b"ABORT_ACK", b"999", b"tok"])

    def test_late_transport_messages_are_dropped_while_tearing_down(self):
        mgr = make_decode_manager()
        receiver = make_receiver(mgr)
        mgr._receivers[ROOM] = receiver
        self.assertFalse(mgr._is_tearing_down(ROOM))
        receiver._close_barrier()
        self.assertTrue(mgr._is_tearing_down(ROOM))
        self.assertFalse(mgr._is_tearing_down(999))

    def test_user_abort_still_records_its_reason(self):
        # Releasing pages safely must not cost the operator the real reason.
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])
        receiver.abort_notified = False
        MooncakeKVReceiver.abort(receiver)
        mgr.record_failure.assert_called_once()
        self.assertIn("Aborted by AbortReq", mgr.record_failure.call_args.args[1])


class TestStrictBarrier(unittest.TestCase):
    """What happens when proof of quiescence never arrives.

    The pages are never released without proof: the wait is reported after the
    quiescence deadline, and once too many requests are stuck the worker fails,
    so a restart reclaims every withheld page safely at once.
    """

    def test_prefill_pages_are_withheld_when_a_transfer_never_returns(self):
        mgr = make_prefill_manager()
        sender = make_sender(mgr)
        lifetime = mgr._room_lifetime(ROOM, create=False)
        self.assertTrue(lifetime.try_lease())  # a worker that never returns

        with patch("sglang.srt.disaggregation.mooncake.conn.QUIESCE_TIMEOUT_S", 0.0):
            with patch(
                "sglang.srt.disaggregation.mooncake.conn.TRANSFER_QUIESCE_TIMEOUTS"
            ) as metric:
                with patch(
                    "sglang.srt.disaggregation.mooncake.conn.logger.error"
                ) as error:
                    for _ in range(5):
                        self.assertFalse(
                            sender.advance_failure_quiescence(),
                            "a page that cannot be proven idle must not be reused",
                        )
        metric.inc.assert_called_once_with()
        error.assert_called_once()
        self.assertIn(
            ROOM, mgr._unquiesced_rooms, "the stuck room must be tracked for escalation"
        )

        # Proof arriving late still recovers the request.
        lifetime.end_lease()
        self.assertTrue(sender.advance_failure_quiescence())
        self.assertNotIn(ROOM, mgr._unquiesced_rooms)

    def test_decode_pages_are_withheld_when_a_peer_never_acknowledges(self):
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])

        with patch("sglang.srt.disaggregation.mooncake.conn.QUIESCE_TIMEOUT_S", 0.0):
            with patch(
                "sglang.srt.disaggregation.mooncake.conn.TRANSFER_QUIESCE_TIMEOUTS"
            ) as metric:
                for _ in range(3):
                    self.assertFalse(
                        receiver.advance_failure_quiescence(),
                        "a dead prefill must not cause a silent release",
                    )
        metric.inc.assert_called_once_with()
        self.assertIn(ROOM, mgr._unquiesced_rooms)

    def test_escalation_once_too_many_requests_are_stuck(self):
        # Withholding pages is only safe while there are few of them; past the
        # cap the worker fails so a restart reclaims them all at once.
        mgr = make_prefill_manager()
        with patch("sglang.srt.disaggregation.mooncake.conn.MAX_UNQUIESCED_ROOMS", 2):
            with patch(
                "sglang.srt.disaggregation.mooncake.conn.TRANSFER_QUIESCE_TIMEOUTS"
            ):
                mgr.report_unquiesced(1, "first stuck request")
                with self.assertRaises(KVTransferBarrierEscalation) as cm:
                    mgr.report_unquiesced(2, "second stuck request")
        self.assertIn("cannot be proven quiesced", str(cm.exception))

    def test_escalation_reaches_the_scheduler(self):
        # The poll path contains backend faults so a bug cannot kill the engine.
        # The escalation is not a bug: swallowing it would release exactly the
        # pages the barrier exists to withhold.
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
        ) as all_reduce:
            with self.assertRaises(KVTransferBarrierEscalation):
                poll_and_all_reduce([poller], object())
        self.assertEqual(
            all_reduce.call_count,
            2,
            "a local escalation must be coordinated before it reaches the scheduler",
        )

    def test_escalation_is_not_swallowed_by_the_inflight_queue_sweep(self):
        # The whole strict guarantee hangs on this exception reaching the
        # scheduler's top-level handler, which tears the worker down. A
        # defensive try/except added to the sweep would silently convert
        # refuse-and-restart into leak-forever.
        from sglang.srt.disaggregation.prefill import (
            SchedulerDisaggregationPrefillMixin,
        )

        poller = SimpleNamespace(
            poll=Mock(return_value=KVPoll.Failed),
            advance_failure_quiescence=Mock(
                side_effect=KVTransferBarrierEscalation("cannot prove idle")
            ),
            is_failure_quiescing=Mock(return_value=True),
        )
        scheduler = SimpleNamespace(
            disagg_prefill_inflight_queue=[
                SimpleNamespace(rid="r", disagg_kv_sender=poller)
            ],
            attn_cp_cpu_group=object(),
            attn_tp_cpu_group=object(),
        )
        with patch(
            "sglang.srt.disaggregation.utils.dist.all_reduce",
            side_effect=lambda tensor, **kw: None,
        ):
            with self.assertRaises(KVTransferBarrierEscalation):
                SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                    scheduler
                )

    def test_a_peer_escalation_reaches_every_scheduler(self):
        poller = SimpleNamespace(
            poll=Mock(return_value=KVPoll.Failed),
            advance_failure_quiescence=Mock(return_value=False),
            is_failure_quiescing=Mock(return_value=True),
        )
        collective = Mock()

        def reduce(tensor, **_kwargs):
            collective()
            if collective.call_count == 2:
                # The local rank is still waiting, but a peer encoded escalation
                # into the coordinated quiescence state.
                tensor.fill_(0)

        with patch(
            "sglang.srt.disaggregation.utils.dist.all_reduce", side_effect=reduce
        ):
            with self.assertRaises(KVTransferBarrierEscalation):
                poll_and_all_reduce([poller], object())
        self.assertEqual(collective.call_count, 2)


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
                b"",  # device_kv_indices (HiSparse split-index transfers only)
                b"token",
            ]
        )

    def test_sweep_retires_destinations_and_status_together(self):
        # Otherwise a later request drawing the same bootstrap_room inherits the
        # old decode's addresses and a stale status.
        mgr = make_prefill_manager(room_sweep_ttl=0.0)
        self._room_with_metadata(mgr)
        self.assertIn(ROOM, mgr.transfer_infos)
        self.assertIn(ROOM, mgr.request_status)

        self.assertEqual(mgr._sweep_room_lifetimes(), 1)
        self.assertNotIn(ROOM, mgr._room_lifetimes)
        self.assertNotIn(ROOM, mgr.transfer_infos)
        self.assertNotIn(ROOM, mgr.request_status)
        self.assertNotIn(ROOM, mgr.req_to_decode_prefix_len)

    def test_emergency_reclaim_spares_young_rooms_then_retires_old_ones_fully(self):
        # Same hazard as the sweep, via the reclaim that runs when the tracked-
        # room cap is hit on insert.
        mgr = make_prefill_manager(room_sweep_ttl=300.0)
        self._room_with_metadata(mgr)

        with patch("sglang.srt.disaggregation.mooncake.conn.MAX_TRACKED_ROOMS", 1):
            # Young, metadata-first room: a sender may still arrive, so the cap
            # is soft and the room survives.
            mgr._room_lifetime(ROOM + 1, create=True)
            self.assertIn(ROOM, mgr.transfer_infos)
            self.assertEqual(len(mgr._room_lifetimes), 2, "the cap is soft")

            # Once aged out, the room is retired with everything keyed by it: a
            # request drawing it later must not inherit the old decode's
            # addresses or a stale status.
            mgr._room_lifetimes[ROOM].created_at = 0
            mgr._room_lifetime(ROOM + 2, create=True)
        self.assertNotIn(ROOM, mgr._room_lifetimes)
        self.assertNotIn(ROOM, mgr.transfer_infos)
        self.assertNotIn(ROOM, mgr.request_status)
        self.assertNotIn(ROOM, mgr.req_to_decode_prefix_len)

    def test_metadata_publication_holds_the_lifetime_generation_lock(self):
        # Otherwise the sweeper can retire the lifetime between validation and
        # publication, leaving orphan metadata for a later generation to inherit.
        mgr = make_prefill_manager()
        mgr.session_lock = threading.Lock()
        mgr.failed_sessions, mgr.session_failures = set(), {}
        mgr.decode_kv_args_table = {}
        mgr.resolve_kv_replica_factor = Mock()

        def assert_generation_locked():
            self.assertTrue(
                mgr._room_lifetimes_lock.locked(),
                "room metadata must publish under the lifetime generation lock",
            )

        class GenerationGuardedDict(dict):
            def setdefault(self, *args, **kwargs):
                assert_generation_locked()
                return super().setdefault(*args, **kwargs)

            def __setitem__(self, key, value):
                assert_generation_locked()
                return super().__setitem__(key, value)

        mgr.transfer_infos = GenerationGuardedDict()
        mgr.req_to_decode_prefix_len = GenerationGuardedDict()
        mgr.request_status = GenerationGuardedDict()

        self._room_with_metadata(mgr)
        self.assertIn(ROOM, mgr._room_lifetimes)
        self.assertIn(ROOM, mgr.transfer_infos)
        self.assertIn(ROOM, mgr.request_status)


class TestRoomCollisionIsolation(unittest.TestCase):
    """A receiver that lost a room collision must not damage the owner."""

    def test_constructor_rejects_loser_before_shared_state_or_metadata(self):
        mgr = make_decode_manager()
        mgr.get_session_id = Mock(side_effect=["owner", "loser"])
        mgr.addr_to_rooms_tracker = {"prefill:1": set()}
        mgr.required_prefill_response_num_table = {}
        mgr.prefill_response_tracker = {}

        owner = MooncakeKVReceiver(mgr, "prefill:1", ROOM)
        mgr.update_status.reset_mock()
        loser = MooncakeKVReceiver(mgr, "prefill:1", ROOM)

        self.assertIs(mgr._receivers[ROOM], owner)
        self.assertEqual(loser.poll(), KVPoll.Failed)
        self.assertFalse(loser._owns_room)
        mgr.update_status.assert_not_called()
        self.assertEqual(mgr.addr_to_rooms_tracker["prefill:1"], {ROOM})

        loser.init(0)
        loser.send_metadata(np.array([123], dtype=np.int32))
        loser.abort()
        mgr.update_status.assert_not_called()
        self.assertIs(mgr._receivers[ROOM], owner)

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

        # Unregistration is owner-checked in both directions.
        mgr.unregister_receiver(loser)
        self.assertIs(mgr._receivers[ROOM], owner)
        mgr.unregister_receiver(owner)
        self.assertTrue(mgr.register_receiver(loser))

    def test_a_lost_room_reports_instead_of_waiting_for_unroutable_proof(self):
        # ACKs are routed to the owning receiver, so the loser can never receive
        # proof; pretending to wait out the deadline would just hide that.
        mgr = make_decode_manager()
        receiver = make_receiver(mgr, bootstrap_infos=[{"rank": 0}])
        receiver._owns_room = False
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.TRANSFER_QUIESCE_TIMEOUTS"
        ) as metric:
            self.assertFalse(receiver.advance_failure_quiescence())
        metric.inc.assert_called_once_with()
        self.assertIn(ROOM, mgr._unquiesced_rooms, "reported without waiting")


class TestEndpointSendSerialization(unittest.TestCase):
    def test_connection_validation_and_send_share_the_endpoint_lock(self):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr._socket_lock = threading.Lock()
        mgr._socket_send_locks = {}
        endpoint = "tcp://127.0.0.1:1"
        observed = []

        class Socket:
            def send_multipart(self, parts, flags=0):
                observed.append(
                    (
                        "send",
                        mgr._socket_send_locks[endpoint].locked(),
                        parts,
                        flags,
                    )
                )

        def connect(*_args, **_kwargs):
            observed.append(("connect", mgr._socket_send_locks[endpoint].locked()))
            return Socket()

        mgr._connect = Mock(side_effect=connect)
        mgr._send_multipart_locked(endpoint, [b"status"], flags=zmq.NOBLOCK)

        self.assertEqual(
            observed,
            [
                ("connect", True),
                ("send", True, [b"status"], zmq.NOBLOCK),
            ],
        )


class TestWatermarkDelivery(unittest.TestCase):
    """A dropped watermark must be retried; it can gate a prefill's progress."""

    def test_a_backpressured_watermark_is_resent_and_never_blocks(self):
        state = {"blocked": True}
        sent, flags_used = [], []

        class Socket:
            def send_multipart(self, parts, flags=0):
                # Sent from the scheduler loop over sockets with no ZMQ send
                # timeout, so it must never wait for a peer.
                flags_used.append(bool(flags & zmq.NOBLOCK))
                if state["blocked"]:
                    raise zmq.Again()
                sent.append(parts)

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
                _connect_to_bootstrap_server=lambda info: (Socket(), threading.Lock()),
            ),
            "session",
        )

        handler.retry_pending_watermarks()
        self.assertEqual(flags_used, [], "nothing pending, nothing sent")

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
        self.assertTrue(all(flags_used), "every attempt must be non-blocking")


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
        from sglang.srt.managers import scheduler as scheduler_mod

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

        # Processing the same abort again must not enqueue the request twice.
        scheduler._pending_chunked_abort_req = req
        scheduler.chunked_req = req
        with patch.object(scheduler_mod, "prepare_abort"), patch.object(
            scheduler_mod, "release_kv_cache"
        ):
            scheduler_mod.Scheduler.process_pending_chunked_abort(scheduler)
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])

    def test_non_disagg_release_is_unchanged(self):
        req, scheduler, _prepare_abort, release_kv_cache = self._run(
            DisaggregationMode.DECODE
        )
        release_kv_cache.assert_called_once()
        self.assertEqual(release_kv_cache.call_args.kwargs, {"is_insert": False})
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()

    def test_pending_bootstrap_failure_preserves_user_abort(self):
        from sglang.srt.disaggregation import prefill as prefill_mod
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        user_abort = FINISH_ABORT("Aborted")
        req = SimpleNamespace(
            rid="chunked",
            bootstrap_room=ROOM,
            disagg_kv_sender=Mock(),
            time_stats=SimpleNamespace(trace_ctx=Mock()),
            req_pool_idx=None,
            kv=None,
            mamba_pool_idx=None,
            metadata_buffer_index=-1,
            pending_bootstrap=True,
            finished_reason=user_abort,
            return_logprob=False,
        )
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(tp_rank=0),
            tree_cache=Mock(),
            req_to_metadata_buffer_idx_allocator=Mock(),
            output_streamer=SimpleNamespace(stream_output=Mock()),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            enable_hicache_storage=False,
        )

        with patch.object(prefill_mod, "prepare_abort") as prepare_abort:
            prefill_mod.SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure(
                scheduler, req
            )

        prepare_abort.assert_not_called()
        self.assertIs(req.finished_reason, user_abort)
        self.assertFalse(req.pending_bootstrap)


class TestTransferInfoWire(unittest.TestCase):
    def test_abort_token_is_an_optional_trailing_frame(self):
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
        self.assertEqual(info.abort_token, b"", "a peer may omit the token frame")
        # The token rides behind the (also optional) device_kv_indices frame.
        self.assertEqual(
            TransferInfo.from_zmq(msg + [b"", b"nonce"]).abort_token,
            b"nonce",
        )


if __name__ == "__main__":
    unittest.main()
