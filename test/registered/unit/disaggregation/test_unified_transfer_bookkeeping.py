"""Bookkeeping of unified-memory PD transfers: per-send and per-receive
completion, and the host waiter a page move uses for them on every rank."""

import threading
import time
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest import mock

import numpy as np

from sglang.srt.disaggregation import decode as decode_mod
from sglang.srt.disaggregation import prefill
from sglang.srt.disaggregation import utils as utils_mod
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.disaggregation.utils import (
    NetworkTransferWaiter,
    record_network_transfer,
)
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
from sglang.srt.mem_cache.kv.alloc import band as band_mod
from sglang.srt.mem_cache.kv.alloc import signals as signals_mod
from sglang.srt.mem_cache.kv.alloc.page_usage import Consumer
from sglang.srt.mem_cache.kv.alloc.signals import TransferRelease, signals_of
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq, decode_queue
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=16, suite="base-a-test-cpu")


class _Manager(CommonKVManager):
    """The status path of a connection, without its sockets."""

    def __init__(self):
        self.request_status = {}
        self.deferred_bootstrap = None
        self._deferred_ack_targets = {}
        self._deferred_ack_poisoned_rooms = set()
        self._staging_outstanding = {}


class TestSendCompletion(CustomTestCase):
    """What a page move waits for on a prefill node: one send's rows, until the
    rank's transfer worker is done with that send."""

    def _queued_send(self, manager, room):
        sender = SimpleNamespace(kv_mgr=manager, bootstrap_room=room)
        peer = prefill._send_peer(sender)
        manager.note_send_enqueued(room)
        return peer

    def test_a_send_ends_when_the_worker_finishes_it_not_the_room(self):
        """Each send ends when the worker finishes it, before the room
        concludes: a move never waits for chunks not yet scheduled."""
        manager = _Manager()
        manager.update_status(7, KVPoll.Bootstrapping)
        manager.update_status(7, KVPoll.Transferring)
        first = self._queued_send(manager, 7)
        second = self._queued_send(manager, 7)
        self.assertFalse(prefill._send_done(first))
        manager.note_send_finished(7)
        self.assertTrue(prefill._send_done(first))
        self.assertFalse(prefill._send_done(second))
        self.assertEqual(manager.request_status[7], KVPoll.Transferring)

    def test_a_send_never_queued_or_of_a_failed_room_waits_for_nothing(self):
        manager = _Manager()
        manager.update_status(8, KVPoll.Bootstrapping)
        skipped = prefill._send_peer(SimpleNamespace(kv_mgr=manager, bootstrap_room=8))
        self.assertTrue(prefill._send_done(skipped), "no worker reads its rows")
        queued = self._queued_send(manager, 8)
        self.assertFalse(prefill._send_done(queued))
        manager.update_status(8, KVPoll.Failed)
        self.assertTrue(prefill._send_done(queued))
        self.assertTrue(prefill._send_done(None))

    def test_the_send_waiter_spans_the_attention_tp_ranks_only(self):
        """REGRESSION: the send waiter reduces over attention TP only; CP ranks
        hold different rows and need not reach the same waits."""
        scheduler = SimpleNamespace(
            attn_tp_cpu_group="attn_tp",
            attn_cp_cpu_group="attn_cp",
            feed_watchdog=lambda: None,
        )
        waiter = prefill._send_waiter(scheduler)
        self.assertEqual(waiter._groups, ["attn_tp"])

    def test_a_send_finished_after_its_room_was_cleared_leaves_no_record(self):
        """REGRESSION: a finish reported after the room was cleared leaves no
        record behind."""
        manager = _Manager()
        manager.update_status(5, KVPoll.Bootstrapping)
        record = manager.send_record(5)
        manager.note_send_enqueued(5)
        manager._forget_sends(5)  # the sender's clear()
        manager.note_send_finished(5, record)
        self.assertFalse(manager._send_records)
        self.assertEqual(manager.send_counts(5), (0, 0))

    def test_a_late_report_does_not_finish_a_send_of_the_room_started_anew(self):
        """A send of the room's previous lifecycle that finishes late must not
        count as the new lifecycle's send: the engine still reads that one."""
        manager = _Manager()
        manager.update_status(6, KVPoll.Bootstrapping)
        old = manager.send_record(6)
        manager.note_send_enqueued(6)
        manager.request_status.pop(6)
        manager.update_status(6, KVPoll.Bootstrapping)
        manager.update_status(6, KVPoll.Transferring)
        queued = self._queued_send(manager, 6)
        manager.note_send_finished(6, old)
        self.assertFalse(prefill._send_done(queued))
        self.assertEqual(manager.send_counts(6), (1, 0))


class TestNetworkTransferWaiter(CustomTestCase):
    def test_the_waiter_releases_the_gil_while_the_engine_reports(self):
        """REGRESSION: the waiter releases the GIL between rounds, so an
        engine thread that needs it to report can finish."""
        done = threading.Event()
        reported = []

        def engine():
            time.sleep(0.02)
            reported.append(True)  # needs the GIL
            done.set()

        waiter = NetworkTransferWaiter(lambda peer: done.is_set(), (None,))
        thread = threading.Thread(target=engine)
        thread.start()
        waiter([SimpleNamespace(peer="send")])
        thread.join()
        self.assertEqual(reported, [True])

    def test_the_waiter_returns_once_every_transfer_is_done(self):
        rounds = defaultdict(int)

        def done(peer):
            rounds[peer] += 1
            return rounds[peer] >= (3 if peer == "slow" else 1)

        NetworkTransferWaiter(done, ())(
            [SimpleNamespace(peer="fast"), SimpleNamespace(peer="slow")]
        )
        self.assertEqual(rounds["slow"], 3)

    def test_each_rank_reduces_one_flag_however_it_lists_the_transfers(self):
        """Ranks may list the same transfers in another order or more than
        once; the collective must not depend on that: one done flag and one
        deadline flag per round."""
        reduced = []

        def reduce(flags, group, op):
            reduced.append(list(flags))
            return flags

        waiter = NetworkTransferWaiter(lambda peer: True, (object(),))
        a, b = SimpleNamespace(peer="a"), SimpleNamespace(peer="b")
        with mock.patch.object(utils_mod, "all_reduce_flags", reduce):
            for listed in ([a, b, a], [b, a]):
                waiter(listed)
        self.assertEqual(reduced, [[1, 1], [1, 1]])


def _agreeing_rank(rank, path, out):
    """One rank of a two-rank group: rank 1's engine finishes later. Each rank
    records how many of its own rounds said done before the waiter returned."""
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{path}", rank=rank, world_size=2
    )
    try:
        start = time.monotonic()
        delay = 0.0 if rank == 0 else 0.2
        rounds = []

        def done(peer):
            rounds.append(time.monotonic() - start >= delay)
            return rounds[-1]

        NetworkTransferWaiter(done, (dist.group.WORLD,))([SimpleNamespace(peer=0)])
        out.put((rank, len(rounds), sum(rounds), time.monotonic() - start))
    finally:
        dist.destroy_process_group()


class TestWaiterAgreesAcrossRanks(CustomTestCase):
    def test_every_rank_returns_in_the_same_round(self):
        """A rank whose engine is done keeps waiting until every rank's is:
        the ranks keep one layout, so they must leave a wait together."""
        import multiprocessing as mp
        import os
        import tempfile

        ctx = mp.get_context("spawn")
        out = ctx.Queue()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store")
            procs = [
                ctx.Process(target=_agreeing_rank, args=(rank, path, out))
                for rank in range(2)
            ]
            for proc in procs:
                proc.start()
            results = sorted(out.get(timeout=60) for _ in procs)
            for proc in procs:
                proc.join(timeout=60)
        (_, rounds0, done0, _), (_, rounds1, done1, _) = results
        self.assertEqual(rounds0, rounds1, "the ranks left after different rounds")
        self.assertEqual(done0, rounds0)
        self.assertGreater(rounds0, 1, "rank 0 left before rank 1 was done")
        self.assertEqual(done1, 1, "rank 1 left on its first done round")


class _Clock:
    """A clock that a sleep advances; it stands in for the waiter's `time`."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class TestWaiterEndsWithTheProtocol(CustomTestCase):
    """REGRESSION: the wait's own deadline lies past the protocol's, feeds the
    watchdog until then, and fails together on every rank once it passes."""

    def test_a_receive_finishing_after_180_s_within_a_300_s_timeout(self):
        clock = _Clock()
        start = clock.now
        beats = []
        done = lambda peer: clock.now - start >= 180.0  # noqa: E731
        waiter = NetworkTransferWaiter(
            done,
            (),
            heartbeat=lambda: beats.append(clock.now),
            interval=1.0,
        )
        with (
            envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.override(300),
            mock.patch.object(utils_mod, "time", clock),
            self.assertLogs(utils_mod.logger, level="WARNING") as logs,
        ):
            waiter([SimpleNamespace(peer="receiving")])
        self.assertGreaterEqual(clock.now - start, 180.0)
        # Periodic warnings while it waits.
        self.assertGreaterEqual(len(logs.output), 17)
        # The watchdog sees a heartbeat every round, never a silent stretch.
        self.assertEqual(beats[0], start)
        self.assertLessEqual(max(b - a for a, b in zip(beats, beats[1:])), 1.0)
        self.assertGreaterEqual(beats[-1] - start, 179.0)

    def test_the_default_deadline_lies_past_the_protocols(self):
        waiter = NetworkTransferWaiter(lambda peer: True, ())
        self.assertGreater(waiter._deadline, 300 + 300 + 30)

    def test_a_transfer_that_never_ends_fails_at_the_deadline(self):
        clock = _Clock()
        start = clock.now
        beats = []
        waiter = NetworkTransferWaiter(
            lambda peer: False,
            (),
            heartbeat=lambda: beats.append(clock.now),
            interval=1.0,
            deadline=50.0,
        )
        sender = SimpleNamespace(bootstrap_room=7)
        with (
            mock.patch.object(utils_mod, "time", clock),
            self.assertRaisesRegex(RuntimeError, r"deadline.*room 7"),
        ):
            waiter([SimpleNamespace(peer=(sender, 0))])
        self.assertGreaterEqual(clock.now - start, 50.0)
        # No heartbeat once the deadline passed: the watchdog sees the stall.
        self.assertLess(max(beats) - start, 50.0)

    def test_ranks_give_up_in_the_same_round(self):
        """The deadline is reduced with the done flag: a rank still within
        its own deadline raises in the round another rank's passed."""
        clock = _Clock()
        reduced = []

        def reduce(flags, group, op):
            reduced.append(list(flags))
            # The other rank is past its deadline.
            return [flags[0], 0]

        waiter = NetworkTransferWaiter(
            lambda peer: False, ("tp",), interval=1.0, deadline=1e9
        )
        with (
            mock.patch.object(utils_mod, "time", clock),
            mock.patch.object(utils_mod, "all_reduce_flags", reduce),
            self.assertRaises(RuntimeError),
        ):
            waiter([SimpleNamespace(peer="sending")])
        self.assertEqual(reduced, [[0, 1]])

    def test_a_failed_transfer_worker_ends_the_wait(self):
        """REGRESSION: a Mooncake transfer worker that raises fails its room,
        so a page move waiting for its unfinished send returns."""
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        manager = _Manager()
        manager.update_status(9, KVPoll.Bootstrapping)
        manager.update_status(9, KVPoll.WaitingForInput)
        sender = SimpleNamespace(kv_mgr=manager, bootstrap_room=9)
        peer = prefill._send_peer(sender)
        manager.note_send_enqueued(9)
        self.assertFalse(prefill._send_done(peer))

        class _OneChunk:
            def get(self):
                return SimpleNamespace(room=9)

        # A worker whose send raises (here: whatever it reads first is
        # missing, as a timed-out aux send raises out of the same body).
        worker = SimpleNamespace(
            enable_trace=False,
            poison_deferred_ack_room=lambda room: None,
            record_failure=lambda room, reason: None,
            update_status=manager.update_status,
            bootstrap_port=0,
        )
        with self.assertRaisesRegex(RuntimeError, "Transfer thread failed"):
            MooncakeKVManager.transfer_worker(worker, _OneChunk(), None)
        self.assertEqual(manager.request_status[9], KVPoll.Failed)
        self.assertTrue(prefill._send_done(peer))
        NetworkTransferWaiter(prefill._send_done, (), interval=0.0)(
            [SimpleNamespace(peer=peer)]
        )

    def test_the_scheduler_watchdog_counts_heartbeats(self):
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.scheduler_components import invariant_checker

        scheduler = SimpleNamespace(
            forward_ct=7,
            watchdog_heartbeats=0,
            is_initializing=False,
            cur_batch_for_debug=object(),
            dump_requests_before_crash=lambda: None,
        )
        with mock.patch.object(invariant_checker, "WatchdogRaw", SimpleNamespace):
            watchdog = invariant_checker.create_scheduler_watchdog(
                scheduler, watchdog_timeout=300.0
            )
        before = watchdog.get_counter()
        Scheduler.feed_watchdog(scheduler)
        self.assertNotEqual(watchdog.get_counter(), before)


class _HostCopy:
    """A host-copy completion event whose firing the test controls."""

    def __init__(self, fired):
        self.fired = fired

    def query(self):
        return self.fired


class TestRankSkewedWaitAfterCompaction(CustomTestCase):
    """REGRESSION: ranks whose host copies ended at different times, and so
    fenced a send's pages differently, still enter the same collective."""

    def _rank(self, backup_done):
        """One rank moves two pages a send reads (one also read by a backup)
        and reuses both; returns what it passed to the collective."""
        setup_allocator_context()
        bundle = build_unified_bundle(
            UnifiedGeometry(
                chain="swa",
                full_layer_ids=(0,),
                swa_layer_ids=(1,),
                lazy=True,
                device="cpu",
            )
        )
        allocator = bundle.token_to_kv_pool_allocator
        storage = allocator.storage
        full = allocator.full_attn_allocator
        reduced = []

        def reduce(flags, group, op):
            reduced.append(list(flags))
            return flags

        def wait(key, stream=None, *, network_waiter=None):
            # The network half as written; no stream on the CPU to wait an
            # unfinished backup on.
            transfers = tuple(
                s for s in signals_of(key) if isinstance(s, TransferRelease)
            )
            signals_mod.wait_signals(transfers, network_waiter=network_waiter)

        storage.set_network_waiter(
            NetworkTransferWaiter(lambda peer: True, (object(),))
        )
        ids = allocator.alloc(12)
        self.assertIsNotNone(ids)
        phys = full.ids.virtual_to_physical[ids].tolist()
        toward_frontier = sorted(
            range(len(ids)),
            key=lambda i: phys[i],
            reverse=full.grow_direction == "up",
        )
        nearest, interior = toward_frontier[:2], toward_frontier[5:7]
        full.free(ids[interior])
        sent = sorted(phys[i] for i in nearest)
        storage.record_network_transfer(
            FakeReq(rid="r", bootstrap_room=1),
            pages=np.array(sent),
            page_size=1,
            writes=False,
            peer="send",
        )
        full.record_transfer(_HostCopy(backup_done), ids[nearest[:1]], writes=False)
        with (
            mock.patch.object(utils_mod, "all_reduce_flags", reduce),
            mock.patch.object(band_mod, "wait_signals", wait),
        ):
            self.assertEqual(full.flush_for_allocation(), 2)
            self.assertEqual(reduced, [], "the move itself waits for no reader")
            pages = full.acquire(2, Consumer.FORWARD)
        self.assertEqual(sorted(pages.tolist()), sent, "the take reuses both pages")
        return reduced

    def test_ranks_whose_backup_ended_apart_enter_the_same_collective(self):
        pending, ended = self._rank(backup_done=False), self._rank(backup_done=True)
        self.assertEqual(pending, ended)
        self.assertEqual(ended, [[1, 1]])


class _Receiver(CommonKVReceiver):
    """A receiver whose transfer reports ``status``, notifying the abort
    through the real path into a socket that records what it sends."""

    def __init__(self, manager, room, n_peers):
        self.kv_mgr = manager
        self.bootstrap_room = room
        self.status = KVPoll.Transferring
        self.require_staging = False
        self.init_time = None  # as after a send that failed partway
        self.abort_notified = False
        self._abort_generation = None
        self.bootstrap_infos = [{"rank": r} for r in range(n_peers)]
        self.sent = []
        sock = SimpleNamespace(send_multipart=self.sent.append)
        self._connect_to_bootstrap_server = lambda info: (sock, threading.Lock())

    def poll(self):
        return self.status

    def failure_exception(self):
        raise NotImplementedError


class TestReceiveCompletion(CustomTestCase):
    """What a page move waits for on a decode node: a receive, until it
    succeeded, or failed and no prefill rank still writes it, by the rule
    deferred release keeps (acknowledgements, or the release timeout)."""

    ROOM = 3

    def _queue(self, *, deferred=True, timeout=30.0):
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.enable_staging = False
        queue.staging_handler = None
        queue.enable_host_receive = False
        queue.enable_deferred_kv_release = deferred
        queue.deferred_kv_release_timeout = timeout
        queue._deferred_releases = IndexedReqList(
            ListKind.PD_RELEASING, req_of=lambda x: x[0].req
        )
        return queue

    def _decode_req(self, *, n_peers=2, deferred=True):
        manager = CommonKVManager.__new__(CommonKVManager)
        manager._deferred_abort_ack_tracker = {}
        manager._deferred_abort_generation = 0
        manager.enable_deferred_decode_kv_release = deferred
        manager.local_ip, manager.rank_port = "10.0.0.1", 6000
        receiver = _Receiver(manager, self.ROOM, n_peers)
        return DecodeRequest(
            req=FakeReq(rid="r", bootstrap_room=self.ROOM), kv_receiver=receiver
        )

    def _ack(self, decode_req, prefill_rank):
        receiver = decode_req.kv_receiver
        receiver.kv_mgr.note_abort_ack(
            self.ROOM, prefill_rank, receiver._abort_generation
        )

    def test_a_receive_ends_at_success_not_while_transferring(self):
        queue = self._queue()
        decode_req = self._decode_req()
        self.assertFalse(queue._receive_done(decode_req))
        decode_req.kv_receiver.status = KVPoll.Success
        self.assertTrue(queue._receive_done(decode_req))
        self.assertTrue(queue._receive_done(SimpleNamespace(kv_receiver=None)))

    def test_a_published_failure_decode_has_not_aborted_waits_for_every_ack(self):
        """REGRESSION: a failed receive a page move sees before the queue
        notified the abort still waits for every peer's acknowledgement."""
        queue = self._queue()
        decode_req = self._decode_req(n_peers=2)
        decode_req.kv_receiver.status = KVPoll.Failed
        self.assertFalse(queue._receive_done(decode_req))
        receiver = decode_req.kv_receiver
        self.assertTrue(receiver.abort_notified)
        self.assertEqual(len(receiver.sent), 2, "every peer is told of the abort")
        self._ack(decode_req, 0)
        self.assertFalse(queue._receive_done(decode_req))
        self._ack(decode_req, 1)
        self.assertTrue(queue._receive_done(decode_req))
        self.assertEqual(len(receiver.sent), 2, "the abort is notified once")

    def test_an_unacknowledged_failure_ends_at_one_deadline_the_hold_keeps(self):
        """The release timeout runs from the first sight of the failure, a
        page move's or the queue's, and a later hold keeps that deadline and
        its own acknowledgement count."""
        queue = self._queue(timeout=10.0)
        decode_req = self._decode_req(n_peers=2)
        decode_req.kv_receiver.status = KVPoll.Failed
        queue.queue = decode_queue(ListKind.PD_RECEIVING, [decode_req])
        now = [100.0]
        with mock.patch.object(decode_mod.time, "monotonic", lambda: now[0]):
            self.assertFalse(queue._receive_done(decode_req))
            now[0] = 104.0
            queue._defer_release(decode_req)
            ((_, start, deadline, _, required_acks),) = queue._deferred_releases.items()
            self.assertEqual((start, deadline, required_acks), (100.0, 110.0, 2))
            now[0] = 109.9
            self.assertFalse(queue._receive_done(decode_req))
            now[0] = 110.0
            self.assertTrue(queue._receive_done(decode_req))

    def test_a_failure_never_published_is_done_at_once(self):
        """No peer has the addresses of a destination never sent: nothing to
        wait for, and nobody to notify."""
        queue = self._queue()
        for field, value in (
            ("awaiting_publish", True),
            ("pending_metadata", (None, (), {})),
        ):
            with self.subTest(field=field):
                decode_req = self._decode_req()
                setattr(decode_req, field, value)
                decode_req.kv_receiver.status = KVPoll.Failed
                self.assertTrue(queue._receive_done(decode_req))
                self.assertEqual(decode_req.kv_receiver.sent, [])

    def test_without_deferred_release_a_failure_is_done_at_once(self):
        """The queue releases such a failure at once, as a static pool does."""
        for queue_flag, manager_flag in ((False, True), (True, False)):
            with self.subTest(queue=queue_flag, manager=manager_flag):
                queue = self._queue(deferred=queue_flag)
                decode_req = self._decode_req(deferred=manager_flag)
                decode_req.kv_receiver.status = KVPoll.Failed
                self.assertTrue(queue._receive_done(decode_req))


class TestTransferBookkeeping(CustomTestCase):
    def test_a_one_slot_state_entry_is_recorded_as_rows(self):
        """REGRESSION: a recurrent state's one-slot (0-d) payload is recorded
        as rows rather than raising."""
        setup_allocator_context()
        bundle = build_unified_bundle(
            UnifiedGeometry(
                chain="tri",
                full_layer_ids=(0,),
                swa_layer_ids=(1,),
                state_layer_ids=(2,),
                lazy=True,
                device="cpu",
            )
        )
        allocator = bundle.token_to_kv_pool_allocator
        storage = allocator.storage
        entries = storage.wire_entries()
        state_name = allocator.roles.state
        self.assertIn(state_name, [entry.sub_pool for entry in entries])
        slot = np.array(3, dtype=np.int64)
        self.assertEqual(slot.ndim, 0)
        req = SimpleNamespace(rid="r1", bootstrap_room=2)
        record_network_transfer(
            storage,
            req,
            peer=None,
            writes=False,
            page_indices=np.array([], dtype=np.int32),
            page_size=allocator.page_size,
            state_types=[entry.state_type for entry in entries],
            state_indices=[
                [slot] if entry.sub_pool == state_name else None for entry in entries
            ],
        )
        (use,) = allocator.mamba_allocator.usage.transfers()
        self.assertEqual(np.asarray(use.ids).tolist(), [3])


if __name__ == "__main__":
    unittest.main()
