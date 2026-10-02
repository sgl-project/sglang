"""CPU control-flow evidence, not CUDA/RDMA validation.

HOST cancellation rides the native ABORT -> drain -> ABORT_ACK protocol: real
Scheduler.abort_request, prefill ABORT handling and decode deferred release, over
the native transfer worker and HOST pipeline of test_nixl_host_staging.Rig.
"""

import threading
import time
import unittest
from queue import Queue
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import test_nixl_host_staging as h

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.decode_hicache_mixin import HiCacheRestoreResult
from sglang.srt.disaggregation.prefill import maybe_release_metadata_buffer
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def ring_handles(rig):
    return [x for x in rig.prefill.agent.handles if b"_hst_" in x.notif]


def aux_handle(rig):
    return next(x for x in rig.prefill.agent.handles if b"_aux" in x.notif)


class HostAbortTest(unittest.TestCase):
    setUp = h.HostStagingTest.setUp

    def rig(self):
        """Decode room 37 with ring space assigned and a ring WRITE still posted."""
        rig = h.Rig(count=2)
        rig.prefill.agent.manual = True  # Posted WRITEs stay PROC until complete().
        rig.submit([2])
        for _ in range(500):
            rig.tick()
            if ring_handles(rig):
                break
            time.sleep(0.001)
        self.assertTrue(ring_handles(rig))
        self.assertTrue(rig.receiver.host_allocs)
        q = rig.queue
        q.enable_deferred_kv_release, q.deferred_kv_release_timeout = True, 30
        q._deferred_releases, q.tree_cache = [], object()
        q.tp_rank, q._process_hicache_local_restores = 0, Mock()
        q.scheduler.output_streamer = NS(stream_output=Mock())
        q._clean_hicache_prefetch_resources = Mock()
        self.released = Mock()
        for name, value in (
            ("release_kv_cache", self.released),
            (
                "prepare_abort",
                lambda req, msg, **kw: setattr(req, "finished_reason", msg),
            ),
        ):
            self.stack.enter_context(
                patch(f"sglang.srt.disaggregation.decode.{name}", value)
            )
        rig.req.req.rid, rig.req.req.return_logprob = "cancel-1", False
        return rig

    def land_everything(self, rig):
        for _ in range(500):
            for handle in rig.prefill.agent.handles:
                rig.prefill.agent.complete(handle)
            rig.prefill.host_staging.progress()
            rig.decode.update_transfer_status()
            rig.decode.check_transfer_done(h.ROOM)
            host = rig.prefill.host_staging
            if (
                not rig.prefill._staging_outstanding.get(h.ROOM)
                and not host.queue
                and not any(slot.part for slot in host.slots)
            ):
                return
            time.sleep(0.001)
        self.fail("prefill never drained")

    def wait_for_ack(self, rig):
        for _ in range(500):
            if rig.decode.is_abort_release_safe(h.ROOM, 1):
                return
            time.sleep(0.001)
        self.fail("drain ACK never arrived")

    def cancel(self, rig):
        s = NS(
            disaggregation_mode=DisaggregationMode.DECODE,
            chunked_req=None,
            mm_receiver=None,
            waiting_queue=[],
            dllm_config=None,
            _pending_chunked_abort_req=None,
            grammar_manager=NS(abort_requests=Mock(), grammar_queue=[]),
            disagg_decode_prealloc_queue=NS(queue=[], retracted_queue=[]),
            disagg_decode_transfer_queue=rig.queue,
            collect_inflight_reqs=lambda: [],
        )
        Scheduler.abort_request(s, NS(rid="cancel", abort_all=False))

    def test_failed_room_keeps_its_metadata_slot_until_its_aux_settles(self):
        rig = self.rig()
        self.assertTrue(rig.sender.is_aux_in_flight())  # Posted, still running.
        rig.sender.abort()
        self.assertEqual(rig.sender.poll(), KVPoll.Failed)
        req = NS(metadata_buffer_index=5, disagg_kv_sender=rig.sender)
        allocator = NS(free=Mock())
        # A reused slot would let the running aux WRITE read (and ship to the
        # decode) another request's metadata: fail-stop, never free.
        with self.assertRaises(h.FailStop):
            maybe_release_metadata_buffer(req, allocator)
        allocator.free.assert_not_called()
        self.land_everything(rig)
        self.assertFalse(rig.sender.is_aux_in_flight())
        with self.assertRaises(KVTransferError):
            rig.sender.failure_exception()
        maybe_release_metadata_buffer(req, allocator)
        allocator.free.assert_called_once_with(5)

    def test_inflight_queue_holds_a_failed_request_on_every_rank(self):
        rig = self.rig()
        rig.sender.abort()
        req = NS(
            rid="held",
            disagg_kv_sender=rig.sender,
            pending_bootstrap=False,
            time_stats=Mock(),
            finished_reason=None,
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            return_logprob=False,
            metadata_buffer_index=-1,
        )
        fail = Mock()
        s = self.scheduler([req], fail)
        # This rank's aux WRITE is still running: held, not concluded.
        self.assertEqual(Scheduler.process_disagg_prefill_inflight_queue(s), [])
        self.assertEqual(s.disagg_prefill_inflight_queue, [req])
        self.land_everything(rig)
        # Settled here, but another rank still has a WRITE running (MIN of
        # "released" is 0): held here too, so the ranks' queues stay aligned.
        with patch(
            "sglang.srt.disaggregation.prefill.all_reduce_min_attn_cp_tp_group",
            lambda values, *groups: [0] * len(values),
        ):
            Scheduler.process_disagg_prefill_inflight_queue(s)
        fail.assert_not_called()
        self.assertEqual(Scheduler.process_disagg_prefill_inflight_queue(s), [req])
        fail.assert_called_once_with(req)

    def scheduler(self, queue, fail):
        s = NS(
            disagg_prefill_inflight_queue=queue,
            attn_cp_cpu_group=None,
            attn_tp_cpu_group=None,
            scheduler_stage_metrics=None,
            handle_inflight_transfer_failure=fail,
            output_streamer=NS(stream_output=Mock()),
            req_to_metadata_buffer_idx_allocator=NS(free=Mock()),
        )
        s.hold_failed_prefill_transfers = lambda reqs, polls: (
            Scheduler.hold_failed_prefill_transfers(s, reqs, polls)
        )
        return s

    def test_fence_stops_a_room_failed_through_another_rank(self):
        # The reduced poll is Failed (another rank failed) while this rank's
        # room still transfers and its last chunk is not dequeued yet. The hold
        # check fences first, so the worker skips the chunk: no aux can start
        # reading a metadata slot the release below frees.
        rig = h.Rig(count=2)
        rig.prefill.agent.manual = True
        self.assertEqual(rig.prefill.check_status(h.ROOM), KVPoll.WaitingForInput)
        self.assertFalse(rig.sender.holds_failed_source())  # Nothing in flight...
        self.assertEqual(rig.prefill.check_status(h.ROOM), KVPoll.Failed)  # ...fenced.
        rig.submit([2])  # The last chunk reaches the worker after the fence.
        time.sleep(0.2)  # The worker dequeues it and must skip it.
        self.assertFalse([x for x in rig.prefill.agent.handles if b"_aux" in x.notif])
        self.assertFalse(rig.sender.is_aux_in_flight())
        # A chunk the worker counted before the fence is still seen in flight.
        rig.prefill._staging_outstanding[h.ROOM] = 1
        self.assertTrue(rig.sender.holds_failed_source())

    def test_pp_consensus_excludes_held_failed_requests(self):
        def req(rid, poll, holds):
            sender = NS(poll=lambda: poll, holds_failed_source=Mock(return_value=holds))
            return NS(rid=rid, disagg_kv_sender=sender)

        held, ok = req("held", KVPoll.Failed, True), req("ok", KVPoll.Failed, False)
        done, busy = (
            req("done", KVPoll.Success, False),
            req("busy", KVPoll.Transferring, False),
        )
        s = self.scheduler([held, ok, done, busy], Mock())
        s._pp_prefill_releasable_rids = lambda: (
            SchedulerPPMixin._pp_prefill_releasable_rids(s)
        )
        s.pp_group = NS(is_first_rank=True)
        self.assertEqual(
            SchedulerPPMixin._pp_pd_get_prefill_transferred_ids(s), ["ok", "done"]
        )
        held.disagg_kv_sender.holds_failed_source.return_value = False
        self.assertEqual(
            SchedulerPPMixin._pp_pd_get_prefill_transferred_ids(s),
            ["held", "ok", "done"],
        )

    def test_pp_release_follows_the_consensus_without_re_holding(self):
        # A rid in the stages' consensus was unheld on every stage when agreed;
        # re-holding it on one stage would keep it there after the rest released.
        rig = self.rig()
        rig.sender.abort()
        req = NS(
            rid="agreed",
            disagg_kv_sender=rig.sender,
            pending_bootstrap=False,
            time_stats=Mock(),
            finished_reason=None,
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            return_logprob=False,
            metadata_buffer_index=-1,
        )
        fail = Mock()
        s = self.scheduler([req], fail)
        s.hold_failed_prefill_transfers = Mock()
        self.assertEqual(
            Scheduler.process_disagg_prefill_inflight_queue(s, ["agreed"]), [req]
        )
        s.hold_failed_prefill_transfers.assert_not_called()
        fail.assert_called_once_with(req)

    def test_decode_release_timeout_must_outlast_writer_deadlines(self):
        limit = h.H.POST_DEADLINE_S + h.H.WRITE_DEADLINE_S
        h.H.check_release_timeout(limit)
        with self.assertRaisesRegex(ValueError, "writer deadlines"):
            h.H.check_release_timeout(limit - 1)

    def test_abandoned_writes_hold_the_ack_until_they_settle(self):
        rig = self.rig()
        agent, aux = rig.prefill.agent, aux_handle(rig)
        for x in ring_handles(rig):
            x.err = True  # The ring WRITE settles as ERR...
        agent.check_xfer_state = lambda x: (
            "ERR" if getattr(x, "err", False) else "DONE" if x.done else "PROC"
        )
        rig.decode.register_deferred_abort_room(h.ROOM)
        with patch(
            "sglang.srt.disaggregation.nixl.conn.NIXL_ERR_SETTLE_TIMEOUT_S", 0.01
        ):
            for _ in range(500):  # ...the aux keeps running: the worker gives up.
                rig.prefill.host_staging.progress()
                if h.ROOM in rig.prefill._abandoned:
                    break
                time.sleep(0.001)
        self.assertIn(h.ROOM, rig.prefill._abandoned)
        self.assertEqual(rig.prefill.check_status(h.ROOM), KVPoll.Failed)
        self.assertFalse(rig.prefill._staging_outstanding.get(h.ROOM))
        self.assertTrue(rig.sender.is_aux_in_flight())
        # Decode's ABORT is not acked while an abandoned WRITE may still land,
        # though the live count is zero: decode keeps the room's memory.
        rig.prefill._handle_abort_notification([b"ABORT", b"37", b"decode", b"1"])
        rig.prefill.host_staging.progress()
        self.assertFalse(rig.decode.is_abort_release_safe(h.ROOM, 1))
        # Nor once the scheduler cleared the room (an ABORT for an unknown room).
        rig.sender.clear()
        rig.prefill._handle_abort_notification([b"ABORT", b"37", b"decode", b"1"])
        self.assertFalse(rig.decode.is_abort_release_safe(h.ROOM, 1))
        for _ in range(500):
            for x in agent.handles:
                if not getattr(x, "err", False):
                    agent.complete(x)
            rig.prefill.host_staging.progress()  # Reaps once all settled.
            if h.ROOM not in rig.prefill._abandoned:
                break
            time.sleep(0.001)
        self.assertTrue(aux.done)
        self.assertNotIn(h.ROOM, rig.prefill._abandoned)
        self.assertFalse(rig.sender.is_aux_in_flight())
        self.wait_for_ack(rig)

    def test_blocked_ack_does_not_delay_progress_or_another_write_deadline(self):
        for blocked in ("send", "lock"):
            with self.subTest(blocked=blocked):
                m = h.NixlKVManager.__new__(h.NixlKVManager)
                host = m.host_staging = h.H.HostStaging.__new__(h.H.HostStaging)
                host.manager = m
                host.slots = [NS(part=object())]
                host._advance = Mock()
                m._staging_outstanding, m.transfer_infos = {}, {}
                m._deferred_ack_targets = {1: ("decode", 1)}
                m._deferred_ack_fanout_snapshots = {}
                m.attn_tp_rank = m.pp_rank = m.attn_cp_rank = 0
                m.pp_size = m.attn_cp_size = 1
                settled, live = object(), object()
                started = time.monotonic() - 20
                m._abandoned = {
                    1: [([settled], started, True)],
                    2: [([live], started, True)],
                }
                m._abandoned_lock = threading.Lock()
                m._aux_in_flight = {1, 2}
                m._xfer_state = lambda handle: "DONE" if handle is settled else "PROC"
                entered, unblock = threading.Event(), threading.Event()
                lock = threading.Lock()
                if blocked == "lock":
                    lock.acquire()
                sent = threading.Event()

                def send(parts):
                    if blocked == "send":
                        entered.set()
                        unblock.wait(5)
                    sent.set()

                def connect(endpoint, **kwargs):
                    if blocked == "lock":
                        entered.set()
                    return NS(send_multipart=send)

                m._connect = connect
                m._socket_send_locks = {"tcp://decode:1": lock}
                m._start_host_abort_ack_worker()
                errors = []

                def progress():
                    try:
                        host.progress()
                    except h.FailStop as exc:
                        errors.append(str(exc))

                try:
                    # A drains at t=20 and its ACK stalls. B is still live.
                    first = threading.Thread(target=progress, daemon=True)
                    first.start()
                    self.assertTrue(entered.wait(2))
                    first.join(1)
                    self.assertFalse(first.is_alive(), "ACK stalled HOST progress")
                    host._advance.assert_called_once()
                    self.assertFalse(sent.is_set())
                    self.assertIn(2, m._abandoned)
                    self.assertIn(2, m._aux_in_flight)
                    # Age B past its 25s deadline without changing the global
                    # clock used by other workers. A's ACK is still blocked.
                    m._abandoned[2] = [([live], time.monotonic() - 26, True)]
                    second = threading.Thread(target=progress, daemon=True)
                    second.start()
                    second.join(1)
                    self.assertFalse(second.is_alive())
                    self.assertTrue(errors and "room 2 past its deadline" in errors[0])
                    self.assertFalse(sent.is_set())
                finally:
                    unblock.set()
                    if blocked == "lock":
                        lock.release()
                    self.assertTrue(sent.wait(2))

    def test_full_ack_queue_drops_best_effort_ack_without_blocking(self):
        m = h.NixlKVManager.__new__(h.NixlKVManager)
        m._host_abort_acks = Queue(maxsize=1)  # Sender stalled; no consumer.
        m._send_abort_ack("decode", 1, 1)
        sender = threading.Thread(
            target=m._send_abort_ack, args=("decode", 1, 2), daemon=True
        )
        sender.start()
        try:
            sender.join(1)
            self.assertFalse(sender.is_alive(), "full ACK queue blocked the caller")
        finally:
            self.assertEqual(m._host_abort_acks.get_nowait(), ("decode", 1, 1))
            sender.join(1)

    def test_native_write_past_the_writer_deadline_fail_stops(self):
        rig = self.rig()
        self.land_everything(rig)  # The worker is idle: only our handles below.
        m = rig.prefill
        stuck = m.agent.initialize_xfer("WRITE", [], [], "decode", b"37_aux")
        staged = h.H.HostWrite(1)  # Bounded by host_staging, not as native.
        # Age this batch only, not the global deadlines used by other workers.
        started = time.monotonic() - h.H.POST_DEADLINE_S - h.H.WRITE_DEADLINE_S - 1
        raised = []

        def wait():  # Unbounded without the deadline: fail, don't hang.
            try:
                m._await_handles([stuck], failure_seen=False, started=started)
            except h.FailStop as e:
                raised.append(str(e))

        waiter = threading.Thread(target=wait, daemon=True)
        waiter.start()
        waiter.join(5)
        self.assertTrue(raised and "writer deadline" in raised[0])
        m._abandoned[h.ROOM] = [([staged], started, False)]
        m._reap_abandoned()  # A staged part past it: left to host_staging.
        self.assertIn(h.ROOM, m._abandoned)
        m._abandoned[h.ROOM] = [([stuck], started, True)]
        with self.assertRaisesRegex(h.FailStop, "past its deadline"):
            m._reap_abandoned()
        m._abandoned.clear()

    def test_cancel_holds_the_room_until_the_drain_ack(self):
        rig = self.rig()
        self.cancel(rig)
        self.assertTrue(rig.receiver.abort_notified)
        # Prefill failed the room and holds its ack behind the posted WRITE.
        self.assertEqual(rig.prefill.check_status(h.ROOM), KVPoll.Failed)
        self.assertIn(h.ROOM, rig.prefill._deferred_ack_targets)
        self.assertEqual(rig.queue.pop_transferred(), [])
        self.assertEqual(len(rig.queue._deferred_releases), 1)
        rig.queue.resolve_deferred_releases()
        self.released.assert_not_called()  # Held: the WRITE may still land.
        self.assertTrue(rig.handler.staging_allocator.allocations)
        self.land_everything(rig)
        self.wait_for_ack(rig)
        rig.queue.resolve_deferred_releases()
        self.released.assert_called_once()
        self.assertFalse(rig.handler.staging_allocator.allocations)
        rig.queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(0)
        self.assertFalse(rig.sender.is_source_pending())

    def test_ring_notif_after_ack_release_is_ignored(self):
        # The drain ack (ZMQ) can overtake the WRITE's notif (NIXL): the room is
        # released first, then its landed WRITE's notif arrives.
        rig = self.rig()
        self.cancel(rig)
        self.assertEqual(rig.queue.pop_transferred(), [])  # Held for the ack.
        for _ in range(500):
            for handle in rig.prefill.agent.handles:
                rig.prefill.agent.complete(handle)  # Lands; notif queued, unread.
            rig.prefill.host_staging.progress()
            if not rig.prefill._staging_outstanding.get(h.ROOM):
                break
            time.sleep(0.001)
        self.wait_for_ack(rig)
        rig.queue.resolve_deferred_releases()
        self.released.assert_called_once()
        rig.decode.update_transfer_status()  # Late notif: must not fail-stop.
        self.assertFalse(rig.handler.staging_allocator.allocations)

    def test_unacked_room_frees_its_ring_regions_on_release(self):
        rig = self.rig()
        # The prefill never answers (dead or partitioned): no ack, no fence.
        rig.receiver._connect_to_bootstrap_server = Mock(side_effect=OSError("gone"))
        self.cancel(rig)
        self.assertEqual(rig.queue.pop_transferred(), [])
        allocator = rig.handler.staging_allocator
        self.assertTrue(allocator.allocations)  # Held through the drain-ack wait.
        rig.queue._deferred_releases = [
            (entry[0], -1, *entry[2:]) for entry in rig.queue._deferred_releases
        ]
        rig.queue.resolve_deferred_releases()  # No fail-stop, no quarantine.
        self.released.assert_called_once()
        self.assertFalse(allocator.allocations)
        # The writer's deadlines fit inside the default wait, so reuse is safe.
        self.assertLessEqual(
            h.H.POST_DEADLINE_S + h.H.WRITE_DEADLINE_S,
            envs.SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT.default,
        )

    def test_ack_that_beats_the_scheduler_rearm_is_kept(self):
        rig = self.rig()
        self.land_everything(rig)
        rig.receiver.abort()
        self.wait_for_ack(rig)  # Drained prefill acks before the re-arm below.
        rig.decode.register_deferred_abort_room(h.ROOM)
        self.assertTrue(rig.decode.is_abort_release_safe(h.ROOM, 1))

    def test_cleared_prefill_room_still_acks_after_its_writes_drain(self):
        rig = self.rig()
        rig.decode.register_deferred_abort_room(h.ROOM)
        rig.sender.abort()  # Prefill-side cancel: the room fails natively.
        rig.sender.clear()
        self.assertNotIn(h.ROOM, rig.prefill.request_status)
        rig.prefill._handle_abort_notification([b"ABORT", b"37", b"decode", b"1"])
        self.assertFalse(rig.decode.is_abort_release_safe(h.ROOM, 1))
        self.land_everything(rig)
        self.wait_for_ack(rig)

    def test_cancel_before_clear_still_acks_once_parts_drain(self):
        rig = self.rig()
        rig.decode.register_deferred_abort_room(h.ROOM)
        rig.prefill._handle_abort_notification([b"ABORT", b"37", b"decode", b"1"])
        rig.sender.clear()  # Scheduler drops the failed room while parts drain.
        self.assertFalse(rig.decode.is_abort_release_safe(h.ROOM, 1))
        self.land_everything(rig)
        self.wait_for_ack(rig)

    def test_failed_restore_waits_for_its_dma_and_fences_the_prefill(self):
        rig = self.rig()
        finish = Mock()
        rig.queue.tree_cache = NS(
            cache_controller=NS(layer_done_counter=NS(events=[NS(finish_event=finish)]))
        )
        rig.req.hicache_restore_status = HiCacheRestoreResult.FAILED
        rig.req.hicache_load_consumer_index = 0
        self.assertEqual(rig.queue.pop_transferred(), [])
        self.assertTrue(rig.receiver.abort_notified)  # The prefill still writes.
        self.assertEqual(rig.prefill.check_status(h.ROOM), KVPoll.Failed)
        self.assertEqual(len(rig.queue._deferred_releases), 1)
        rig.req.hicache_restore_status = HiCacheRestoreResult.PENDING
        rig.queue.queue = [rig.req]
        rig.queue._deferred_releases = []
        with (
            patch.object(rig.queue, "_poll_with_staging", return_value=[KVPoll.Failed]),
            patch.object(envs.SGLANG_NIXL_HOST_STAGING_MB, "get", return_value=1),
        ):
            rig.queue.pop_transferred()
        finish.synchronize.assert_called_once()  # Restore DMA done before release.


if __name__ == "__main__":
    unittest.main()
