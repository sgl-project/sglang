"""CPU control-flow evidence, not CUDA/RDMA validation.

HOST cancellation rides the native ABORT -> drain -> ABORT_ACK protocol: real
Scheduler.abort_request, prefill ABORT handling and decode deferred release, over
the native transfer worker and HOST pipeline of test_nixl_host_staging.Rig.
"""

import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import test_nixl_host_staging as h

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.decode_hicache_mixin import HiCacheRestoreResult
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def ring_handles(rig):
    return [x for x in rig.prefill.agent.handles if b"_hst_" in x.notif]


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
        self.assertTrue(rig.decode.is_abort_release_safe(h.ROOM, 1))
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
        self.assertTrue(rig.decode.is_abort_release_safe(h.ROOM, 1))
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
        rig.receiver.abort()  # Drained prefill acks before the re-arm below.
        self.assertTrue(rig.decode.is_abort_release_safe(h.ROOM, 1))
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
        self.assertTrue(rig.decode.is_abort_release_safe(h.ROOM, 1))

    def test_cancel_before_clear_still_acks_once_parts_drain(self):
        rig = self.rig()
        rig.decode.register_deferred_abort_room(h.ROOM)
        rig.prefill._handle_abort_notification([b"ABORT", b"37", b"decode", b"1"])
        rig.sender.clear()  # Scheduler drops the failed room while parts drain.
        self.assertFalse(rig.decode.is_abort_release_safe(h.ROOM, 1))
        self.land_everything(rig)
        self.assertTrue(rig.decode.is_abort_release_safe(h.ROOM, 1))

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
        with patch.object(
            rig.queue, "_poll_with_staging", return_value=[KVPoll.Failed]
        ):
            rig.queue.pop_transferred()
        finish.synchronize.assert_called_once()  # Restore DMA done before release.


if __name__ == "__main__":
    unittest.main()
