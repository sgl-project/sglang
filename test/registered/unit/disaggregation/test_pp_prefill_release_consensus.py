"""PP prefill stages must release each transfer in the same iteration and the same way.

Each PP stage sends the KV of its own layers, so its sender can conclude a
transfer at a different time, or with a different result, than the other
stages. PP0 decides from every stage's status in PPConsensusStore and forwards
the decision; every stage applies it instead of its own local poll.
"""

import threading
import time
import unittest
from collections import deque
from functools import partial
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation import prefill as prefill_module  # noqa: E402
from sglang.srt.disaggregation import utils  # noqa: E402
from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.common.conn import (  # noqa: E402
    CommonKVManager,
    _pp_held_key,
    _PPStagingOutstanding,
)
from sglang.srt.disaggregation.pp_consensus_store import (  # noqa: E402
    PPConsensusStore,
)
from sglang.srt.disaggregation.prefill import (  # noqa: E402
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import ReqToMetadataIdxAllocator  # noqa: E402
from sglang.srt.distributed.pp_sync_window import PPSyncSendWindow  # noqa: E402
from sglang.srt.managers.scheduler_components.request_list import (  # noqa: E402
    IndexedReqList,
    ListKind,
    move,
)
from sglang.srt.runtime_context import get_parallel  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.scheduler_test_utils import (  # noqa: E402
    FakeReq,
    leaving_retirer,
)

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _PPGroup:
    """In-process stand-in for the PP CPU group shared by both stages."""

    def __init__(self):
        self._broadcast = None
        self._inflight = deque()
        self.sends = 0

    def broadcast_object(self, obj, src=0):
        if obj is not None:
            self._broadcast = obj
        return self._broadcast

    def send_object(self, obj, dst, async_send=False, tag=0):
        self.sends += 1
        self._inflight.append(obj)
        return []

    def recv_object(self, src, tag=0):
        return self._inflight.popleft()


class _Sender:
    """Reads the stage's real store the way CommonKVSender does."""

    def __init__(self, kv_mgr, room):
        self.kv_mgr = kv_mgr
        self.bootstrap_room = room
        self.local_polls = 0

    def poll(self):
        self.local_polls += 1
        return self.kv_mgr.request_status[self.bootstrap_room]

    def poll_pp_release(self):
        return CommonKVManager.check_status_pp_release(self.kv_mgr, self.bootstrap_room)

    def clear(self):
        pass

    def failure_exception(self):
        raise RuntimeError(self.kv_mgr.failure_records.get(self.bootstrap_room))

    def get_transfer_metric(self):
        return None


class _Stage(SchedulerDisaggregationPrefillMixin):
    """One prefill PP stage running the real transfer-queue release path."""

    def __init__(self, pp_rank, pp_group):
        self.pp_rank = pp_rank
        self.store = PPConsensusStore(2, pp_rank, pp_group)
        self.kv_mgr = SimpleNamespace(
            request_status=self.store,
            failure_records={},
            failure_lock=threading.Lock(),
        )
        self.pp_group = SimpleNamespace(
            recv_object=pp_group.recv_object, send_object=pp_group.send_object
        )
        self.pp_transfer_poll_send_window = PPSyncSendWindow(rounds=1)
        self.attn_cp_cpu_group = self.attn_tp_cpu_group = None
        self.disagg_prefill_transfer_queue = IndexedReqList(ListKind.PD_SENDING)
        self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(4)
        self.request_retirer = leaving_retirer()
        self.metrics_reporter = SimpleNamespace(enable_metrics=True)
        self.metrics_collector = Mock()
        self.scheduler_stage_metrics = None
        self.streamed = []
        self.output_streamer = SimpleNamespace(
            stream_output=lambda reqs, *args: self.streamed.extend(
                (req.rid, req.finished_reason) for req in reqs
            )
        )
        self._release_mm_inputs = lambda req: None

    def release(self):
        # Both stages share one process; scope the parallel place per stage.
        with get_parallel().override(pp_rank=self.pp_rank, pp_size=2, tp_rank=0):
            return self.process_disagg_prefill_transfer_queue()

    def transfer(self, rid, room, status=KVPoll.Transferring):
        self.store[room] = status
        req = FakeReq(
            rid=rid,
            bootstrap_room=room,
            bootstrap_host=prefill_module.FAKE_BOOTSTRAP_HOST,
            metadata_buffer_index=self.req_to_metadata_buffer_idx_allocator.alloc(),
            pending_bootstrap=False,
            finished_reason=None,
            return_logprob=False,
            return_routed_experts=False,
            disagg_kv_sender=_Sender(self.kv_mgr, room),
            time_stats=SimpleNamespace(
                set_prefill_kv_transfer_finish_time=lambda: None,
                set_completion_time=lambda: None,
                trace_ctx=SimpleNamespace(abort=lambda **kw: None),
            ),
        )
        req.finish_outside_step = lambda reason: setattr(req, "finished_reason", reason)
        move([req], self.disagg_prefill_transfer_queue)
        return req

    def fail(self, room, reason):
        """Fail a room the way a backend does: record the reason, then mark Failed."""
        CommonKVManager.record_failure(self.kv_mgr, room, reason)
        self.store[room] = KVPoll.Failed


def _classify(req, message, status_code=None):
    # A real transfer error carries a 500; a requested abort carries none.
    req.finished_reason = (
        "error" if status_code == HTTPStatus.INTERNAL_SERVER_ERROR else "abort"
    )


class TestPPPrefillReleaseConsensus(CustomTestCase):
    def setUp(self):
        self._patches = [
            patch.object(utils, "_all_reduce_polls", lambda polls, group: polls),
            patch.object(prefill_module, "prepare_abort", _classify),
        ]
        for p in self._patches:
            p.start()
        self.pp_group = _PPGroup()
        self.pp0 = _Stage(0, self.pp_group)
        self.pp1 = _Stage(1, self.pp_group)

    def tearDown(self):
        self.pp1.store.close()
        self.pp0.store.close()
        for p in self._patches:
            p.stop()

    def _replicated(self, key, expected):
        deadline = time.monotonic() + 2
        while self.pp0.store.collect(key) != expected:
            self.assertLess(time.monotonic(), deadline, f"{key} not replicated")
            time.sleep(0.01)

    def _iterate(self):
        """One scheduler iteration: PP0 decides and forwards, PP1 applies."""
        return self.pp0.release(), self.pp1.release()

    def _both(self, rid, room):
        return self.pp0.transfer(rid, room), self.pp1.transfer(rid, room)

    def test_release_waits_for_every_stage_then_frees_together(self):
        self._both("x", 8)
        self.pp0.store[8] = KVPoll.Success
        self._replicated(8, [KVPoll.Success, KVPoll.Transferring])

        # PP1's transfer is still running: neither stage releases.
        self._iterate()
        for stage in (self.pp0, self.pp1):
            self.assertEqual(len(stage.disagg_prefill_transfer_queue), 1)
            self.assertEqual(stage.streamed, [])

        self.pp1.store[8] = KVPoll.Success
        self._replicated(8, [KVPoll.Success, KVPoll.Success])
        self._iterate()
        for stage in (self.pp0, self.pp1):
            self.assertEqual(list(stage.disagg_prefill_transfer_queue), [])
            self.assertEqual([rid for rid, _ in stage.streamed], ["x"])
            self.assertEqual(
                stage.req_to_metadata_buffer_idx_allocator.available_size(), 4
            )

    def test_late_local_conclusion_waits_for_pp0(self):
        _, req1 = self._both("x", 10)
        self._replicated(10, [KVPoll.Transferring] * 2)

        self.pp0.release()
        # PP1 concludes locally after PP0 already decided "not yet".
        self.pp1.store[10] = KVPoll.Success
        self.pp1.release()

        self.assertEqual(list(self.pp1.disagg_prefill_transfer_queue), [req1])
        self.assertEqual(self.pp1.streamed, [])
        # The local poll that fires conclusion side effects only runs on release.
        self.assertEqual(req1.disagg_kv_sender.local_polls, 0)

        self.pp0.store[10] = KVPoll.Success
        self._replicated(10, [KVPoll.Success] * 2)
        self._iterate()
        self.assertEqual(self.pp1.streamed, [("x", req1.finished_reason)])
        self.assertEqual(req1.disagg_kv_sender.local_polls, 1)

    def test_failure_on_any_stage_fails_the_transfer_on_every_stage(self):
        self._both("x", 20)
        self.pp0.store[20] = KVPoll.Success
        self.pp1.fail(20, "Failed to send kv chunk")
        self._replicated(20, [KVPoll.Success, KVPoll.Failed])

        self._iterate()

        # PP0 succeeded locally, yet fails the request with PP1.
        for stage in (self.pp0, self.pp1):
            self.assertEqual(stage.streamed, [("x", "error")])
            self.assertEqual(list(stage.disagg_prefill_transfer_queue), [])
            stage.metrics_collector.increment_transfer_failed_reqs.assert_called_once()

    def test_failure_waits_for_every_stage_to_conclude(self):
        # One stage failed the room while the other stage's transfer worker
        # still sends from (and into) its pages: no stage may free the KV yet.
        for room, failed, sending in (
            (30, self.pp0, self.pp1),
            (31, self.pp1, self.pp0),
        ):
            with self.subTest(failed_stage=failed.pp_rank):
                self.pp0.streamed.clear()
                self.pp1.streamed.clear()
                self._both("x", room)
                failed.fail(room, "Failed to send kv chunk")
                expected = [KVPoll.Transferring] * 2
                expected[failed.pp_rank] = KVPoll.Failed
                self._replicated(room, expected)

                self._iterate()
                for stage in (self.pp0, self.pp1):
                    self.assertEqual(len(stage.disagg_prefill_transfer_queue), 1)
                    self.assertEqual(stage.streamed, [])

                # The sending stage concludes: both stages release together.
                sending.store[room] = KVPoll.Success
                expected[sending.pp_rank] = KVPoll.Success
                self._replicated(room, expected)
                self._iterate()
                for stage in (self.pp0, self.pp1):
                    self.assertEqual(list(stage.disagg_prefill_transfer_queue), [])
                    self.assertEqual(stage.streamed, [("x", "error")])

    def test_success_waits_for_staging_chunks_on_every_stage(self):
        # A stage holds Success while its staging chunks are outstanding; PP0
        # must see PP1's hold through the store.
        for stage in (self.pp0, self.pp1):
            mgr = stage.kv_mgr
            mgr.defer_decode_allocation = False
            mgr.staging_holds_success = partial(
                CommonKVManager.staging_holds_success, mgr
            )
            mgr._staging_outstanding = _PPStagingOutstanding(mgr)
        self._both("x", 40)
        self.pp1.kv_mgr._staging_outstanding[40] += 1
        self.pp0.store[40] = KVPoll.Success
        self.pp1.store[40] = KVPoll.Success
        self._replicated(40, [KVPoll.Success, KVPoll.Success])

        self._iterate()
        for stage in (self.pp0, self.pp1):
            self.assertEqual(len(stage.disagg_prefill_transfer_queue), 1)

        self.pp1.kv_mgr._staging_outstanding[40] -= 1
        self._replicated(_pp_held_key(40), [None, 0])
        self._iterate()
        for stage in (self.pp0, self.pp1):
            self.assertEqual(list(stage.disagg_prefill_transfer_queue), [])
            self.assertEqual([rid for rid, _ in stage.streamed], ["x"])

    def test_every_staging_counter_mutator_syncs_the_held_flag(self):
        mgr = self.pp1.kv_mgr
        mgr.defer_decode_allocation = False
        mgr.staging_holds_success = partial(CommonKVManager.staging_holds_success, mgr)
        counts = mgr._staging_outstanding = _PPStagingOutstanding(mgr)
        self.pp1.store[41] = KVPoll.Transferring
        for hold, release in (
            (lambda: counts.__setitem__(41, 1), lambda: counts.pop(41)),
            (lambda: counts.setdefault(41, 1), lambda: counts.__delitem__(41)),
            (lambda: counts.update({41: 2}), lambda: counts.clear()),
            (lambda: counts.__ior__({41: 1}), lambda: counts.popitem()),
        ):
            hold()
            self.assertEqual(self.pp1.store[_pp_held_key(41)], 1)
            release()
            self.assertEqual(self.pp1.store[_pp_held_key(41)], 0)

    def test_idle_iteration_sends_nothing(self):
        self.assertEqual(self._iterate(), ([], []))
        self.assertEqual(self.pp_group.sends, 0)


if __name__ == "__main__":
    unittest.main()
