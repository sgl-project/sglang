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
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation import prefill as prefill_module  # noqa: E402
from sglang.srt.disaggregation import utils  # noqa: E402
from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.common.conn import CommonKVManager  # noqa: E402
from sglang.srt.disaggregation.pp_consensus_store import (  # noqa: E402
    PPConsensusStore,
)
from sglang.srt.disaggregation.prefill import (  # noqa: E402
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import (  # noqa: E402
    ABORT_REASON_DECODE,
    ABORT_REASON_REQUEST,
    ReqToMetadataIdxAllocator,
    TransferOutcome,
    poll_transfer_outcomes_attn_cp_tp_group,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.runtime_context import get_parallel  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

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

    def poll_pp_consensus(self):
        return CommonKVManager.check_status_pp_consensus(
            self.kv_mgr, self.bootstrap_room
        )

    def is_failure_requested_abort(self):
        return utils.is_abort_failure_reason(
            self.kv_mgr.failure_records.get(self.bootstrap_room)
        )

    def is_failure_requested_abort_pp_consensus(self):
        return CommonKVManager.is_failure_requested_abort_pp_consensus(
            self.kv_mgr, self.bootstrap_room
        )

    def clear(self):
        pass

    def failure_exception(self):
        raise RuntimeError(self.kv_mgr.failure_records.get(self.bootstrap_room))

    def get_transfer_metric(self):
        return None


class _Stage(SchedulerDisaggregationPrefillMixin):
    """One prefill PP stage running the real inflight-queue release path."""

    _release_aborted_request = Scheduler._release_aborted_request

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
        self.pp_poll_sync_work_list_inflight = []
        self.attn_cp_cpu_group = self.attn_tp_cpu_group = None
        self.disagg_prefill_inflight_queue = []
        self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(4)
        self.tree_cache = Mock()
        self.metrics_reporter = SimpleNamespace(enable_metrics=True)
        self.metrics_collector = Mock()
        self.scheduler_stage_metrics = None
        self.streamed = []
        self.output_streamer = SimpleNamespace(
            stream_output=lambda reqs, *args: self.streamed.extend(
                (req.rid, req.finished_reason) for req in reqs
            )
        )

    def release(self):
        # Both stages share one process; scope the parallel place per stage.
        with get_parallel().override(pp_rank=self.pp_rank, pp_size=2, tp_rank=0):
            return self.process_disagg_prefill_inflight_queue()

    def inflight(self, rid, room, status=KVPoll.Transferring):
        self.store[room] = status
        req = SimpleNamespace(
            rid=rid,
            bootstrap_room=room,
            bootstrap_host=prefill_module.FAKE_BOOTSTRAP_HOST,
            metadata_buffer_index=self.req_to_metadata_buffer_idx_allocator.alloc(),
            pending_bootstrap=False,
            finished_reason=None,
            return_logprob=False,
            return_routed_experts=False,
            session=None,
            cache_request_handle=None,
            disagg_kv_sender=_Sender(self.kv_mgr, room),
            time_stats=SimpleNamespace(
                set_prefill_kv_transfer_finish_time=lambda: None,
                set_completion_time=lambda: None,
                trace_ctx=SimpleNamespace(abort=lambda **kw: None),
            ),
            release_mm_inputs_on_abort=lambda: None,
        )
        req.finish_outside_step = lambda reason: setattr(req, "finished_reason", reason)
        self.disagg_prefill_inflight_queue.append(req)
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
            patch.object(prefill_module, "release_kv_cache", lambda *a, **k: None),
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
        return self.pp0.inflight(rid, room), self.pp1.inflight(rid, room)

    def test_release_waits_for_every_stage_then_frees_together(self):
        self._both("x", 8)
        self.pp0.store[8] = KVPoll.Success
        self._replicated(8, [KVPoll.Success, KVPoll.Transferring])

        # PP1's transfer is still running: neither stage releases.
        self._iterate()
        for stage in (self.pp0, self.pp1):
            self.assertEqual(len(stage.disagg_prefill_inflight_queue), 1)
            self.assertEqual(stage.streamed, [])

        self.pp1.store[8] = KVPoll.Success
        self._replicated(8, [KVPoll.Success, KVPoll.Success])
        self._iterate()
        for stage in (self.pp0, self.pp1):
            self.assertEqual(stage.disagg_prefill_inflight_queue, [])
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

        self.assertEqual(self.pp1.disagg_prefill_inflight_queue, [req1])
        self.assertEqual(self.pp1.streamed, [])
        # The local poll that fires conclusion side effects only runs on release.
        self.assertEqual(req1.disagg_kv_sender.local_polls, 0)

    def test_failure_on_one_stage_carries_its_classification_everywhere(self):
        for room, reason, expected in (
            (20, ABORT_REASON_DECODE, "abort"),
            (21, ABORT_REASON_REQUEST, "abort"),
            (22, "Failed to send kv chunk", "error"),
        ):
            with self.subTest(reason=reason):
                self.pp0.streamed.clear()
                self.pp1.streamed.clear()
                self._both("x", room)
                self.pp0.store[room] = KVPoll.Success
                self.pp1.fail(room, reason)
                self._replicated(room, [KVPoll.Success, KVPoll.Failed])

                self._iterate()

                # PP0 succeeded locally and holds no reason, yet concludes
                # the request the way PP1 saw it.
                self.assertEqual(self.pp0.streamed, [("x", expected)])
                self.assertEqual(self.pp1.streamed, [("x", expected)])
                self.assertEqual(self.pp0.disagg_prefill_inflight_queue, [])
                self.assertEqual(self.pp1.disagg_prefill_inflight_queue, [])

    def test_real_error_on_any_stage_outranks_an_abort_on_another(self):
        self._both("x", 13)
        self.pp0.fail(13, ABORT_REASON_REQUEST)
        self.pp1.fail(13, "Failed to send kv chunk")
        self._replicated(13, [KVPoll.Failed, KVPoll.Failed])

        self._iterate()

        self.assertEqual(self.pp0.streamed, [("x", "error")])
        self.assertEqual(self.pp1.streamed, [("x", "error")])
        self.pp0.metrics_collector.increment_transfer_failed_reqs.assert_called_once()

    def test_requested_abort_is_not_counted_as_transfer_failure(self):
        self._both("x", 14)
        self.pp1.fail(14, ABORT_REASON_DECODE)
        self._replicated(14, [KVPoll.Transferring, KVPoll.Failed])

        self._iterate()

        for stage in (self.pp0, self.pp1):
            self.assertEqual(stage.streamed, [("x", "abort")])
            stage.metrics_collector.increment_transfer_failed_reqs.assert_not_called()

    def test_idle_iteration_sends_nothing(self):
        self.assertEqual(self._iterate(), ([], []))
        self.assertEqual(self.pp_group.sends, 0)


class _LocalSender:
    def __init__(self, poll, abort=False):
        self._poll = poll
        self._abort = abort

    def poll(self):
        return self._poll

    def is_failure_requested_abort(self):
        return self._abort


class TestTransferOutcomeClassification(CustomTestCase):
    def test_outcomes_without_pp(self):
        senders = [
            _LocalSender(KVPoll.Success),
            _LocalSender(KVPoll.Transferring),
            _LocalSender(KVPoll.Failed, abort=True),
            _LocalSender(KVPoll.Failed, abort=False),
        ]
        with patch.object(utils, "_all_reduce_polls", lambda polls, group: polls):
            polls, outcomes = poll_transfer_outcomes_attn_cp_tp_group(
                senders, None, None
            )
        self.assertEqual(
            polls,
            [KVPoll.Success, KVPoll.Transferring, KVPoll.Failed, KVPoll.Failed],
        )
        self.assertEqual(
            outcomes,
            [
                TransferOutcome.SUCCESS,
                None,
                TransferOutcome.ABORT,
                TransferOutcome.ERROR,
            ],
        )

    def test_tp_peer_error_outranks_local_abort(self):
        # A TP peer failed with a real error: MIN over the group clears this
        # rank's abort-only flag.
        def peer_reduce(flags, group):
            half = len(flags) // 2
            return flags[:half] + [0] * half

        with patch.object(utils, "_all_reduce_polls", peer_reduce):
            _, outcomes = poll_transfer_outcomes_attn_cp_tp_group(
                [_LocalSender(KVPoll.Failed, abort=True)], None, None
            )
        self.assertEqual(outcomes, [TransferOutcome.ERROR])


if __name__ == "__main__":
    unittest.main()
