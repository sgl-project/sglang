"""Admission gates must not strand transfers or cancellation cleanup."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQueueMaintenance(CustomTestCase):
    def test_retraction_and_polling_gate_admission_only(self):
        for threshold in (1, 0.5):
            for retracted in (False, True):
                for count in (0, 1):
                    with self.subTest(
                        threshold=threshold, retracted=retracted, count=count
                    ):
                        config = SimpleNamespace(
                            disaggregation_decode_enable_offload_kvcache=False,
                            disaggregation_decode_host_receive_threshold=threshold,
                        )
                        scheduler = SimpleNamespace(
                            scheduler_stage_metrics=None,
                            enable_decode_hicache=False,
                            enable_hisparse=False,
                            waiting_queue=[],
                            polling_count=count,
                            polling_interval=2,
                            disagg_decode_prealloc_queue=MagicMock(),
                            disagg_decode_transfer_queue=MagicMock(),
                        )
                        prealloc = scheduler.disagg_decode_prealloc_queue
                        transfer = scheduler.disagg_decode_transfer_queue
                        prealloc.resume_retracted_reqs.return_value = []
                        prealloc.retracted_queue = [object()] if retracted else []
                        order = []
                        completed = object()

                        def preallocate(**kwargs):
                            order.append("preallocate")
                            return [], []

                        def transferred():
                            order.append("transfer")
                            return [completed]

                        prealloc.pop_preallocated.side_effect = preallocate
                        transfer.pop_transferred.side_effect = transferred
                        with patch(
                            "sglang.srt.disaggregation.decode.get_disagg",
                            return_value=config,
                        ):
                            SchedulerDisaggregationDecodeMixin.process_decode_queue(
                                scheduler
                            )
                        prealloc.pop_preallocated.assert_called_once_with(
                            allow_preallocation=not retracted and count == 1
                        )
                        transfer.resolve_deferred_releases.assert_called_once()
                        transfer.pop_transferred.assert_called_once()
                        self.assertEqual(scheduler.waiting_queue, [completed])
                        self.assertEqual(
                            order,
                            ["preallocate", "transfer"]
                            if threshold == 1
                            else ["transfer", "preallocate"],
                        )

    def test_maintenance_polls_failure_and_cleans_cancelled_pending_request(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.tp_rank = 0
        queue.gloo_group = object()
        queue._resolve_pending_reqs = MagicMock()
        queue.scheduler = SimpleNamespace(
            enable_priority_scheduling=False,
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            output_streamer=MagicMock(),
        )

        def entry(rid, finished=None):
            return SimpleNamespace(
                req=SimpleNamespace(
                    rid=rid,
                    bootstrap_room=42,
                    finished_reason=finished,
                    return_logprob=False,
                    time_stats=MagicMock(),
                ),
                kv_receiver=MagicMock(),
                waiting_for_input=False,
            )

        live, failed, cancelled = (
            entry("live"),
            entry("failed"),
            entry("cancelled", FINISH_ABORT()),
        )
        receivers = [item.kv_receiver for item in (live, failed, cancelled)]
        queue.queue = [live, failed, cancelled]
        queue.pending_reqs = [failed, cancelled]

        def abort(req, *args, **kwargs):
            req.finished_reason = FINISH_ABORT()

        with (
            patch(
                "sglang.srt.disaggregation.decode.poll_and_all_reduce",
                return_value=[
                    KVPoll.WaitingForInput,
                    KVPoll.Failed,
                    KVPoll.Bootstrapping,
                ],
            ) as poll,
            patch("sglang.srt.disaggregation.decode.prepare_abort", side_effect=abort),
        ):
            ready, dead = queue.pop_preallocated(allow_preallocation=False)
        poll.assert_called_once()
        self.assertEqual(ready, [])
        self.assertEqual(dead, [failed, cancelled])
        self.assertEqual(queue.queue, [live])
        self.assertEqual(queue.pending_reqs, [])
        self.assertTrue(live.waiting_for_input)
        receivers[0].clear.assert_not_called()
        for receiver in receivers[1:]:
            receiver.clear.assert_called_once()
        # No allocator, running batch or KV budget exists in this fixture:
        # a maintenance-only pass must not inspect admission resources.


if __name__ == "__main__":
    unittest.main()
