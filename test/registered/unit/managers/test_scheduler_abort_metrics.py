"""Unit tests for request-level abort metric accounting."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    _build_aborted_request_labels,
)
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _make_req(
    rid: str,
    *,
    priority=None,
):
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", [1]),
        sampling_params=SamplingParams(),
        priority=priority,
    )


def _make_scheduler(*, enable_metrics=True, enable_priority_scheduling=False):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.chunked_req = None
    scheduler.waiting_queue = []
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.last_batch = None
    scheduler.grammar_manager = MagicMock()
    scheduler.grammar_manager.abort_requests.return_value = []
    scheduler.enable_hicache_storage = False
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.ipc_channels = MagicMock()
    scheduler.ps = SimpleNamespace(
        pp_size=1,
        pp_rank=0,
        attn_tp_rank=0,
        attn_cp_rank=0,
    )
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=enable_metrics)
    scheduler.metrics_collector = MagicMock()
    scheduler.metrics_collector.aborted_request_labels = {
        "model_name": "test-model",
        "engine_type": "unified",
    }
    scheduler.enable_priority_scheduling = enable_priority_scheduling
    return scheduler


class TestSchedulerAbortMetrics(CustomTestCase):
    def test_abort_all_counts_each_matched_request(self):
        scheduler = _make_scheduler()
        waiting_reqs = [_make_req("waiting-0"), _make_req("waiting-1")]
        running_req = _make_req("running-0")
        scheduler.waiting_queue = list(waiting_reqs)
        scheduler.running_batch.reqs = [running_req]

        scheduler.abort_request(AbortReq(rid="", abort_all=True))

        self.assertEqual(scheduler.waiting_queue, [])
        self.assertIsInstance(running_req.to_finish, FINISH_ABORT)
        self.assertEqual(
            scheduler.metrics_collector.observe_one_aborted_request.call_count,
            3,
        )
        for req in [*waiting_reqs, running_req]:
            self.assertTrue(req._abort_metric_accounted)

    def test_prefix_abort_counts_each_matching_request(self):
        scheduler = _make_scheduler()
        matching_reqs = [_make_req("batch-0"), _make_req("batch-1")]
        other_req = _make_req("other-0")
        scheduler.waiting_queue = [*matching_reqs, other_req]

        scheduler.abort_request(AbortReq(rid="batch-", abort_all=False))

        self.assertEqual(scheduler.waiting_queue, [other_req])
        self.assertEqual(
            scheduler.metrics_collector.observe_one_aborted_request.call_count,
            2,
        )
        for req in matching_reqs:
            self.assertTrue(req._abort_metric_accounted)
        self.assertFalse(other_req._abort_metric_accounted)

    def test_non_matching_abort_is_not_counted(self):
        scheduler = _make_scheduler()
        running_req = _make_req("other-rid")
        scheduler.running_batch.reqs = [running_req]

        scheduler.abort_request(AbortReq(rid="missing-rid", abort_all=False))

        scheduler.metrics_collector.observe_one_aborted_request.assert_not_called()
        self.assertFalse(running_req._abort_metric_accounted)
        self.assertIsNone(running_req.to_finish)

    def test_repeated_abort_of_running_request_is_counted_once(self):
        scheduler = _make_scheduler()
        running_req = _make_req("request-0")
        scheduler.running_batch.reqs = [running_req]
        abort_req = AbortReq(rid="request-0", abort_all=False)

        scheduler.abort_request(abort_req)
        scheduler.abort_request(abort_req)

        scheduler.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_non_primary_scheduler_ranks_do_not_emit(self):
        for rank_name in ("pp_rank", "attn_tp_rank", "attn_cp_rank"):
            with self.subTest(rank_name=rank_name):
                scheduler = _make_scheduler()
                setattr(scheduler.ps, rank_name, 1)
                req = _make_req("request-0")

                scheduler._account_aborted_request(req)

                self.assertTrue(req._abort_metric_accounted)
                observe_abort = scheduler.metrics_collector.observe_one_aborted_request
                observe_abort.assert_not_called()

    def test_metrics_disabled_does_not_emit(self):
        scheduler = _make_scheduler(enable_metrics=False)
        req = _make_req("request-0")

        scheduler._account_aborted_request(req)

        self.assertTrue(req._abort_metric_accounted)
        scheduler.metrics_collector.observe_one_aborted_request.assert_not_called()


class TestTokenizerAbortMetrics(CustomTestCase):
    def test_tokenizer_only_dispatches_abort(self):
        tokenizer_manager = TokenizerManager.__new__(TokenizerManager)
        tokenizer_manager.server_args = SimpleNamespace(tokenizer_worker_num=1)
        tokenizer_manager.rid_to_state = {"request-0": object()}
        tokenizer_manager._dispatch_to_scheduler = MagicMock()
        tokenizer_manager.enable_metrics = True
        tokenizer_manager.metrics_collector = MagicMock()

        tokenizer_manager.abort_request(rid="request-0")

        tokenizer_manager._dispatch_to_scheduler.assert_called_once_with(
            AbortReq(rid="request-0", abort_all=False)
        )
        observe_abort = tokenizer_manager.metrics_collector.observe_one_aborted_request
        observe_abort.assert_not_called()


class TestAbortMetricCollector(CustomTestCase):
    def test_build_labels_without_server_args_returns_copy(self):
        labels = {"model_name": "test-model", "tp_rank": 0}

        aborted_labels = _build_aborted_request_labels(labels, None)

        self.assertEqual(aborted_labels, labels)
        self.assertIsNot(aborted_labels, labels)

    def test_build_labels_uses_service_level_dimensions(self):
        labels = {
            "model_name": "test-model",
            "engine_type": "unified",
            "tp_rank": 3,
            "pp_rank": 2,
            "priority": "",
        }
        server_args = SimpleNamespace(
            served_model_name="test-model",
            disaggregation_mode=DisaggregationMode.NULL.value,
            extra_metric_labels={"cluster": "prod-a"},
        )

        aborted_labels = _build_aborted_request_labels(labels, server_args)

        self.assertEqual(
            aborted_labels,
            {
                "model_name": "test-model",
                "engine_type": "unified",
                "priority": "",
                "cluster": "prod-a",
            },
        )

    def test_scheduler_collector_increments_abort_counter(self):
        collector = SchedulerMetricsCollector.__new__(SchedulerMetricsCollector)
        collector.num_aborted_requests_total = MagicMock()
        labels = {"model_name": "test-model", "engine_type": "unified"}

        collector.observe_one_aborted_request(labels)

        collector.num_aborted_requests_total.labels.assert_called_once_with(**labels)
        bound_counter = collector.num_aborted_requests_total.labels.return_value
        bound_counter.inc.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
