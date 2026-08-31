import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DemotedRequest,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.observability.decode_metric_collector import (
    DEFAULT_OUTPUT_LEN_BUCKETS,
    DecodeMetricCollector,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_CPU_TENSOR_DISAGG = SimpleNamespace(
    disaggregation_decode_retraction_backup="cpu_tensor"
)


class _FakeBatch:
    """Minimal running-batch stand-in recording release_req calls."""

    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.batch_is_full = True
        self.release_calls = []

    def is_empty(self):
        return not self.reqs

    def batch_size(self):
        return len(self.reqs)

    def release_req(self, index, _, __, *, is_demoted=False):
        victim = self.reqs[index]
        if is_demoted:
            victim.is_demoted = True
        else:
            victim.is_retracted = True
        self.release_calls.append((victim.rid, index, is_demoted))
        return True

    def filter_batch(self, keep_indices):
        self.reqs = [self.reqs[i] for i in keep_indices]


def _make_demotion_candidate(rid, seqlen, output_len, *, last_demote_output_len=0):
    return SimpleNamespace(
        rid=rid,
        seqlen=seqlen,
        output_ids=[0] * output_len,
        origin_input_ids=[0] * (seqlen - output_len),
        is_retracted=False,
        is_demoted=False,
        last_demote_output_len=last_demote_output_len,
        finished=lambda: False,
        sampling_params=SimpleNamespace(max_new_tokens=128),
        time_stats=SimpleNamespace(set_retract_time=MagicMock()),
    )


def _make_demotion_scheduler(batch, *, budget, enable_metrics=False):
    """Scheduler stub wired to a real DecodePreallocQueue so add_demoted_req
    performs its actual budget deduction."""
    prealloc_queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    prealloc_queue.demotion_queue = []
    scheduler = SimpleNamespace(
        running_batch=batch,
        enable_overlap=False,
        remain_cpu_demote_tokens=budget,
        server_args=SimpleNamespace(
            proactive_demotion_max_input_len=10,
            proactive_demotion_min_output_len=11,
        ),
        disagg_decode_prealloc_queue=prealloc_queue,
        new_token_ratio_tracker=SimpleNamespace(current=0.0),
        metrics_reporter=SimpleNamespace(
            num_demoted_reqs=0,
            enable_metrics=enable_metrics,
            metrics_collector=MagicMock(),
        ),
        req_to_token_pool=MagicMock(),
        token_to_kv_pool_allocator=MagicMock(),
        tree_cache=MagicMock(),
        hisparse_coordinator=None,
    )
    prealloc_queue.scheduler = scheduler
    return scheduler


class TestProactiveDecodeDemotion(CustomTestCase):
    def test_req_tracks_demote_separately_from_retract(self):
        demoted_req = Req(
            "demoted", "", array("q", [1]), SamplingParams(max_new_tokens=8)
        )
        demoted_req.reset_for_retract(is_demoted=True)
        self.assertTrue(demoted_req.is_demoted)
        self.assertFalse(demoted_req.is_retracted)
        self.assertEqual(demoted_req.retraction_count, 0)

        retracted_req = Req(
            "retracted", "", array("q", [1]), SamplingParams(max_new_tokens=8)
        )
        retracted_req.reset_for_retract()
        self.assertFalse(retracted_req.is_demoted)
        self.assertTrue(retracted_req.is_retracted)
        self.assertEqual(retracted_req.retraction_count, 1)

    def test_collector_uses_generation_metric_buckets(self):
        self.assertEqual(DEFAULT_OUTPUT_LEN_BUCKETS[-1], 1_100_000)
        self.assertEqual(len(DEFAULT_OUTPUT_LEN_BUCKETS), 35)

    def test_collector_window_and_quantiles(self):
        now = [0.0]
        collector = DecodeMetricCollector(
            bucket_bounds=[0, 1, 2, 4, 8, 16, 32],
            clock=lambda: now[0],
        )
        for length in (1, 2, 4, 8, 8, 32, 32, 32, 32, 32):
            collector.observe_output_len(length)
        self.assertIsNone(collector.maybe_update())
        now[0] = 15.0
        self.assertEqual(collector.maybe_update(), (8, 32))
        self.assertIsNone(collector.maybe_update())

        collector.observe_output_len(1)
        now[0] = 30.0
        self.assertEqual(collector.maybe_update(), (1, 1))

    def test_server_args_defaults_and_validation(self):
        args = ServerArgs(model_path="dummy")
        args.resolve_once()
        self.assertFalse(args.enable_proactive_decode_promotion)
        self.assertEqual(args.proactive_decode_demotion_cache_usage, 0.70)
        self.assertEqual(args.proactive_safe_cpu_demote_cache_usage, 0.2)
        self.assertEqual(args.candidate_demotion_output_len_threthold, 2.0)
        self.assertEqual(args.proactive_demotion_max_input_len, 4096)
        self.assertEqual(args.proactive_demotion_min_output_len, 8192)
        self.assertEqual(args.proactive_demotion_recovery_duration, 180.0)

        with self.assertRaises(ValueError):
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                proactive_decode_demotion_cache_usage=1.1,
            ).resolve_once()

        with self.assertRaises(ValueError):
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                proactive_safe_cpu_demote_cache_usage=0.0,
            ).resolve_once()

        with self.assertRaises(ValueError):
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                proactive_demotion_max_input_len=0,
            ).resolve_once()

        with self.assertRaises(ValueError):
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                proactive_demotion_min_output_len=0,
            ).resolve_once()

    def _make_demoted_queue(self, req, recovery_duration):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.demotion_queue = [
            DemotedRequest(req=req, demoted_start_time=0.0, demoted_tokens=7)
        ]
        queue.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                proactive_demotion_recovery_duration=recovery_duration
            ),
            remain_cpu_demote_tokens=0,
        )
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
        queue.token_to_kv_pool_allocator = MagicMock()
        queue.tree_cache = MagicMock()
        queue._uses_swa_tail_prealloc = lambda: False
        queue._allocatable_token_budgets = lambda **_: 10
        queue._prealloc_required_tokens = lambda _: (1, 1)
        queue._pre_alloc = MagicMock()
        return queue

    def test_demoted_request_waits_then_restores(self):
        req = SimpleNamespace(is_retracted=False, is_demoted=True)
        queue = self._make_demoted_queue(req, recovery_duration=10.0)
        with patch(
            "sglang.srt.disaggregation.decode.time.monotonic", return_value=5.0
        ):
            self.assertEqual(queue.resume_demoted_reqs(), [])
        self.assertEqual(len(queue.demotion_queue), 1)
        self.assertEqual(queue.scheduler.remain_cpu_demote_tokens, 0)

        with patch(
            "sglang.srt.disaggregation.decode.time.monotonic", return_value=15.0
        ), patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=_CPU_TENSOR_DISAGG,
        ), patch("sglang.srt.disaggregation.decode.retraction_restore") as restore:
            self.assertEqual(queue.resume_demoted_reqs(), [req])
        self.assertEqual(queue.demotion_queue, [])
        self.assertFalse(req.is_retracted)
        self.assertFalse(req.is_demoted)
        restore.assert_called_once()
        # Resume returns the demoted tokens to the CPU offload budget.
        self.assertEqual(queue.scheduler.remain_cpu_demote_tokens, 7)

    def test_release_memory_occupation_returns_budget(self):
        """Dropping a demoted CPU backup must return its tokens to the budget,
        or the budget leaks and demotion locks up permanently."""
        req = SimpleNamespace(is_retracted=False, is_demoted=True)
        queue = self._make_demoted_queue(req, recovery_duration=10.0)
        queue.queue = []
        queue.retracted_queue = []
        queue.kv_manager = SimpleNamespace()
        queue._cancel_prefill_dp_rank_queries = lambda: None
        with patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=_CPU_TENSOR_DISAGG,
        ), patch("sglang.srt.disaggregation.decode.retraction_discard") as discard:
            queue.release_memory_occupation()
        discard.assert_called_once()
        self.assertEqual(queue.demotion_queue, [])
        self.assertEqual(queue.scheduler.remain_cpu_demote_tokens, 7)

    def _make_retract_check_scheduler(self, remain_cpu_demote_tokens):
        return SimpleNamespace(
            decode_metric_collector=SimpleNamespace(maybe_update=lambda: None),
            remain_cpu_demote_tokens=remain_cpu_demote_tokens,
            pool_stats_observer=SimpleNamespace(
                get_pool_stats=lambda: SimpleNamespace(
                    get_max_pool_usage=lambda: 0.96
                )
            ),
            server_args=SimpleNamespace(
                proactive_decode_demotion_cache_usage=0.70,
            ),
        )

    def test_demotion_gated_by_cpu_budget_not_queue(self):
        """The old gate blocked any new wave while the demotion queue was
        non-empty, so a mostly-recovered queue still froze demotion; the gate
        must instead track the remaining CPU token budget."""
        check = SchedulerDisaggregationDecodeMixin.need_to_proactive_retract_request
        self.assertFalse(check(self._make_retract_check_scheduler(0)))
        self.assertFalse(check(self._make_retract_check_scheduler(-5)))
        # Budget remaining allows a new wave even mid-recovery.
        self.assertTrue(check(self._make_retract_check_scheduler(100)))

    def test_triggers_without_output_len_quantiles(self):
        """The fixed rule must fire on cache pressure alone; an empty quantile
        window (no completed requests yet) must not suppress demotion."""
        scheduler = self._make_retract_check_scheduler(100)
        scheduler.decode_metric_collector = SimpleNamespace(
            maybe_update=lambda: (None, None)
        )
        self.assertTrue(
            SchedulerDisaggregationDecodeMixin.need_to_proactive_retract_request(
                scheduler
            )
        )

    def test_proactive_demotion_filters_and_spends_budget(self):
        short = _make_demotion_candidate("short", 10, 5)
        medium = _make_demotion_candidate("medium", 20, 11)
        long = _make_demotion_candidate("long", 30, 30)
        batch = _FakeBatch([long, short, medium])
        scheduler = _make_demotion_scheduler(batch, budget=40, enable_metrics=True)

        self.assertTrue(
            SchedulerDisaggregationDecodeMixin.proactively_demote_longest_request(
                scheduler
            )
        )
        demotion_queue = scheduler.disagg_decode_prealloc_queue.demotion_queue
        self.assertEqual(batch.reqs, [short])
        self.assertEqual([entry.req for entry in demotion_queue], [long, medium])
        self.assertEqual(
            [entry.demoted_tokens for entry in demotion_queue], [30, 20]
        )
        # 40 - 30 - 20: the last victim may overshoot the budget by one request.
        self.assertEqual(scheduler.remain_cpu_demote_tokens, -10)
        self.assertFalse(long.is_retracted)
        self.assertTrue(long.is_demoted)
        self.assertFalse(medium.is_retracted)
        self.assertTrue(medium.is_demoted)
        self.assertEqual(
            batch.release_calls, [("long", 0, True), ("medium", 1, True)]
        )
        self.assertEqual(scheduler.metrics_reporter.num_demoted_reqs, 2)
        scheduler.metrics_reporter.metrics_collector.increment_demoted_reqs.assert_called_once_with(
            num_demoted_reqs=2,
            num_demoted_input_tokens=9,
            num_demoted_output_tokens=41,
        )

    def test_demotion_stops_when_budget_exhausted(self):
        """The wave must stop on budget exhaustion even while over-long
        candidates remain; without the budget check it drained every
        candidate and the CPU backup grew past the configured cap."""
        medium = _make_demotion_candidate("medium", 20, 11)
        long = _make_demotion_candidate("long", 30, 30)
        batch = _FakeBatch([long, medium])
        scheduler = _make_demotion_scheduler(batch, budget=25)

        self.assertTrue(
            SchedulerDisaggregationDecodeMixin.proactively_demote_longest_request(
                scheduler
            )
        )
        # long (seqlen 30) exhausts the budget of 25; medium stays running.
        demotion_queue = scheduler.disagg_decode_prealloc_queue.demotion_queue
        self.assertEqual([entry.req for entry in demotion_queue], [long])
        self.assertEqual(batch.reqs, [medium])
        self.assertEqual(scheduler.remain_cpu_demote_tokens, -5)
        self.assertFalse(medium.is_demoted)

    def test_redemotion_requires_incremental_output(self):
        """Re-demotion must require min_output_len tokens generated since the
        last demotion; comparing total output length alone re-demoted a
        just-recovered request that had generated almost nothing new."""
        # Total output 30 but only 5 new tokens since the last demotion.
        recovered = _make_demotion_candidate(
            "recovered", 35, 30, last_demote_output_len=25
        )
        fresh = _make_demotion_candidate("fresh", 33, 30)
        batch = _FakeBatch([recovered, fresh])
        scheduler = _make_demotion_scheduler(batch, budget=100)

        self.assertTrue(
            SchedulerDisaggregationDecodeMixin.proactively_demote_longest_request(
                scheduler
            )
        )
        demotion_queue = scheduler.disagg_decode_prealloc_queue.demotion_queue
        self.assertEqual([entry.req for entry in demotion_queue], [fresh])
        self.assertEqual(batch.reqs, [recovered])
        self.assertFalse(recovered.is_demoted)
        # Demotion records the output length the next wave must build on.
        self.assertEqual(fresh.last_demote_output_len, 30)

    def test_demotion_queue_cache_usage_matches_budget_spend(self):
        """demotion_queue_cache_usage must equal the budget debited by
        add_demoted_req divided by the pool size, so the gauge tracks the
        proactive_safe_cpu_demote_cache_usage cap without separate accounting."""
        long = _make_demotion_candidate("long", 30, 20)
        medium = _make_demotion_candidate("medium", 20, 12)
        batch = _FakeBatch([long, medium])
        initial_budget = 100
        scheduler = _make_demotion_scheduler(batch, budget=initial_budget)
        queue = scheduler.disagg_decode_prealloc_queue
        queue.max_total_num_tokens = 500

        self.assertEqual(queue.demotion_queue_cache_usage(), 0.0)
        self.assertEqual(queue.demoted_reqs(), [])

        self.assertTrue(
            SchedulerDisaggregationDecodeMixin.proactively_demote_longest_request(
                scheduler
            )
        )

        spent = initial_budget - scheduler.remain_cpu_demote_tokens
        self.assertEqual(
            queue.demotion_queue_cache_usage(), spent / queue.max_total_num_tokens
        )
        self.assertEqual(queue.demoted_reqs(), [long, medium])


if __name__ == "__main__":
    unittest.main()
