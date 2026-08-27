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
        self.assertEqual(args.proactive_decode_demotion_output_len_threthold, 8)
        self.assertEqual(args.proactive_decode_demotion_cache_usage, 0.95)
        self.assertEqual(args.proactive_decode_safe_cache_usage, 0.75)
        self.assertEqual(args.candidate_demotion_output_len_threthold, 2.0)
        self.assertEqual(args.proactive_demotion_recovery_duration, 180.0)

        with self.assertRaises(ValueError):
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                proactive_decode_demotion_cache_usage=1.1,
            ).resolve_once()

    def _make_demoted_queue(self, req, recovery_duration):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.demotion_queue = [
            DemotedRequest(req=req, demoted_start_time=0.0)
        ]
        queue.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                proactive_demotion_recovery_duration=recovery_duration
            )
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

        with patch(
            "sglang.srt.disaggregation.decode.time.monotonic", return_value=15.0
        ), patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=SimpleNamespace(
                disaggregation_decode_retraction_backup="cpu_tensor"
            ),
        ), patch("sglang.srt.disaggregation.decode.retraction_restore") as restore:
            self.assertEqual(queue.resume_demoted_reqs(), [req])
        self.assertEqual(queue.demotion_queue, [])
        self.assertFalse(req.is_retracted)
        self.assertFalse(req.is_demoted)
        restore.assert_called_once()

    def _make_retract_check_scheduler(self, demotion_queue):
        return SimpleNamespace(
            decode_metric_collector=SimpleNamespace(maybe_update=lambda: None),
            if_output_len_imbalance=True,
            disagg_decode_prealloc_queue=SimpleNamespace(
                demotion_queue=demotion_queue
            ),
            pool_stats_observer=SimpleNamespace(
                get_pool_stats=lambda: SimpleNamespace(
                    get_max_pool_usage=lambda: 0.96
                )
            ),
            server_args=SimpleNamespace(
                proactive_decode_demotion_output_len_threthold=8,
                proactive_decode_demotion_cache_usage=0.95,
            ),
        )

    def test_no_new_demotion_wave_while_queue_nonempty(self):
        """Demotion freed GPU KV that new admissions consumed, so each wave
        demoted further requests and the CPU backup grew without bound; a
        non-empty demotion queue must block the next wave."""
        busy = self._make_retract_check_scheduler(demotion_queue=[MagicMock()])
        self.assertFalse(
            SchedulerDisaggregationDecodeMixin.need_to_proactive_retract_request(
                busy
            )
        )
        idle = self._make_retract_check_scheduler(demotion_queue=[])
        self.assertTrue(
            SchedulerDisaggregationDecodeMixin.need_to_proactive_retract_request(
                idle
            )
        )

    def test_empty_window_clears_stale_imbalance_flag(self):
        """An early return on an empty quantile window left a stale
        if_output_len_imbalance=True driving demotion waves forever."""
        scheduler = self._make_retract_check_scheduler(demotion_queue=[])
        scheduler.decode_metric_collector = SimpleNamespace(
            maybe_update=lambda: (None, None)
        )
        self.assertFalse(
            SchedulerDisaggregationDecodeMixin.need_to_proactive_retract_request(
                scheduler
            )
        )
        self.assertFalse(scheduler.if_output_len_imbalance)

    @staticmethod
    def _make_demotion_candidate(rid, seqlen, output_len):
        return SimpleNamespace(
            rid=rid,
            seqlen=seqlen,
            output_ids=[0] * output_len,
            origin_input_ids=[0] * (seqlen - output_len),
            is_retracted=False,
            is_demoted=False,
            is_demoted_recovered=False,
            finished=lambda: False,
            sampling_params=SimpleNamespace(max_new_tokens=128),
            time_stats=SimpleNamespace(set_retract_time=MagicMock()),
        )

    def test_proactive_demotion_filters_and_repeats_to_safe_usage(self):
        make_req = self._make_demotion_candidate

        short = make_req("short", 10, 5)
        medium = make_req("medium", 20, 11)
        long = make_req("long", 30, 30)

        release_calls = []

        class Batch:
            reqs = [long, short, medium]
            batch_is_full = True

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
                release_calls.append((victim.rid, index, is_demoted))
                return True

            def filter_batch(self, keep_indices):
                self.reqs = [self.reqs[i] for i in keep_indices]

        demotion_queue = MagicMock()
        metrics_collector = MagicMock()
        usages = iter((0.95, 0.88, 0.84))
        scheduler = SimpleNamespace(
            running_batch=Batch(),
            waiting_queue=[],
            enable_overlap=False,
            server_args=SimpleNamespace(
                candidate_demotion_output_len_threthold=1.0,
                proactive_decode_safe_cache_usage=0.85,
            ),
            decode_metric_collector=SimpleNamespace(
                p50_output_len=10, get_output_len=lambda: (10, 32)
            ),
            pool_stats_observer=SimpleNamespace(
                get_pool_stats=lambda: SimpleNamespace(
                    get_max_pool_usage=lambda: next(usages)
                )
            ),
            disagg_decode_prealloc_queue=demotion_queue,
            new_token_ratio_tracker=SimpleNamespace(current=0.0),
            metrics_reporter=SimpleNamespace(
                num_demoted_reqs=0,
                enable_metrics=True,
                metrics_collector=metrics_collector,
            ),
        )

        self.assertTrue(
            SchedulerDisaggregationDecodeMixin.proactively_demote_longest_request(
                scheduler
            )
        )
        self.assertEqual(scheduler.running_batch.reqs, [short])
        self.assertEqual(
            [call.args[0] for call in demotion_queue.add_demoted_req.call_args_list],
            [long, medium],
        )
        self.assertFalse(long.is_retracted)
        self.assertTrue(long.is_demoted)
        self.assertFalse(medium.is_retracted)
        self.assertTrue(medium.is_demoted)
        self.assertEqual(release_calls, [("long", 0, True), ("medium", 1, True)])
        self.assertEqual(scheduler.metrics_reporter.num_demoted_reqs, 2)
        metrics_collector.increment_demoted_reqs.assert_called_once_with(
            num_demoted_reqs=2,
            num_demoted_input_tokens=9,
            num_demoted_output_tokens=41,
        )

    def test_demote_recovered_waiting_victim_without_batch_lookup(self):
        """A demoted-recovered candidate lives in the waiting queue, not the
        running batch; looking it up via batch.reqs.index() raised ValueError
        and crashed the scheduler."""
        running = self._make_demotion_candidate("running", 20, 15)
        recovered = self._make_demotion_candidate("recovered", 40, 35)
        recovered.is_demoted_recovered = True

        batch_release_calls = []

        class Batch:
            reqs = [running]
            batch_is_full = True

            def is_empty(self):
                return not self.reqs

            def batch_size(self):
                return len(self.reqs)

            def release_req(self, index, _, __, *, is_demoted=False):
                batch_release_calls.append((self.reqs[index].rid, is_demoted))
                return True

            def filter_batch(self, keep_indices):
                self.reqs = [self.reqs[i] for i in keep_indices]

        demotion_queue = MagicMock()
        usages = iter((0.95, 0.84))
        scheduler = SimpleNamespace(
            running_batch=Batch(),
            waiting_queue=[recovered],
            enable_overlap=False,
            server_args=SimpleNamespace(
                candidate_demotion_output_len_threthold=1.0,
                proactive_decode_safe_cache_usage=0.85,
            ),
            decode_metric_collector=SimpleNamespace(
                p50_output_len=10, get_output_len=lambda: (10, 32)
            ),
            pool_stats_observer=SimpleNamespace(
                get_pool_stats=lambda: SimpleNamespace(
                    get_max_pool_usage=lambda: next(usages)
                )
            ),
            disagg_decode_prealloc_queue=demotion_queue,
            new_token_ratio_tracker=SimpleNamespace(current=0.0),
            metrics_reporter=SimpleNamespace(
                num_demoted_reqs=0,
                enable_metrics=False,
                metrics_collector=MagicMock(),
            ),
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            tree_cache=MagicMock(),
            hisparse_coordinator=None,
        )

        with patch(
            "sglang.srt.disaggregation.decode.release_req", return_value=True
        ) as module_release:
            self.assertTrue(
                SchedulerDisaggregationDecodeMixin.proactively_demote_longest_request(
                    scheduler
                )
            )

        # The waiting-queue victim goes through the module-level release path;
        # the running batch is untouched by it.
        module_release.assert_called_once()
        self.assertIs(module_release.call_args.kwargs["req"], recovered)
        self.assertTrue(module_release.call_args.kwargs["is_demoted"])
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(batch_release_calls, [])
        self.assertEqual(scheduler.running_batch.reqs, [running])
        demotion_queue.add_demoted_req.assert_called_once_with(recovered)


if __name__ == "__main__":
    unittest.main()
