import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DemotedRequest,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.observability.decode_metric_collector import (
    DEFAULT_OUTPUT_LEN_BUCKETS,
    DecodeMetricCollector,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestProactiveDecodeDemotion(CustomTestCase):
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
        self.assertEqual(args.proactive_decode_demotion_cache_usage, 0.9)
        self.assertEqual(args.proactive_decode_safe_cache_usage, 0.85)
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
        req = SimpleNamespace(is_retracted=True)
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
        restore.assert_called_once()

    def test_proactive_demotion_filters_and_repeats_to_safe_usage(self):
        def make_req(rid, seqlen, output_len):
            return SimpleNamespace(
                rid=rid,
                seqlen=seqlen,
                output_ids=[0] * output_len,
                is_retracted=False,
                finished=lambda: False,
                sampling_params=SimpleNamespace(max_new_tokens=128),
                time_stats=SimpleNamespace(set_retract_time=MagicMock()),
            )

        short = make_req("short", 10, 5)
        medium = make_req("medium", 20, 11)
        long = make_req("long", 30, 30)

        class Batch:
            reqs = [short, medium, long]
            batch_is_full = True

            def is_empty(self):
                return not self.reqs

            def batch_size(self):
                return len(self.reqs)

            def release_req(self, index, _, __):
                self.reqs[index].is_retracted = True
                return True

            def filter_batch(self, keep_indices):
                self.reqs = [self.reqs[i] for i in keep_indices]

        demotion_queue = MagicMock()
        usages = iter((0.95, 0.88, 0.84))
        scheduler = SimpleNamespace(
            running_batch=Batch(),
            enable_overlap=False,
            server_args=SimpleNamespace(
                candidate_demotion_output_len_threthold=1.0,
                proactive_decode_safe_cache_usage=0.85,
            ),
            decode_metric_collector=SimpleNamespace(p50_output_len=10),
            pool_stats_observer=SimpleNamespace(
                get_pool_stats=lambda: SimpleNamespace(
                    get_max_pool_usage=lambda: next(usages)
                )
            ),
            disagg_decode_prealloc_queue=demotion_queue,
            new_token_ratio_tracker=SimpleNamespace(current=0.0),
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


if __name__ == "__main__":
    unittest.main()
