import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.decode import DecodeRequest
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_policy import (
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_components.load_inquirer import (
    SchedulerLoadInquirer,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.observability.metrics_collector import SchedulerStats
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq, decode_queue

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSchedulePolicyWaitingQueueMatching(unittest.TestCase):
    def make_policy(self, policy, supports_fast_match_prefix):
        schedule_policy = object.__new__(SchedulePolicy)
        schedule_policy.policy = policy
        schedule_policy.tree_cache = SimpleNamespace(
            supports_fast_match_prefix=lambda: supports_fast_match_prefix
        )
        return schedule_policy

    def test_lpm_queue_limit_respects_fast_matching_capability(self):
        policy = self.make_policy(CacheAwarePolicy.LPM, False)
        self.assertTrue(policy.waiting_queue_prefix_matched([None] * 128))
        self.assertFalse(policy.waiting_queue_prefix_matched([None] * 129))

        policy.tree_cache = SimpleNamespace(supports_fast_match_prefix=lambda: True)
        self.assertTrue(policy.waiting_queue_prefix_matched([None] * 129))


class TestSchedulerLoadInquirer(unittest.TestCase):
    def make_inquirer(
        self,
        waiting_queue_prefix_matched=True,
        storage_span=None,
        disaggregation_mode=DisaggregationMode.NULL,
    ):
        waiting_req = SimpleNamespace(
            seqlen=100,
            num_matched_prefix_tokens=20,
            cache_request_handle=CacheRequestHandle("waiting", 0),
        )
        chunked_req = SimpleNamespace(seqlen=50, prefix_len=10)
        return SchedulerLoadInquirer(
            disaggregation_mode=disaggregation_mode,
            server_args=None,
            max_total_num_tokens=1000,
            max_running_requests=10,
            pool_stats_observer=None,
            tp_worker=None,
            token_to_kv_pool_allocator=None,
            spec_algorithm=None,
            get_running_batch=None,
            get_waiting_queue=lambda: [waiting_req],
            waiting_queue_prefix_matched=lambda: waiting_queue_prefix_matched,
            get_chunked_req=lambda: chunked_req,
            get_recent_cache_hit_rate=lambda: 0.75,
            get_stats=None,
            get_request_lists=None,
            get_disagg_decode_prealloc_queue=None,
            get_spec_total_num_accept_tokens=None,
            get_spec_total_num_forward_ct=None,
            get_total_prefill_uncached_tokens=None,
            get_total_prefill_busy_us=None,
            get_decode_moment_totals=None,
            get_waiting_storage_span=Mock(return_value=storage_span),
        )

    def test_waiting_tokens_are_estimated_when_prefix_matching_is_skipped(self):
        inquirer = self.make_inquirer(waiting_queue_prefix_matched=False)

        self.assertEqual(
            SchedulerLoadInquirer.get_input_token_load(inquirer),
            65,
        )

    def test_waiting_tokens_use_exact_match_when_prefix_matching_is_done(self):
        inquirer = self.make_inquirer(waiting_queue_prefix_matched=True)

        self.assertEqual(
            SchedulerLoadInquirer.get_input_token_load(inquirer),
            120,
        )

    def test_storage_exclusion_is_clipped_to_uncached_prompt(self):
        for span, expected in (
            (None, 120),
            ((0, 10), 120),
            ((0, 40), 100),
            ((20, 80), 60),
            ((60, 120), 80),
            ((20, 120), 40),
            ((20, 20), 120),
            ((100, 120), 120),
        ):
            with self.subTest(span=span):
                inquirer = self.make_inquirer(storage_span=span)
                self.assertEqual(inquirer.get_input_token_load(), expected)
                inquirer.get_waiting_storage_span.assert_called_once_with(
                    CacheRequestHandle("waiting", 0)
                )

    def test_decode_loads_count_prealloc_requests_waiting_for_input(self):
        """A preallocated request whose input has arrived is ready, with its tokens."""
        ready, pending = FakeReq(seqlen=30), FakeReq(seqlen=70)
        prealloc = decode_queue(
            ListKind.PD_PREALLOC,
            [
                DecodeRequest(req=ready, kv_receiver=None, waiting_for_input=True),
                DecodeRequest(req=pending, kv_receiver=None),
            ],
        )
        inquirer = replace(
            self.make_inquirer(disaggregation_mode=DisaggregationMode.DECODE),
            pool_stats_observer=SimpleNamespace(
                get_pool_stats=lambda: SimpleNamespace(
                    get_kv_token_stats=lambda: (0, 0.0)
                )
            ),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            get_running_batch=lambda: SimpleNamespace(reqs=[]),
            get_stats=SchedulerStats,
            get_disagg_decode_prealloc_queue=lambda: SimpleNamespace(queue=prealloc),
            get_request_lists=lambda: {
                ListKind.WAITING: [],
                ListKind.PD_PREALLOC: prealloc,
                ListKind.PD_RECEIVING: decode_queue(ListKind.PD_RECEIVING),
                ListKind.PD_RETRACTED: IndexedReqList(ListKind.PD_RETRACTED),
            },
            get_total_prefill_uncached_tokens=lambda: 0,
            get_total_prefill_busy_us=lambda: 0,
            get_decode_moment_totals=lambda: (0,),
        )
        module = "sglang.srt.managers.scheduler_components.load_inquirer"
        with (
            patch(
                f"{module}.get_lora", return_value=SimpleNamespace(enable_lora=False)
            ),
            patch(f"{module}.get_parallel", return_value=SimpleNamespace(dp_rank=0)),
        ):
            loads = inquirer.get_loads()

        self.assertEqual(loads.disaggregation.decode_prealloc_queue_reqs, 2)
        self.assertEqual(loads.queues.prealloc_ready, 1)
        self.assertEqual(loads.num_prealloc_ready_tokens, 30)

    def test_storage_exclusion_requires_per_request_matching(self):
        for prefix_matched, expected in ((True, 60), (False, 65)):
            with self.subTest(prefix_matched=prefix_matched):
                inquirer = self.make_inquirer(
                    waiting_queue_prefix_matched=prefix_matched,
                    storage_span=(20, 80),
                )
                self.assertEqual(inquirer.get_input_token_load(), expected)
                if not prefix_matched:
                    inquirer.get_waiting_storage_span.assert_not_called()

    def test_storage_exclusion_does_not_change_decode(self):
        for prefix_matched in (False, True):
            with self.subTest(prefix_matched=prefix_matched):
                inquirer = self.make_inquirer(
                    waiting_queue_prefix_matched=prefix_matched,
                    storage_span=(20, 100),
                    disaggregation_mode=DisaggregationMode.DECODE,
                )
                self.assertEqual(inquirer.get_input_token_load(), 0)
                inquirer.get_waiting_storage_span.assert_not_called()

    def test_storage_exclusion_preserves_the_historical_hit_rate_estimate(self):
        for span in (None, (20, 50), (20, 80), (80, 100), (0, 120)):
            with self.subTest(span=span):
                inquirer = self.make_inquirer(
                    waiting_queue_prefix_matched=False,
                    storage_span=span,
                )
                self.assertEqual(inquirer.get_input_token_load(), 65)
                inquirer.get_waiting_storage_span.assert_not_called()

    def test_historical_estimate_already_includes_storage_hits(self):
        requests = [
            SimpleNamespace(seqlen=100, cache_request_handle=CacheRequestHandle(rid, 0))
            for rid in ("storage_hit", "cold")
        ]
        get_storage_span = Mock(
            side_effect=lambda handle: (0, 80) if handle.rid == "storage_hit" else None
        )
        inquirer = replace(
            self.make_inquirer(waiting_queue_prefix_matched=False),
            get_waiting_queue=lambda: requests,
            get_chunked_req=lambda: None,
            get_recent_cache_hit_rate=lambda: 0.4,
            get_waiting_storage_span=get_storage_span,
        )

        # One 80-token storage hit and one cold prompt need 20 + 100 tokens.
        self.assertEqual(inquirer.get_input_token_load(), 120)
        get_storage_span.assert_not_called()

    def test_storage_exclusion_without_a_provider_keeps_the_original_estimate(self):
        inquirer = replace(
            self.make_inquirer(),
            get_waiting_storage_span=None,
        )
        self.assertEqual(inquirer.get_input_token_load(), 120)

    def test_scheduler_wires_storage_provider(self):
        from sglang.srt.managers.scheduler import Scheduler

        req = self.make_inquirer().get_waiting_queue()[0]
        get_storage_span = Mock(return_value=(20, 80))
        scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.PREFILL,
            server_args=None,
            max_total_num_tokens=1000,
            kv_shard_widening=1,
            max_running_requests=10,
            pool_stats_observer=None,
            tp_worker=None,
            token_to_kv_pool_allocator=None,
            spec_algorithm=None,
            tree_cache=SimpleNamespace(waiting_storage_span=get_storage_span),
            policy=SimpleNamespace(waiting_queue_prefix_matched=lambda queue: True),
            metrics_reporter=SimpleNamespace(recent_cache_hit_rate=0.75),
            waiting_queue=[req],
            chunked_req=None,
            request_lists=None,
        )

        Scheduler.init_load_inquirer(scheduler)

        inquirer = scheduler.load_inquirer
        self.assertIs(inquirer.get_waiting_storage_span, get_storage_span)
        self.assertEqual(inquirer.get_input_token_load(), 20)
        get_storage_span.assert_called_once_with(req.cache_request_handle)

    def test_output_token_load_counts_total_generation_including_decoded(self):
        """The 25 tokens `running` has already emitted stay in its score: the
        load is the request's whole estimated generation, not what is left, so
        it matches what the DP controller charged the rank at dispatch."""
        running = SimpleNamespace(
            sampling_params=SimpleNamespace(max_new_tokens=100),
            output_ids=list(range(25)),
        )
        waiting = SimpleNamespace(
            sampling_params=SimpleNamespace(max_new_tokens=40),
            output_ids=[],
        )
        inquirer = SimpleNamespace(
            get_running_batch=lambda: SimpleNamespace(reqs=[running]),
            get_waiting_queue=lambda: [waiting],
            get_chunked_req=lambda: None,
        )

        self.assertEqual(
            SchedulerLoadInquirer.get_output_token_load(inquirer),
            140,
        )

    def test_output_token_load_clamps_unbounded_request_to_ceiling(self):
        """A `max_new_tokens=None` request is resolved scheduler-side to a
        context-sized cap; scoring it at face value would let one request swamp
        the rank's decode score, so it is clamped to
        `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION`."""
        unbounded = SimpleNamespace(
            sampling_params=SimpleNamespace(max_new_tokens=131_000),
            output_ids=[],
        )
        short = SimpleNamespace(
            sampling_params=SimpleNamespace(max_new_tokens=40),
            output_ids=[],
        )
        inquirer = SimpleNamespace(
            get_running_batch=lambda: SimpleNamespace(reqs=[unbounded]),
            get_waiting_queue=lambda: [short],
            get_chunked_req=lambda: None,
        )

        with envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.override(4096):
            self.assertEqual(
                SchedulerLoadInquirer.get_output_token_load(inquirer),
                4096 + 40,
            )

    def test_output_token_load_includes_chunked_req(self):
        chunked = SimpleNamespace(
            sampling_params=SimpleNamespace(max_new_tokens=30),
            output_ids=[],
        )
        inquirer = SimpleNamespace(
            get_running_batch=lambda: SimpleNamespace(reqs=[]),
            get_waiting_queue=lambda: [],
            get_chunked_req=lambda: chunked,
        )

        self.assertEqual(SchedulerLoadInquirer.get_output_token_load(inquirer), 30)


if __name__ == "__main__":
    unittest.main()
