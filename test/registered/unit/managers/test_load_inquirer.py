import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_components.load_inquirer import (
    SchedulerLoadInquirer,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSchedulePolicyWaitingQueueMatching(unittest.TestCase):
    def make_policy(self, policy, supports_fast_match_prefix):
        schedule_policy = object.__new__(SchedulePolicy)
        schedule_policy.policy = policy
        schedule_policy.tree_cache = SimpleNamespace(
            supports_fast_match_prefix=lambda: supports_fast_match_prefix
        )
        return schedule_policy

    def test_cache_agnostic_policy_requires_fast_matching(self):
        policy = self.make_policy(CacheAgnosticPolicy.FCFS, False)
        self.assertFalse(policy.waiting_queue_prefix_matched([]))

        policy.tree_cache = SimpleNamespace(supports_fast_match_prefix=lambda: True)
        self.assertTrue(policy.waiting_queue_prefix_matched([]))

    def test_lpm_queue_limit_respects_fast_matching_capability(self):
        policy = self.make_policy(CacheAwarePolicy.LPM, False)
        self.assertTrue(policy.waiting_queue_prefix_matched([None] * 128))
        self.assertFalse(policy.waiting_queue_prefix_matched([None] * 129))

        policy.tree_cache = SimpleNamespace(supports_fast_match_prefix=lambda: True)
        self.assertTrue(policy.waiting_queue_prefix_matched([None] * 129))


class TestSchedulerLoadInquirer(unittest.TestCase):
    def make_inquirer(self, waiting_queue_prefix_matched):
        waiting_req = SimpleNamespace(seqlen=100, num_matched_prefix_tokens=20)
        chunked_req = SimpleNamespace(seqlen=50, prefix_len=10)
        return SimpleNamespace(
            disaggregation_mode=DisaggregationMode.NULL,
            get_waiting_queue=lambda: [waiting_req],
            waiting_queue_prefix_matched=lambda: waiting_queue_prefix_matched,
            get_chunked_req=lambda: chunked_req,
            get_recent_cache_hit_rate=lambda: 0.75,
        )

    def test_waiting_tokens_are_estimated_when_prefix_matching_is_skipped(self):
        inquirer = self.make_inquirer(waiting_queue_prefix_matched=False)

        self.assertEqual(
            SchedulerLoadInquirer.get_num_waiting_uncached_tokens(inquirer),
            65,
        )

    def test_waiting_tokens_use_exact_match_when_prefix_matching_is_done(self):
        inquirer = self.make_inquirer(waiting_queue_prefix_matched=True)

        self.assertEqual(
            SchedulerLoadInquirer.get_num_waiting_uncached_tokens(inquirer),
            120,
        )


class TestLoadSnapshotPendingWork(CustomTestCase):
    """total_tokens DP balancing replaces its own estimate for a rank with the
    rank's snapshot, so work already sent to the rank must show up in it."""

    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def snapshot(self, *, used_tokens=0, grammar_queue=(), chunked_req=None):
        stats = SimpleNamespace(
            kv_transfer_speed_gb_s=0.0,
            kv_transfer_latency_ms=0.0,
            num_grammar_queue_reqs=len(grammar_queue),
            num_paused_reqs=0,
            num_retracted_reqs=0,
            gen_throughput=0.0,
            cache_hit_rate=0.0,
            utilization=0.0,
        )
        pool_stats = SimpleNamespace(get_kv_token_stats=lambda: (used_tokens, 0.0))
        inquirer = SchedulerLoadInquirer(
            disaggregation_mode=DisaggregationMode.PREFILL,
            server_args=None,
            max_total_num_tokens=1 << 20,
            max_running_requests=64,
            pool_stats_observer=SimpleNamespace(get_pool_stats=lambda: pool_stats),
            tp_worker=None,
            token_to_kv_pool_allocator=None,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            get_running_batch=lambda: SimpleNamespace(reqs=[]),
            get_waiting_queue=lambda: [],
            waiting_queue_prefix_matched=lambda: False,
            get_recent_cache_hit_rate=lambda: 0.0,
            get_stats=lambda: stats,
            get_chunked_req=lambda: chunked_req,
            get_grammar_queue=lambda: list(grammar_queue),
            get_disagg_prefill_bootstrap_queue=lambda: SimpleNamespace(queue=[]),
            get_disagg_prefill_inflight_queue=lambda: [],
            get_disagg_decode_prealloc_queue=lambda: None,
            get_disagg_decode_transfer_queue=lambda: None,
            get_spec_total_num_accept_tokens=lambda: 0,
            get_spec_total_num_forward_ct=lambda: 0,
            get_total_prefill_uncached_tokens=lambda: 0,
            get_total_prefill_busy_us=lambda: 0,
            get_decode_moment_totals=lambda: [0.0] * 6,
        )
        with get_parallel().override(dp_rank=0):
            return inquirer.get_loads()

    def test_requests_waiting_for_their_grammar_count_as_pending(self):
        """A constrained request waits in the grammar queue while its grammar
        compiles; a rank that reports it as idle meanwhile gets the next ones."""
        loads = self.snapshot(grammar_queue=[SimpleNamespace(seqlen=50_000)])
        self.assertEqual(loads.num_total_tokens, 50_000)
        self.assertEqual(loads.num_waiting_reqs, 1)

    def test_a_chunked_request_counts_its_unprefilled_tokens(self):
        """Mid chunked prefill the KV pool holds only the chunks already run; the
        rest of the prompt is still pending, and the run chunks count once."""
        req = SimpleNamespace(seqlen=50_000, prefix_len=0, extend_end=8_192)
        loads = self.snapshot(used_tokens=8_192, chunked_req=req)
        self.assertEqual(loads.num_total_tokens, 50_000)


if __name__ == "__main__":
    unittest.main()
