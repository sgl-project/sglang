"""Unit tests for the UniBoost schedule policy and its gamma-Ada controller.

Runs the real SchedulePolicy and GammaAdaController on fake requests; no GPU.
"""

import itertools
import unittest
from unittest.mock import patch

from sglang.srt.managers import schedule_policy, uniboost_gamma_ada
from sglang.srt.managers.req_types import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.schedule_policy import (
    UNIBOOST_MATCH_WINDOW_SIZE,
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.uniboost_gamma_ada import (
    GAMMA_ADA_MAX,
    GAMMA_ADA_REFIT_EVERY,
    GammaAdaController,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FakeTimeStats:
    def __init__(self, arrival):
        self.scheduler_recv_time = arrival
        self.wait_queue_entry_time = arrival


_first_tokens = itertools.count(1)


class FakeReq:
    def __init__(
        self,
        rid,
        prompt_len,
        arrival=100.0,
        cached=0,
        arrival_tokens=0,
        shared_prefix=False,
    ):
        self.rid = rid
        # Prompts differ from their first token unless they share a prefix.
        first = 0 if shared_prefix else next(_first_tokens)
        self.origin_input_ids = [first] + list(range(1, prompt_len))
        self.output_ids = []
        self.num_matched_prefix_tokens = cached
        self.prefix_len = cached
        self.time_stats = FakeTimeStats(arrival)
        self.arrival_processed_tokens = arrival_tokens
        self.cache_request_handle = None
        self.last_node = None
        self.uniboost_first_seen_step = None
        self.finished_reason = None
        self.extra_key = None
        self.cache_salt = None
        self.priority = 0


class FakeTreeCache:
    def __init__(self, disable=True):
        self.disable = disable

    def supports_fast_match_prefix(self):
        return False

    def supports_prefix_sharing(self):
        return False

    def staged_storage_prefix_end(self, handle):
        return 0


def make_policy(policy_name="uniboost", tree_cache=None, **kwargs):
    return SchedulePolicy(
        policy_name,
        tree_cache if tree_cache is not None else FakeTreeCache(),
        enable_hierarchical_cache=False,
        enable_priority_scheduling=False,
        schedule_low_priority_values_first=False,
        **kwargs,
    )


def sorted_rids(policy, queue):
    return [r.rid for r in policy.calc_priority(queue)]


def finish(ctl, latencies, now=1000.0):
    """Record one finished request per wall-clock latency."""
    for latency in latencies:
        req = FakeReq("r", 1, arrival=now - latency)
        req.finished_reason = FINISH_LENGTH(length=1)
        with patch.object(uniboost_gamma_ada.time, "perf_counter", return_value=now):
            ctl.record_finish(req)


class UniboostTestBase(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


class TestGammaAdaController(CustomTestCase):
    def test_no_refit_until_enough_latencies(self):
        ctl = GammaAdaController(50.0, step_clock=None)
        finish(ctl, [1.0] * (GAMMA_ADA_REFIT_EVERY - 1))
        self.assertEqual((ctl.refit_count, ctl.gamma), (0, 50.0))
        finish(ctl, [1.0])
        self.assertEqual(ctl.refit_count, 1)

    def test_heavy_tail_lowers_gamma_and_flat_tail_raises_it(self):
        heavy = GammaAdaController(100.0, step_clock=None)
        # 6 of 256 latencies (2%) sit between p95 and p99.
        finish(heavy, [1.0] * 250 + [50.0] * 6)
        self.assertLess(heavy.gamma, 100.0)
        flat = GammaAdaController(10.0, step_clock=None)
        finish(flat, [1.0] * GAMMA_ADA_REFIT_EVERY)
        self.assertGreater(flat.gamma, 10.0)
        self.assertLessEqual(flat.gamma, GAMMA_ADA_MAX)

    def test_step_clock_counts_from_first_sight(self):
        step = [10]
        ctl = GammaAdaController(10.0, step_clock=lambda: step[0])
        req = FakeReq("r", 1)
        ctl.stamp_first_seen([req])
        step[0] += 3
        ctl.stamp_first_seen([req])  # a later pass, or after a retraction
        self.assertEqual(req.uniboost_first_seen_step, 10)
        step[0] += 4
        req.finished_reason = FINISH_LENGTH(length=1)
        ctl.record_finish(req)
        self.assertEqual(list(ctl._samples), [7.0])

    def test_wall_clock_latency_starts_at_scheduler_recv(self):
        ctl = GammaAdaController(10.0, step_clock=None)
        req = FakeReq("r", 1, arrival=10.0)
        req.time_stats.wait_queue_entry_time = 40.0  # re-stamped by a retraction
        req.finished_reason = FINISH_LENGTH(length=1)
        with patch.object(uniboost_gamma_ada.time, "perf_counter", return_value=50.0):
            ctl.record_finish(req)
        self.assertEqual(list(ctl._samples), [40.0])

    def test_aborted_and_unfinished_requests_are_skipped(self):
        ctl = GammaAdaController(10.0, step_clock=None)
        aborted = FakeReq("a", 1)
        aborted.finished_reason = FINISH_ABORT()
        ctl.record_finish(aborted)
        ctl.record_finish(FakeReq("u", 1))
        self.assertEqual(len(ctl._samples), 0)


class TestUniboostOrdering(UniboostTestBase):
    def test_short_job_jumps_ahead_at_same_arrival(self):
        q = [FakeReq("long", 20000), FakeReq("short", 100)]
        self.assertEqual(sorted_rids(make_policy(), q), ["short", "long"])

    def test_equal_work_keeps_fcfs(self):
        q = [FakeReq("second", 5000, arrival=100.5), FakeReq("first", 5000)]
        self.assertEqual(sorted_rids(make_policy(), q), ["first", "second"])

    def test_bounded_boost_cannot_starve_an_old_request(self):
        q = [FakeReq("new_short", 100), FakeReq("old_long", 20000, arrival=-900.0)]
        self.assertEqual(sorted_rids(make_policy(), q), ["old_long", "new_short"])

    def test_cached_prefix_counts_as_no_work(self):
        q = [FakeReq("cold", 20000), FakeReq("hot", 20000, cached=19900)]
        self.assertEqual(sorted_rids(make_policy(), q), ["hot", "cold"])

    def test_rank_invariant_arrival_ignores_wall_time(self):
        # Two ranks see the same requests with different local wall stamps.
        def queue(wall_offset):
            return [
                FakeReq("b", 300, arrival=wall_offset + 1.0, arrival_tokens=40000),
                FakeReq("a", 3000, arrival=wall_offset + 2.0, arrival_tokens=0),
            ]

        policy = make_policy(uniboost_rank_invariant=True)
        self.assertEqual(sorted_rids(policy, queue(0.0)), ["a", "b"])
        self.assertEqual(sorted_rids(policy, queue(5.0)), ["a", "b"])
        # The wall clock would admit "b" first.
        self.assertEqual(sorted_rids(make_policy(), queue(0.0)), ["b", "a"])


class TestUniboostConfig(UniboostTestBase):
    def test_disabled_tree_cache_keeps_uniboost(self):
        self.assertEqual(make_policy().policy, CacheAwarePolicy.UNIBOOST)
        self.assertEqual(make_policy("lpm").policy, CacheAgnosticPolicy.FCFS)

    def test_invalid_config_is_rejected(self):
        for gamma in (0.0, float("inf"), 1e-320, 1e-310):
            with self.assertRaises(ValueError):
                make_policy(uniboost_gamma=gamma)
        with self.assertRaises(ValueError):
            make_policy("fcfs", enable_uniboost_gamma_ada=True)

    def test_gamma_ada_clock_follows_rank_invariance(self):
        def clock():
            return 0

        policy = make_policy(enable_uniboost_gamma_ada=True, uniboost_step_clock=clock)
        self.assertIsNone(policy.uniboost_gamma_ada.step_clock)
        policy = make_policy(
            enable_uniboost_gamma_ada=True,
            uniboost_rank_invariant=True,
            uniboost_step_clock=clock,
        )
        self.assertIs(policy.uniboost_gamma_ada.step_clock, clock)
        self.assertIsNone(make_policy().uniboost_gamma_ada)


class TestUniboostMatching(UniboostTestBase):
    def run_pass(self, policy, queue):
        matched = []

        def fake_match(req, tree_cache, token_ids=None, probe=False):
            matched.append(req.rid)
            req.last_node = object()

        with patch.object(schedule_policy, "match_kv_cache", side_effect=fake_match):
            queue[:] = policy.calc_priority(queue)
        return matched

    def test_short_queue_is_matched_every_pass(self):
        policy = make_policy(tree_cache=FakeTreeCache(disable=False))
        queue = [FakeReq(f"r{i}", 1000, arrival=100.0 + i) for i in range(8)]
        self.assertEqual(len(self.run_pass(policy, queue)), 8)
        self.assertEqual(len(self.run_pass(policy, queue)), 8)

    def test_deep_queue_matches_each_request_once_then_the_window(self):
        policy = make_policy(tree_cache=FakeTreeCache(disable=False))
        n = UNIBOOST_MATCH_WINDOW_SIZE + 30
        queue = [FakeReq(f"r{i}", 1000, arrival=100.0 + i) for i in range(n)]
        first = self.run_pass(policy, queue)
        self.assertEqual(set(first), {r.rid for r in queue})
        second = self.run_pass(policy, queue)
        window = {r.rid for r in queue[:UNIBOOST_MATCH_WINDOW_SIZE]}
        self.assertEqual(
            (len(second), set(second)), (UNIBOOST_MATCH_WINDOW_SIZE, window)
        )
        # A retraction clears last_node, so the next pass matches it again.
        queue[-1].last_node = None
        self.assertIn(queue[-1].rid, self.run_pass(policy, queue))

    def test_shared_uncached_prefix_runs_once_led_by_its_best_member(self):
        # "long" and "short" share an uncached prefix: "short" leads it, and
        # "long" waits at the end of the whole queue until the prefix is cached.
        policy = make_policy(tree_cache=FakeTreeCache(disable=False))
        others = [
            FakeReq(f"r{i}", 1000, arrival=101.0 + i)
            for i in range(UNIBOOST_MATCH_WINDOW_SIZE)
        ]
        queue = [
            FakeReq("long", 20000, shared_prefix=True),
            FakeReq("short", 1050, shared_prefix=True),
        ] + others
        self.run_pass(policy, queue)
        self.assertEqual(queue[0].rid, "short")
        self.assertEqual(queue[-1].rid, "long")

    def test_disabled_tree_cache_never_matches(self):
        policy = make_policy()
        queue = [FakeReq(f"r{i}", 1000) for i in range(UNIBOOST_MATCH_WINDOW_SIZE + 5)]
        self.assertEqual(self.run_pass(policy, queue), [])


class TestUniboostChunkLimit(UniboostTestBase):
    def chunk_limit(self, policy, waiting_queue):
        chunked = FakeReq("chunked", 0)
        chunked.full_untruncated_fill_ids = list(range(30000))
        chunked.prefix_len = 0
        with patch.object(PrefillAdder, "_replay_len", return_value=0):
            return policy.shortest_prefill_chunk_limit(
                chunked, waiting_queue, 8192, 16, processed_tokens=0
            )

    def test_reserves_budget_for_requests_ranked_ahead(self):
        self.assertEqual(
            self.chunk_limit(make_policy(), [FakeReq("short", 1000)]), 8192 - 1008
        )


if __name__ == "__main__":
    unittest.main()
