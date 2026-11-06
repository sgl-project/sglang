import math
import unittest
from array import array
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.managers.uniboost import (
    TokenClockRate,
    UniBoostClock,
    boost,
    geometric_bin,
    sort_waiting_queue,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_req(rid, ids):
    return Req(rid, "", array("q", ids), SamplingParams())


def _req(rid, uncached, *, arrival=0, cached=0, output=0, entry_s=None):
    """Req with u(r) = uncached; arrival on the token clock, entry_s on wall."""
    req = _make_req(rid, list(range(uncached + cached)))
    req.num_matched_prefix_tokens = cached
    if output:
        req.output_ids.extend([0] * output)
    req.uniboost_arrival_tokens = arrival
    req.uniboost_arrival_s = entry_s
    return req


def _sort(
    reqs,
    *,
    gamma=1e-4,
    bin_size=256,
    clock=UniBoostClock.TOKENS,
    rate=None,
    deprioritized=(),
    priority=False,
):
    queue = list(reqs)
    sort_waiting_queue(
        queue,
        gamma=gamma,
        bin_size=bin_size,
        clock=clock,
        rate_tokens_per_s=rate,
        temporary_deprioritized=set(deprioritized),
        enable_priority_scheduling=priority,
        priority_sign=-1,
    )
    return [r.rid for r in queue]


def _policy(policy="uniboost", tree_cache=None, **kwargs):
    return SchedulePolicy(
        policy=policy,
        tree_cache=tree_cache or RadixCache.create_simulated(),
        enable_hierarchical_cache=True,
        enable_priority_scheduling=False,
        schedule_low_priority_values_first=False,
        **kwargs,
    )


class TestBoostFunction(CustomTestCase):
    """b_gamma(w) = -ln(1 - exp(-gamma * w)) / gamma (paper Eq. 2)."""

    def test_nonincreasing_in_work_and_strict_while_positive(self):
        for gamma in (1e-5, 1e-4, 1e-3, 1.0):
            boosts = [boost(2.0**i, gamma) for i in range(30)]
            for larger, smaller in zip(boosts, boosts[1:]):
                self.assertGreaterEqual(larger, smaller)
                if larger > 0.0:
                    self.assertGreater(larger, smaller)

    def test_finite_and_nonnegative_across_the_domain(self):
        # gamma * work past ~40 rounds 1 - exp(-x) to 1.0: the underflow branch.
        for gamma in (1e-8, 1e-5, 1e-4, 1e-2, 1.0):
            for work in (1e-4, 1e-1, 1.0, 10.0, 1e3, 1e6, 8e6, 1e9):
                value = boost(work, gamma)
                self.assertTrue(math.isfinite(value), (gamma, work, value))
                self.assertGreaterEqual(value, 0.0)
        self.assertEqual(boost(8_000_000.0, 1e-4), 0.0)

    def test_nonpositive_argument_is_infinite(self):
        self.assertEqual(boost(0.0, 1e-4), float("inf"))
        self.assertEqual(boost(5.0, -1.0), float("inf"))

    def test_quantized_boost_is_capped_by_the_bin_size(self):
        # q(u) >= k and b is decreasing, so b(q(u)) <= b(k), equal for u <= k.
        k, gamma = 256, 1e-4
        ceiling = boost(k, gamma)
        for u in (1, k, k + 1, 2 * k, 3 * k, 100_000, 262_144):
            self.assertLessEqual(boost(geometric_bin(u, k), gamma), ceiling)
        self.assertEqual(boost(geometric_bin(1, k), gamma), ceiling)


class TestGeometricBin(CustomTestCase):
    """q(u) = k * 2^floor(log2(max(u, k) / k)) (paper Eq. 8)."""

    def test_ladder_values(self):
        cases = {0: 256, 1: 256, 256: 256, 511: 256, 512: 512, 1025: 1024}
        cases.update({100_000: 65_536, 250_000: 131_072})
        for u, expected in cases.items():
            self.assertEqual(geometric_bin(u, 256), expected, u)
        cases = {1: 100, 199: 100, 200: 200, 399: 200, 400: 400}
        for u, expected in cases.items():
            self.assertEqual(geometric_bin(u, 100), expected, u)

    def test_every_output_is_on_the_ladder(self):
        for k in (1, 3, 256, 1000):
            ladder = {k * 2**j for j in range(20)}
            for u in list(range(4 * k + 3)) + [100_000, 262_144]:
                self.assertIn(geometric_bin(u, k), ladder, (k, u))


class TestTokenClockSort(CustomTestCase):
    """phi(r) = A(r) - b_gamma(q(u(r))) on the token clock, ascending."""

    def test_small_work_newcomer_overtakes_older_large_work(self):
        # gamma 1e-4, k 256: b(256) = 36779 tokens vs b(65536) = 14.3, so a
        # newcomer 500 tokens later still sorts ahead.
        cold = _req("cold", 100_000, arrival=0)
        hot = _req("hot", 256, arrival=500)
        self.assertEqual(_sort([cold, hot]), ["hot", "cold"])

    def test_starvation_bound(self):
        # A newcomer overtakes the large request iff it arrived less than
        # b(k) - b(q(u_cold)) clock tokens later; equal-work newcomers stay FCFS.
        for gamma in (5e-5, 1e-4, 2e-4):
            for u_cold in (100_000, 250_000):
                with self.subTest(gamma=gamma, u_cold=u_cold):
                    bound = boost(256, gamma) - boost(geometric_bin(u_cold, 256), gamma)
                    lags = [int(bound * f) for f in (0.0, 0.5, 0.9, 1.1, 2.0, 10.0)]
                    reqs = [_req("cold", u_cold)]
                    reqs += [_req(f"hot{a}", 256, arrival=a) for a in lags]
                    order = _sort(reqs, gamma=gamma)
                    at = order.index("cold")
                    self.assertEqual(order[:at], [f"hot{a}" for a in lags if a < bound])
                    self.assertEqual(
                        order[at + 1 :], [f"hot{a}" for a in lags if a > bound]
                    )

    def test_large_gamma_is_fcfs_and_small_gamma_is_shortest_work(self):
        reqs = [
            _req("new_short", 256, arrival=900),
            _req("old_huge", 250_000, arrival=100),
            _req("mid", 5_000, arrival=500),
        ]
        self.assertEqual(_sort(reqs, gamma=1.0), ["old_huge", "mid", "new_short"])
        self.assertEqual(_sort(reqs, gamma=1e-9), ["new_short", "mid", "old_huge"])

    def test_retracted_output_is_charged_as_work(self):
        # Generated tokens are re-prefilled after a retraction.
        replay = _req("replay", 100, output=900)
        fresh = _req("fresh", 500)
        self.assertEqual(_sort([replay, fresh]), ["fresh", "replay"])

    def test_dedup_follower_sorts_last(self):
        follower = _req("follower", 256, arrival=0)
        other = _req("other", 250_000, arrival=1_000_000)
        order = _sort([follower, other], deprioritized={"follower"})
        self.assertEqual(order, ["other", "follower"])

    def test_priority_is_the_primary_key(self):
        low = _req("low", 256, arrival=0)
        low.priority = 1
        high = _req("high", 250_000, arrival=10**9)
        high.priority = 5
        self.assertEqual(_sort([low, high], priority=True), ["high", "low"])

    def test_retraction_keeps_the_arrival_stamp(self):
        req = _req("retracted", 100, arrival=1234, entry_s=5.0)
        req.reset_for_retract()
        self.assertEqual(req.uniboost_arrival_tokens, 1234)
        self.assertEqual(req.uniboost_arrival_s, 5.0)


class TestWallClockSort(CustomTestCase):
    """Wall clock: phi = A_s - b(q(u) / rate, gamma_s), in seconds."""

    def test_unmeasured_rate_is_pure_arrival_order(self):
        reqs = [
            _req("new_short", 256, entry_s=900.0),
            _req("old_huge", 250_000, entry_s=100.0),
            _req("mid", 5_000, entry_s=500.0),
        ]
        order = _sort(reqs, gamma=0.1, clock=UniBoostClock.WALL, rate=None)
        self.assertEqual(order, ["old_huge", "mid", "new_short"])

    def test_starvation_bound_in_seconds(self):
        k, rate = 256, 36_700.0
        for gamma_s in (0.02, 0.1, 0.5):
            with self.subTest(gamma_s=gamma_s):
                bound_s = boost(k / rate, gamma_s) - boost(
                    geometric_bin(100_000, k) / rate, gamma_s
                )
                lags = [bound_s * f for f in (0.5, 0.9, 1.1, 2.0)]
                reqs = [_req("cold", 100_000, entry_s=0.0)]
                reqs += [
                    _req(f"hot{i}", 256, entry_s=lag) for i, lag in enumerate(lags)
                ]
                order = _sort(reqs, gamma=gamma_s, clock=UniBoostClock.WALL, rate=rate)
                self.assertEqual(order, ["hot0", "hot1", "cold", "hot2", "hot3"])

    def test_wall_matches_tokens_at_gamma_scaled_by_rate(self):
        # Both axes scale by 1/rate at gamma_s = gamma_t * rate, so the two
        # clocks must order the same requests identically.
        rate, gamma_t = 36_700.0, 1e-4
        spec = [("cold", 100_000, 0.0), ("hot1", 256, 0.0005), ("mid", 5_000, 0.002)]
        wall = _sort(
            [_req(rid, u, entry_s=e) for rid, u, e in spec],
            gamma=gamma_t * rate,
            clock=UniBoostClock.WALL,
            rate=rate,
        )
        tokens = _sort(
            [_req(rid, u, arrival=int(e * rate)) for rid, u, e in spec], gamma=gamma_t
        )
        self.assertEqual(wall, tokens)
        self.assertEqual(wall, ["hot1", "mid", "cold"])


class TestTokenClockRate(CustomTestCase):
    def test_first_interval_sets_the_rate_exactly(self):
        rate = TokenClockRate()
        self.assertIsNone(rate.observe(0, 0.0))
        self.assertEqual(rate.observe(30_000, 1.0), 30_000.0)

    def test_idle_stretch_folds_into_the_next_sample(self):
        # No sample and no baseline move on zero delta: the rate stays
        # load-inclusive across decode-only stretches.
        rate = TokenClockRate()
        rate.observe(0, 0.0)
        self.assertIsNone(rate.observe(0, 10.0))
        self.assertAlmostEqual(rate.observe(100, 11.0), 100.0 / 11.0)

    def test_counter_regression_is_ignored(self):
        rate = TokenClockRate()
        rate.observe(30_000, 0.0)
        self.assertIsNone(rate.observe(0, 1.0))
        self.assertAlmostEqual(rate.observe(40_000, 3.0), 10_000.0 / 3.0)

    def test_ema_follows_a_rate_step(self):
        rate = TokenClockRate()
        rate.observe(0, 0.0)
        tokens, now = 0, 0.0
        for step in (5_000, 36_700):
            for _ in range(200):
                tokens, now = tokens + step, now + 1.0
                rate.observe(tokens, now)
        self.assertAlmostEqual(rate.tokens_per_s, 36_700.0, delta=1.0)


class TestSchedulePolicyUniBoost(CustomTestCase):
    def test_calc_priority_stamps_once_and_sorts(self):
        cold = _make_req("cold", list(range(100_000)))
        policy = _policy(uniboost_gamma=1e-4)
        policy.calc_priority([cold], processed_tokens=0)
        hot = _make_req("hot", list(range(500_000, 500_256)))
        queue = [cold, hot]
        queue = policy.calc_priority(queue, processed_tokens=500)
        self.assertEqual(cold.uniboost_arrival_tokens, 0)
        self.assertEqual(hot.uniboost_arrival_tokens, 500)
        self.assertEqual([r.rid for r in queue], ["hot", "cold"])

    def test_wall_clock_measures_the_rate_from_processed_tokens(self):
        # Disjoint token ids: a shared prefix would make hot a dedup follower.
        cold = _make_req("cold", list(range(100_000)))
        hot = _make_req("hot", list(range(500_000, 500_256)))
        cold.uniboost_arrival_tokens, cold.uniboost_arrival_s = 0, 0.0
        hot.uniboost_arrival_tokens, hot.uniboost_arrival_s = 0, 100.0
        queue = [cold, hot]
        policy = _policy(uniboost_gamma=0.01, uniboost_clock="wall")
        # Replaces only schedule_policy's binding of the time module.
        with patch("sglang.srt.managers.schedule_policy.time") as clock:
            clock.perf_counter.side_effect = [0.0, 1.0]
            queue = policy.calc_priority(queue, processed_tokens=0)
            self.assertEqual([r.rid for r in queue], ["cold", "hot"])
            queue = policy.calc_priority(queue, processed_tokens=30_000)
        self.assertEqual(policy.uniboost_rate.tokens_per_s, 30_000.0)
        # At 30k tok/s and gamma 0.01 1/s hot's boost (~937 s) beats its 100 s lag.
        self.assertEqual([r.rid for r in queue], ["hot", "cold"])

    def test_unset_gamma_takes_the_clock_default(self):
        self.assertEqual(_policy().uniboost_gamma, 1e-4)
        self.assertEqual(_policy(uniboost_clock="wall").uniboost_gamma, 0.1)
        self.assertEqual(
            _policy(uniboost_clock="wall", uniboost_gamma=0.05).uniboost_gamma, 0.05
        )

    def test_large_queue_keeps_uniboost(self):
        queue = [_make_req(f"r{i}", [i]) for i in range(200)]
        policy = _policy()
        self.assertIs(policy._determine_active_policy(queue), CacheAwarePolicy.UNIBOOST)

    def test_without_prefix_cache_falls_back_to_fcfs(self):
        with self.assertLogs("sglang.srt.managers.schedule_policy", level="WARNING"):
            policy = _policy(tree_cache=RadixCache.create_simulated(disable=True))
        self.assertIs(policy.policy, CacheAgnosticPolicy.FCFS)


if __name__ == "__main__":
    unittest.main()
