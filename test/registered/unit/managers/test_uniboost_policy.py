"""Unit tests for the UniBoost schedule policy and its gamma-Ada controller.

No GPU or server launch: exercises the real SchedulePolicy / GammaAdaController
with mock Req objects.

Coverage (each case guards a distinct failure mode):
  * GammaAdaController control law -- heavy tail drives gamma down (more SJF),
    light tail drives it up (more FCFS), always clamped into
    [gamma_min, gamma_max] (derived property of the target formula);
  * the async refit thread -- refits happen on the daemon thread, start() is
    idempotent, no refit below min_samples, interval_s=0 does not busy-loop;
  * uniboost waiting-queue ordering -- SJF pull-forward, FCFS tie-break,
    bounded boost (anti-starvation), cache-aware effective work;
  * fallback gating -- tp_size > 1 without SGLANG_UNIBOOST_ALLOW_TP,
    gamma <= 0, and the tree-cache-disabled carve-out;
  * the finished-request recording hook -- aborts excluded, each request
    recorded once, and calc_priority never refits synchronously.
"""

import random
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    GammaAdaController,
    SchedulePolicy,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class FakeTimeStats:
    def __init__(self, wait_queue_entry_time=0.0, completion_time=0.0):
        self.wait_queue_entry_time = wait_queue_entry_time
        self.completion_time = completion_time


class FakeReq:
    def __init__(
        self,
        rid,
        prompt_len,
        arrival,
        cached=0,
        finished=False,
        finished_reason=None,
        completion_time=0.0,
    ):
        self.rid = rid
        self.origin_input_ids = list(range(prompt_len))
        self.output_ids = []
        self.num_matched_prefix_tokens = cached
        self.uniboost_latency_recorded = False
        self.time_stats = FakeTimeStats(arrival, completion_time)
        self._finished = finished
        self.finished_reason = finished_reason
        self.extra_key = None
        self.priority = 0

    def finished(self):
        return self._finished


class FakeDisabledTreeCache:
    disable = True

    def supports_fast_match_prefix(self):
        return False


def make_policy(policy_name="uniboost"):
    return SchedulePolicy(
        policy_name,
        FakeDisabledTreeCache(),
        enable_hierarchical_cache=False,
        enable_priority_scheduling=False,
        schedule_low_priority_values_first=False,
    )


def drive(ctl, ticks=30):
    for k in range(ticks):
        ctl.maybe_update(now=k + 1)


def wait_until(cond, timeout=5.0, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(step)
    return cond()


class UniboostTestBase(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # SchedulePolicy._init_uniboost reads tp_size off the published args.
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


class TestGammaAdaController(CustomTestCase):
    def test_no_update_before_interval(self):
        ctl = GammaAdaController(gamma_init=50.0, interval_s=5.0, min_samples=100)
        for _ in range(200):
            ctl.record(1.0)
        self.assertIsNone(ctl.maybe_update(now=1.0))
        self.assertIsNotNone(ctl.maybe_update(now=10.0))

    def test_no_update_below_min_samples(self):
        ctl = GammaAdaController(gamma_init=50.0, interval_s=0.0, min_samples=200)
        for _ in range(199):
            ctl.record(1.0)
        self.assertIsNone(ctl.maybe_update(now=1.0))

    def test_heavy_tail_lowers_gamma(self):
        ctl = GammaAdaController(
            gamma_init=100.0,
            gamma_min=1.0,
            gamma_max=200.0,
            interval_s=0.0,
            min_samples=100,
        )
        rng = random.Random(0)
        for i in range(1000):
            ctl.record(1.0 + (100.0 if i % 33 == 0 else rng.uniform(0, 0.05)))
        drive(ctl)
        self.assertLess(ctl.gamma, 20.0)
        self.assertGreaterEqual(ctl.gamma, ctl.gamma_min)

    def test_light_tail_raises_gamma(self):
        ctl = GammaAdaController(
            gamma_init=20.0,
            gamma_min=1.0,
            gamma_max=200.0,
            interval_s=0.0,
            min_samples=100,
        )
        rng = random.Random(1)
        for _ in range(1000):
            ctl.record(1.0 + rng.uniform(0, 0.02))
        drive(ctl)
        self.assertGreater(ctl.gamma, 150.0)
        self.assertLessEqual(ctl.gamma, ctl.gamma_max)

    def test_gamma_stays_positive_and_bounded(self):
        ctl = GammaAdaController(
            gamma_init=1.0,
            gamma_min=1.0,
            gamma_max=200.0,
            interval_s=0.0,
            min_samples=10,
        )
        for _ in range(100):
            ctl.record(1e6)
        for k in range(50):
            g = ctl.maybe_update(now=k + 1)
            if g is not None:
                self.assertGreater(g, 0.0)
                self.assertGreaterEqual(g, ctl.gamma_min)
                self.assertLessEqual(g, ctl.gamma_max)

    def test_gamma_min_clamped_strictly_positive(self):
        # b_gamma divides by gamma, so a 0 gamma_min must be clamped away.
        ctl = GammaAdaController(gamma_init=0.0, gamma_min=0.0, gamma_max=10.0)
        self.assertGreater(ctl.gamma_min, 0.0)
        self.assertGreater(ctl.gamma, 0.0)


class TestGammaAdaAsyncController(CustomTestCase):
    def test_refit_runs_on_daemon_thread(self):
        ctl = GammaAdaController(
            gamma_init=100.0,
            gamma_min=1.0,
            gamma_max=200.0,
            interval_s=0.05,
            min_samples=50,
            window=1000,
        )
        self.addCleanup(ctl.stop)
        rng = random.Random(2)
        for i in range(500):
            ctl.record(1.0 + (100.0 if i % 25 == 0 else rng.uniform(0, 0.05)))
        before = threading.active_count()
        ctl.start()
        t1 = ctl._thread
        self.assertIsNotNone(t1)
        self.assertTrue(t1.daemon)
        self.assertTrue(t1.is_alive())
        # Idempotent start: no second thread.
        ctl.start()
        self.assertIs(ctl._thread, t1)
        self.assertEqual(threading.active_count(), before + 1)
        self.assertTrue(wait_until(lambda: ctl.refit_count >= 3), ctl.refit_count)
        # Heavy tail: gamma driven down, always within bounds.
        self.assertLess(ctl.gamma, 100.0)
        self.assertGreaterEqual(ctl.gamma, ctl.gamma_min)
        self.assertLessEqual(ctl.gamma, ctl.gamma_max)
        # Keep recording while the thread refits (lock-free append vs scan).
        for _ in range(2000):
            ctl.record(1.0 + rng.uniform(0, 0.05))
        self.assertTrue(wait_until(lambda: ctl.refit_count >= 6))
        self.assertGreaterEqual(ctl.gamma, ctl.gamma_min)
        self.assertLessEqual(ctl.gamma, ctl.gamma_max)
        ctl.stop()
        self.assertFalse(t1.is_alive())

    def test_thread_never_refits_without_samples(self):
        ctl = GammaAdaController(gamma_init=42.0, interval_s=0.02, min_samples=10)
        self.addCleanup(ctl.stop)
        ctl.start()
        time.sleep(0.3)
        self.assertEqual(ctl.refit_count, 0)
        self.assertEqual(ctl.gamma, 42.0)

    def test_zero_interval_does_not_busy_loop(self):
        # interval_s=0 must fall back to a short sleep cadence, still refit,
        # and stay bounded.
        ctl = GammaAdaController(gamma_init=50.0, interval_s=0.0, min_samples=5)
        self.addCleanup(ctl.stop)
        for _ in range(50):
            ctl.record(1.0)
        ctl.start()
        self.assertTrue(wait_until(lambda: ctl.refit_count >= 2))
        self.assertGreaterEqual(ctl.gamma, ctl.gamma_min)
        self.assertLessEqual(ctl.gamma, ctl.gamma_max)


class TestUniboostOrdering(UniboostTestBase):
    def setUp(self):
        self.pol = make_policy()
        self.assertEqual(self.pol.policy, CacheAwarePolicy.UNIBOOST)
        self.assertEqual(self.pol.uniboost_gamma, 10.0)
        # Note: 0.0 arrival means "unset" in sglang time stats (the sort falls
        # back to the current clock), so keep mock arrivals strictly positive.
        self.t0 = 10000.0

    def rids_after_sort(self, q):
        self.pol.calc_priority(q)
        return [r.rid for r in q]

    def test_short_job_jumps_ahead_at_same_arrival(self):
        q = [FakeReq("long", 20000, self.t0), FakeReq("short", 100, self.t0)]
        self.assertEqual(self.rids_after_sort(q), ["short", "long"])

    def test_equal_work_preserves_fcfs(self):
        a = FakeReq("first", 5000, self.t0)
        b = FakeReq("second", 5000, self.t0 + 0.5)
        self.assertEqual(self.rids_after_sort([b, a]), ["first", "second"])

    def test_anti_starvation(self):
        # A much older long job stays ahead of a fresh short job: the boost is
        # bounded, so age eventually dominates.
        old_long = FakeReq("old_long", 20000, self.t0 - 1000.0)
        new_short = FakeReq("new_short", 100, self.t0)
        self.assertEqual(
            self.rids_after_sort([new_short, old_long]), ["old_long", "new_short"]
        )

    def test_cache_aware_effective_work(self):
        # A mostly-cached long prompt counts as short work.
        cold = FakeReq("cold", 20000, self.t0)
        hot = FakeReq("hot", 20000, self.t0, cached=19900)
        self.assertEqual(self.rids_after_sort([cold, hot]), ["hot", "cold"])

    def test_boost_value_decreasing_and_bounded(self):
        b1 = self.pol._uniboost_boost_value(10)
        b2 = self.pol._uniboost_boost_value(1000)
        b3 = self.pol._uniboost_boost_value(100000)
        self.assertGreater(b1, b2)
        self.assertGreater(b2, b3)
        self.assertGreaterEqual(b3, 0.0)
        self.assertEqual(self.pol._uniboost_boost_value(0), 1e9)


class TestUniboostFallbacks(UniboostTestBase):
    def test_fcfs_policy_unaffected(self):
        pol = make_policy("fcfs")
        self.assertEqual(pol.policy, CacheAgnosticPolicy.FCFS)
        a = FakeReq("a", 10, 1.0)
        b = FakeReq("b", 10, 2.0)
        q = [a, b]
        pol.calc_priority(q)
        self.assertEqual([r.rid for r in q], ["a", "b"])

    def test_tp_gating(self):
        fake_args = SimpleNamespace(tp_size=2)
        with patch(
            "sglang.srt.managers.schedule_policy.get_global_server_args",
            return_value=fake_args,
        ):
            pol = make_policy()
            self.assertEqual(pol.policy, CacheAgnosticPolicy.FCFS)

            with envs.SGLANG_UNIBOOST_ALLOW_TP.override(True):
                pol = make_policy()
                self.assertEqual(pol.policy, CacheAwarePolicy.UNIBOOST)

    def test_gamma_zero_falls_back_to_fcfs(self):
        with envs.SGLANG_UNIBOOST_GAMMA.override(0.0):
            pol = make_policy()
            self.assertEqual(pol.policy, CacheAgnosticPolicy.FCFS)

    def test_disabled_tree_cache_keeps_uniboost(self):
        # The tree-cache-disabled carve-out: uniboost still runs (all prompt
        # tokens count as uncached), unlike lpm/dfs-weight which fall to FCFS.
        self.assertEqual(make_policy().policy, CacheAwarePolicy.UNIBOOST)
        self.assertEqual(make_policy("lpm").policy, CacheAgnosticPolicy.FCFS)


class TestGammaAdaWiring(UniboostTestBase):
    def make_ada_policy(self):
        with envs.SGLANG_UNIBOOST_GAMMA_ADA.override(True):
            pol = make_policy()
        self.assertIsNotNone(pol.uniboost_gamma_controller)
        self.addCleanup(pol.uniboost_gamma_controller.stop)
        return pol

    def test_calc_priority_only_reads_gamma(self):
        pol = self.make_ada_policy()
        ctl = pol.uniboost_gamma_controller
        # The controller thread was started by _init_uniboost.
        self.assertIsNotNone(ctl._thread)
        self.assertTrue(ctl._thread.is_alive())
        self.assertTrue(ctl._thread.daemon)

        # calc_priority must not refit: it only reads controller.gamma.
        ctl.gamma = 123.0  # pretend the refit thread updated it
        q = [FakeReq("a", 100, 10.0), FakeReq("b", 200, 10.0)]
        n_refits = ctl.refit_count
        pol.calc_priority(q)
        self.assertEqual(pol.uniboost_gamma, 123.0)
        self.assertEqual(ctl.refit_count, n_refits)

    def test_recording_hook(self):
        pol = self.make_ada_policy()
        ctl = pol.uniboost_gamma_controller

        done = FakeReq("done", 10, 5.0, finished=True, completion_time=8.5)
        aborted = FakeReq(
            "aborted",
            10,
            5.0,
            finished=True,
            finished_reason=FINISH_ABORT(),
            completion_time=6.0,
        )
        pending = FakeReq("pending", 10, 5.0, finished=False)

        pol.uniboost_record_finished([done, aborted, pending])
        self.assertEqual(list(ctl._samples), [3.5])

        # Double-record guard.
        pol.uniboost_record_finished([done])
        self.assertEqual(len(ctl._samples), 1)

    def test_env_knob_defaults_land_on_controller(self):
        ctl = self.make_ada_policy().uniboost_gamma_controller
        self.assertEqual(ctl.gamma_min, 1.0)
        self.assertEqual(ctl.gamma_max, 200.0)
        self.assertEqual(ctl.interval_s, 5.0)
        self.assertEqual(ctl.min_samples, 200)
        self.assertAlmostEqual(ctl.beta, 0.3)
        self.assertEqual(ctl.tail_sensitivity, 4.0)

    def test_recording_noop_without_gamma_ada(self):
        # Without gamma-Ada the scheduler hook must be a clean no-op.
        pol = make_policy()
        self.assertIsNone(pol.uniboost_gamma_controller)
        done = FakeReq("done", 10, 5.0, finished=True, completion_time=8.5)
        pol.uniboost_record_finished([done])  # must not raise
        self.assertFalse(done.uniboost_latency_recorded)


if __name__ == "__main__":
    unittest.main()
