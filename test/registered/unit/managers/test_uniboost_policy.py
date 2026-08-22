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
  * fallback gating -- gamma <= 0, the tree-cache-disabled carve-out, and the
    tp_size > 1 mode selection (rank-invariant by default,
    SGLANG_UNIBOOST_FORCE_FCFS_TP escape hatch, deprecated no-op ALLOW_TP);
  * the finished-request recording hook -- aborts excluded, each request
    recorded once, and calc_priority never refits synchronously;
  * frontier-K matching -- per pass, radix prefix matching is bounded to the
    top-K sorted candidates (deep queues keep last-known matches), and <=0
    restores full-queue matching;
  * rank-invariant mode (tp_size > 1) -- two simulated TP ranks fed the
    identical request stream but different per-rank wall clocks and daemon
    timings produce identical queue orderings on every pass and bit-identical
    gamma trajectories under the step-clocked controller, the sort key is
    wall-clock-free with a quantized/tie-broken total order, and the legacy
    tp == 1 wall-clock behavior is unchanged.
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
        self.uniboost_arrival_seq = None
        self.uniboost_enqueue_step = None
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


class FakeEnabledTreeCache(FakeDisabledTreeCache):
    disable = False


def make_policy(policy_name="uniboost", tree_cache=None):
    return SchedulePolicy(
        policy_name,
        tree_cache if tree_cache is not None else FakeDisabledTreeCache(),
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
        # tp_size > 1 now defaults to the rank-invariant uniboost mode (no
        # more silent fcfs fallback); SGLANG_UNIBOOST_FORCE_FCFS_TP is the
        # escape hatch and SGLANG_UNIBOOST_ALLOW_TP a deprecated no-op.
        fake_args = SimpleNamespace(tp_size=2)
        with patch(
            "sglang.srt.managers.schedule_policy.get_global_server_args",
            return_value=fake_args,
        ):
            pol = make_policy()
            self.assertEqual(pol.policy, CacheAwarePolicy.UNIBOOST)
            self.assertTrue(pol.uniboost_tp_invariant)

            with envs.SGLANG_UNIBOOST_FORCE_FCFS_TP.override(True):
                pol = make_policy()
                self.assertEqual(pol.policy, CacheAgnosticPolicy.FCFS)

            with envs.SGLANG_UNIBOOST_ALLOW_TP.override(True):
                pol = make_policy()
                self.assertEqual(pol.policy, CacheAwarePolicy.UNIBOOST)
                self.assertTrue(pol.uniboost_tp_invariant)

        # tp == 1 keeps the validated wall-clock mode.
        pol = make_policy()
        self.assertEqual(pol.policy, CacheAwarePolicy.UNIBOOST)
        self.assertFalse(pol.uniboost_tp_invariant)

    def test_gamma_zero_falls_back_to_fcfs(self):
        with envs.SGLANG_UNIBOOST_GAMMA.override(0.0):
            pol = make_policy()
            self.assertEqual(pol.policy, CacheAgnosticPolicy.FCFS)

    def test_disabled_tree_cache_keeps_uniboost(self):
        # The tree-cache-disabled carve-out: uniboost still runs (all prompt
        # tokens count as uncached), unlike lpm/dfs-weight which fall to FCFS.
        self.assertEqual(make_policy().policy, CacheAwarePolicy.UNIBOOST)
        self.assertEqual(make_policy("lpm").policy, CacheAgnosticPolicy.FCFS)


class TestUniboostFrontierK(UniboostTestBase):
    """Frontier-K: per-pass radix matching is bounded to the top-K candidates.

    Guards the fix for the full-queue-matching scheduler tax (20-40% goodput
    loss at 1000+ queued requests on the hybrid-mamba radix cache in
    real-trace replay): a future diff that re-matches the whole queue per
    pass, or that stops matching the admission frontier, turns these red.
    """

    def queue(self, n):
        # Higher index = later arrival; equal work so FCFS order is stable.
        return [FakeReq(f"r{i}", 1000, 100.0 + i) for i in range(n)]

    def test_matching_bounded_to_top_k(self):
        with envs.SGLANG_UNIBOOST_MATCH_TOPK.override(8):
            pol = make_policy(tree_cache=FakeEnabledTreeCache())
        matched = []
        with patch(
            "sglang.srt.managers.schedule_policy.match_prefix_for_req",
            side_effect=lambda tc, r, **kw: matched.append(r.rid),
        ):
            q = self.queue(50)
            pol.calc_priority(q)
        self.assertEqual(len(matched), 8)
        # The frontier is the head of the sorted queue.
        self.assertEqual(matched, [r.rid for r in q[:8]])

    def test_zero_topk_restores_full_queue_matching(self):
        with envs.SGLANG_UNIBOOST_MATCH_TOPK.override(0):
            pol = make_policy(tree_cache=FakeEnabledTreeCache())
        calls = []
        with patch.object(
            SchedulePolicy,
            "_compute_prefix_matches",
            side_effect=lambda wq, policy: calls.append(len(wq)),
        ):
            q = self.queue(50)
            pol.calc_priority(q)
        self.assertEqual(calls, [50])

    def test_disabled_tree_cache_never_matches(self):
        pol = make_policy()  # disabled cache; topk default
        with patch("sglang.srt.managers.schedule_policy.match_prefix_for_req") as m:
            q = self.queue(20)
            pol.calc_priority(q)
        m.assert_not_called()


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


def make_tp_policy(tree_cache=None):
    """A SchedulePolicy constructed as one rank of a tp_size=2 group."""
    with patch(
        "sglang.srt.managers.schedule_policy.get_global_server_args",
        return_value=SimpleNamespace(tp_size=2),
    ):
        return make_policy(tree_cache=tree_cache)


class RankClock:
    """One rank's simulated ``time.perf_counter``: distinct offset and drift.

    Two RankClocks with different offsets/drifts model TP ranks whose local
    wall clocks disagree -- the failure injection for the rank-invariance
    tests: any code path that reads the patched clock produces different
    values on the two simulated ranks.
    """

    def __init__(self, offset, drift):
        self.now = offset
        self.drift = drift

    def __call__(self):
        self.now += self.drift
        return self.now


class TestUniboostTpInvariant(UniboostTestBase):
    """Rank-invariant UniBoost under tensor parallelism (tp_size > 1).

    Simulates two TP ranks as two SchedulePolicy instances fed the identical
    broadcast request stream, but with injected per-rank differences: each
    rank's FakeReqs carry different wall-clock arrival stamps and each rank's
    passes run under a different monkeypatched ``perf_counter``. Any use of
    scheduler-local wall time in Phi or in gamma-Ada would make these tests
    red by diverging the two ranks' queue orders or gamma trajectories --
    exactly the TP desync (mismatched batch shapes -> NCCL hang) the
    rank-invariant mode exists to prevent.
    """

    # (rid, prompt_len, cached) -- identical on both ranks, like the
    # broadcast request stream. Mixes long, medium, short, and a mostly
    # cached long prompt (cache-aware effective work).
    STREAM = [
        ("L0", 20000, 0),
        ("L1", 20000, 0),
        ("L2", 18000, 0),
        ("M3", 5000, 0),
        ("S4", 100, 0),
        ("L5", 20000, 0),
        ("H6", 20000, 19900),
        ("M7", 4000, 0),
        ("S8", 200, 0),
        ("L9", 17000, 0),
        ("M10", 6000, 0),
        ("S11", 150, 0),
    ]

    def _rank_pass(self, pol, queue, clock):
        """Run one scheduling pass for one rank under its own wall clock."""
        with patch(
            "sglang.srt.managers.schedule_policy.time.perf_counter",
            side_effect=clock,
        ):
            pol.calc_priority(queue)
        return [r.rid for r in queue]

    def test_two_ranks_agree_on_queue_order(self):
        pols = [make_tp_policy(), make_tp_policy()]
        clocks = [RankClock(1_000.0, 0.013), RankClock(50_000.0, 0.007)]
        queues = [[], []]
        for pol in pols:
            self.assertTrue(pol.uniboost_tp_invariant)

        cursor = 0
        orders_seen = []
        for admitted_per_pass in (2, 2, 2):
            wave = self.STREAM[cursor : cursor + 4]
            cursor += 4
            pass_orders = []
            for rank in range(2):
                # Per-rank wall stamps differ; rids/lengths/cache are shared.
                queues[rank].extend(
                    FakeReq(rid, plen, clocks[rank](), cached=cached)
                    for rid, plen, cached in wave
                )
                pass_orders.append(
                    self._rank_pass(pols[rank], queues[rank], clocks[rank])
                )
            # (a) identical queue orderings on every pass.
            self.assertEqual(pass_orders[0], pass_orders[1])
            orders_seen.append(pass_orders[0])
            for rank in range(2):
                # Admit the head of the queue, in lockstep on both ranks.
                del queues[rank][:admitted_per_pass]

        # The agreed order is uniboost, not plain FCFS: the short job S4
        # arrived after three long jobs but is admitted ahead of them, and
        # the mostly cached long prompt H6 outranks the earlier cold L2.
        second_pass = orders_seen[1]
        self.assertLess(second_pass.index("S4"), second_pass.index("L2"))
        self.assertLess(second_pass.index("H6"), second_pass.index("L2"))

    def test_two_ranks_identical_gamma_trajectories(self):
        with (
            envs.SGLANG_UNIBOOST_GAMMA_ADA.override(True),
            envs.SGLANG_UNIBOOST_GAMMA_ADA_MIN_SAMPLES.override(20),
            envs.SGLANG_UNIBOOST_GAMMA_ADA_REFIT_EVERY_K.override(10),
        ):
            pols = [make_tp_policy(), make_tp_policy()]
        clocks = [RankClock(0.0, 0.031), RankClock(7_777.0, 0.005)]

        n = 80
        # Token-determined completion schedule, identical on both ranks:
        # a 3..7-step body plus a heavy tail every 10th request.
        finish_after = [3 + (i % 5) for i in range(n)]
        for i in range(0, n, 10):
            finish_after[i] = 60 + i

        trajectories = []
        for rank, pol in enumerate(pols):
            ctl = pol.uniboost_gamma_controller
            self.assertIsNotNone(ctl)
            # (b) step-clocked mode: no wall-clock daemon on either rank
            # (per-rank daemon wake timing is exactly the drift source).
            self.assertIsNone(ctl._thread)
            self.assertEqual(ctl.refit_every_k, 10)

            reqs = [
                FakeReq(f"r{i}", 100 + (i % 7) * 500, clocks[rank]()) for i in range(n)
            ]
            with patch(
                "sglang.srt.managers.schedule_policy.time.perf_counter",
                side_effect=clocks[rank],
            ):
                pol.calc_priority(list(reqs))
                gammas = []
                for step in range(1, 201):
                    finished = [reqs[i] for i in range(n) if finish_after[i] == step]
                    for req in finished:
                        req._finished = True
                        # Per-rank garbage wall stamp: must be ignored.
                        req.time_stats.completion_time = clocks[rank]()
                    pol.uniboost_record_finished(finished)
                    gammas.append(ctl.gamma)
            trajectories.append(gammas)
            self.assertGreaterEqual(ctl.refit_count, 3)

        # Bit-identical gamma trajectories despite different per-rank clocks.
        self.assertEqual(trajectories[0], trajectories[1])
        # And the controller actually adapted (not a frozen-gamma tautology).
        self.assertNotEqual(trajectories[0][-1], trajectories[0][0])

    def test_invariant_mode_never_reads_wall_clock(self):
        with (
            envs.SGLANG_UNIBOOST_GAMMA_ADA.override(True),
            envs.SGLANG_UNIBOOST_GAMMA_ADA_MIN_SAMPLES.override(1),
            envs.SGLANG_UNIBOOST_GAMMA_ADA_REFIT_EVERY_K.override(1),
        ):
            pol = make_tp_policy()
        req = FakeReq("r", 100, 5.0)
        other = FakeReq("s", 9000, 6.0)
        boom = patch(
            "sglang.srt.managers.schedule_policy.time.perf_counter",
            side_effect=AssertionError(
                "rank-invariant uniboost read the per-rank wall clock"
            ),
        )
        with boom:
            q = [req, other]
            pol.calc_priority(q)
            req._finished = True
            pol.uniboost_record_finished([req])
        self.assertEqual(pol.uniboost_gamma_controller.refit_count, 1)

    def test_invariant_arrival_is_seq_not_wall_stamp(self):
        # Stream position decides arrival: a request whose per-rank wall
        # stamp says "much older" but that came later in the broadcast
        # stream must NOT jump ahead (equal work => FCFS by sequence).
        pol = make_tp_policy()
        first = FakeReq("first", 5000, 999_999.0)  # huge per-rank stamp
        second = FakeReq("second", 5000, 1.0)  # tiny stamp, later in stream
        q = [first, second]
        pol.calc_priority(q)
        self.assertEqual([r.rid for r in q], ["first", "second"])
        self.assertEqual(first.uniboost_arrival_seq, 0)
        self.assertEqual(second.uniboost_arrival_seq, 1)
        # Sequence numbers are sticky across passes; new arrivals extend.
        third = FakeReq("third", 5000, 0.5)
        q.append(third)
        pol.calc_priority(q)
        self.assertEqual([r.rid for r in q], ["first", "second", "third"])
        self.assertEqual(third.uniboost_arrival_seq, 2)

    def test_phi_quantization_tie_breaks_on_seq(self):
        # (3) deterministic total order: sub-quantum Phi differences cannot
        # reorder requests -- the rank-invariant sequence breaks the tie --
        # while super-quantum differences still do.
        pol = make_tp_policy()
        a = FakeReq("a", 100, 1.0)
        b = FakeReq("b", 200, 2.0)
        q = [a, b]
        pol._uniboost_assign_arrival_seq(q)  # a -> seq 0, b -> seq 1

        def sort_with_boosts(boosts):
            with patch.object(
                pol, "_uniboost_boost_value", side_effect=lambda w: boosts[w]
            ):
                return [r.rid for r in sorted(q, key=pol._uniboost_invariant_key)]

        # Raw phi_b is 2e-5 BELOW phi_a (b "won" by float noise), but the
        # difference is under the 1e-4 quantum: seq order must hold.
        self.assertEqual(sort_with_boosts({100: 0.5, 200: 0.51002}), ["a", "b"])
        # A real (super-quantum) difference still reorders.
        self.assertEqual(sort_with_boosts({100: 0.5, 200: 0.5102}), ["b", "a"])

    def test_invariant_step_latency_recording(self):
        with (
            envs.SGLANG_UNIBOOST_GAMMA_ADA.override(True),
            envs.SGLANG_UNIBOOST_GAMMA_ADA_MIN_SAMPLES.override(100),
            envs.SGLANG_UNIBOOST_GAMMA_ADA_REFIT_EVERY_K.override(100),
        ):
            pol = make_tp_policy()
        ctl = pol.uniboost_gamma_controller
        req = FakeReq("done", 100, 5.0)
        aborted = FakeReq("aborted", 100, 5.0, finished_reason=FINISH_ABORT())
        pol.calc_priority([req, aborted])  # enqueue_step = 0 for both
        for _ in range(3):
            pol.uniboost_record_finished([])  # three empty batch results
        req._finished = True
        aborted._finished = True
        pol.uniboost_record_finished([req, aborted])
        # Latency is in forward steps (4 batch results since enqueue), not
        # seconds; the abort is excluded but still marked as recorded.
        self.assertEqual(list(ctl._samples), [4.0])
        self.assertTrue(aborted.uniboost_latency_recorded)
        # Double-record guard holds in the invariant path too.
        pol.uniboost_record_finished([req])
        self.assertEqual(len(ctl._samples), 1)

    def test_step_clocked_refit_cadence(self):
        # Completion-count trigger: refits every K completions, wall clock
        # irrelevant (interval_s huge), min_samples still gates.
        ctl = GammaAdaController(
            gamma_init=50.0, interval_s=1e9, min_samples=5, refit_every_k=5
        )
        results = [ctl.record_completion(float(3 + (i % 4))) for i in range(4)]
        self.assertEqual(results, [None] * 4)
        self.assertEqual(ctl.refit_count, 0)
        self.assertIsNotNone(ctl.record_completion(30.0))
        self.assertEqual(ctl.refit_count, 1)
        for _ in range(4):
            self.assertIsNone(ctl.record_completion(3.0))
        self.assertIsNotNone(ctl.record_completion(40.0))
        self.assertEqual(ctl.refit_count, 2)

    def test_tp1_wall_clock_path_unchanged(self):
        # (c) legacy behavior: tp == 1 keeps per-rank wall-clock arrivals and
        # the daemon-thread controller; none of the invariant bookkeeping
        # engages.
        pol = make_policy()
        self.assertFalse(pol.uniboost_tp_invariant)
        a = FakeReq("late", 5000, 100.0)
        b = FakeReq("early", 5000, 50.0)
        q = [a, b]
        pol.calc_priority(q)
        # Wall-clock FCFS: the earlier *stamp* wins, not the stream position.
        self.assertEqual([r.rid for r in q], ["early", "late"])
        self.assertIsNone(a.uniboost_arrival_seq)
        self.assertIsNone(a.uniboost_enqueue_step)

        with envs.SGLANG_UNIBOOST_GAMMA_ADA.override(True):
            pol_ada = make_policy()
        ctl = pol_ada.uniboost_gamma_controller
        self.addCleanup(ctl.stop)
        # Daemon refit thread on; completion-count trigger off.
        self.assertIsNotNone(ctl._thread)
        self.assertTrue(ctl._thread.is_alive())
        self.assertEqual(ctl.refit_every_k, 0)


if __name__ == "__main__":
    unittest.main()
