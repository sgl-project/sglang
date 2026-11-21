import collections
import gc
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from sglang.srt.managers.scheduler_components import gc_freezer
from sglang.srt.managers.scheduler_components.gc_freezer import GCFreezer
from sglang.srt.utils.gc_utils import freeze_gc, refreeze_gc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Node:
    def __init__(self, parent=None):
        self.parent, self.children = parent, []
        if parent is not None:
            parent.children.append(self)


class TestGCFreezer(CustomTestCase):
    def setUp(self):
        self.now = [100.0]
        self.forward_ct = [7]
        self.gc_ops = Mock()
        self.gc_ops.refreeze_gc.return_value = collections.Counter({"_Node": 3})
        self.gc_ops.get_objects.return_value = []
        for target, value in (
            ("time", SimpleNamespace(monotonic=lambda: self.now[0])),
            ("gc", self.gc_ops),
            ("freeze_gc", self.gc_ops.freeze_gc),
            ("refreeze_gc", self.gc_ops.refreeze_gc),
            ("logger", self.gc_ops.logger),
        ):
            patcher = patch.object(gc_freezer, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _freezer(self, *, freeze_on_idle_secs):
        return GCFreezer(freeze_on_idle_secs=freeze_on_idle_secs)

    def _gc_calls(self):
        names = ("collect", "freeze_gc", "refreeze_gc", "unfreeze")
        return [c for c in self.gc_ops.mock_calls if c[0] in names]

    def _two_idle_periods(self, freezer):
        freezer.on_idle(self.forward_ct[0])
        self.now[0] += 4.9
        freezer.on_idle(self.forward_ct[0])
        self.assertEqual(self._gc_calls(), [])
        self.now[0] += 0.1
        for _ in range(3):
            freezer.on_idle(self.forward_ct[0])
        self.forward_ct[0] += 1
        freezer.on_idle(self.forward_ct[0])
        self.now[0] += 5.0
        freezer.on_idle(self.forward_ct[0])

    def test_idle_freeze_runs_once_per_idle_period(self):
        self._two_idle_periods(self._freezer(freeze_on_idle_secs=5.0))
        self.assertEqual(self._gc_calls(), [call.freeze_gc("Scheduler")] * 2)

    def test_frozen_garbage_is_checked_and_warned_at_most_once_per_interval(self):
        freezer = self._freezer(freeze_on_idle_secs=5.0)
        freezer._next_garbage_check = 0.0
        self._two_idle_periods(freezer)
        self.assertEqual(
            self._gc_calls(),
            [call.refreeze_gc("Scheduler"), call.freeze_gc("Scheduler")],
        )
        self.gc_ops.logger.warning.assert_called_once()

    def test_cache_reset_unfreezes_and_the_next_idle_check_freezes(self):
        self._freezer(freeze_on_idle_secs=0.0).on_cache_reset()
        self.assertEqual(self._gc_calls(), [])
        freezer = self._freezer(freeze_on_idle_secs=5.0)
        freezer.on_idle(self.forward_ct[0])
        freezer.on_cache_reset()
        self.assertEqual(self._gc_calls(), [call.unfreeze()])
        freezer.on_idle(self.forward_ct[0])
        self.assertEqual(
            self._gc_calls(),
            [call.unfreeze(), call.freeze_gc("Scheduler")],
        )

    def test_heap_growth_warns_at_each_half_again(self):
        freezer = self._freezer(freeze_on_idle_secs=0.0)
        self.gc_ops.get_freeze_count.return_value = 0
        warned = []
        for count in (1000, 1400, 1500, 2000, 2250):
            self.gc_ops.get_objects.return_value = [object()] * count
            self.gc_ops.logger.warning.reset_mock()
            freezer._check_heap_growth()
            warned.append(self.gc_ops.logger.warning.called)
        self.assertEqual(warned, [False, False, True, False, True])


class _FakeGC:
    """The thresholds and collections lockstep uses; a pass takes pause_us(interval)."""

    def __init__(self, clock, pause_us, keeps_third=True):
        self.threshold = (700, 10, 10)
        self.keeps_third = keeps_third
        self.clock = clock
        self.pause_us = pause_us
        self.interval = None
        self.collections = 0

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, *values):
        t = tuple(values) + self.threshold[len(values) :]
        self.threshold = t if self.keeps_third else (t[0], t[1], 0)

    def collect(self):
        self.collections += 1
        self.clock[0] += self.pause_us(self.interval) / 1e6


class TestGCLockstep(CustomTestCase):
    """Lockstep full GC: on by default with attention-DP gathers, with an
    interval that every rank derives from the same gathered pause."""

    def _patch(self, pause_us=lambda n: 1, keeps_third=True):
        clock = [0.0]
        fake = _FakeGC(clock, pause_us, keeps_third)
        self.freeze_gc = Mock()
        self.logger = Mock()
        for target, value in (
            ("gc", fake),
            ("freeze_gc", self.freeze_gc),
            ("logger", self.logger),
            (
                "time",
                SimpleNamespace(monotonic=lambda: 0.0, perf_counter=lambda: clock[0]),
            ),
        ):
            patcher = patch.object(gc_freezer, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return fake

    @staticmethod
    def _run(ranks, gathers, fake=None):
        """One gather at a time on every rank; returns each rank's pass gathers."""
        cbs = [r.dp_gather_callbacks() for r in ranks]
        passes = [[] for _ in ranks]
        for gather in range(1, gathers + 1):
            for rank, cb in enumerate(cbs):
                if fake is not None:
                    fake.interval = ranks[rank]._interval
                cb["before_gather"]()
                if cb["get_gc_pause_us"]():
                    passes[rank].append(gather)
            longest = max(cb["get_gc_pause_us"]() for cb in cbs)
            for cb in cbs:
                cb["on_gc_pause"](longest)
        return passes

    def test_on_only_with_dp_gathers_and_without_a_third_gc_threshold(self):
        self._patch()
        self.assertEqual(GCFreezer(freeze_on_idle_secs=0.0).dp_gather_callbacks(), {})
        on = GCFreezer(
            freeze_on_idle_secs=0.0, dp_lockstep=True, gc_threshold=[700, 10]
        )
        self.assertEqual(
            set(on.dp_gather_callbacks()),
            {"before_gather", "get_gc_pause_us", "on_gc_pause"},
        )
        self.logger.info.assert_not_called()
        off = GCFreezer(
            freeze_on_idle_secs=0.0, dp_lockstep=True, gc_threshold=[700, 10, 10]
        )
        self.assertEqual(off.dp_gather_callbacks(), {})
        self.assertIn("--gc-threshold", self.logger.info.call_args.args[0])

    def test_off_when_the_collector_ignores_the_third_threshold(self):
        fake = self._patch(keeps_third=False)
        freezer = GCFreezer(freeze_on_idle_secs=0.0, dp_lockstep=True)
        self.assertEqual(freezer.dp_gather_callbacks(), {})
        self.assertIn("third threshold", self.logger.info.call_args.args[0])
        self.assertEqual(fake.threshold[:2], (700, 10))

    def test_first_gather_freezes_and_the_first_pass_comes_after_the_initial_interval(
        self,
    ):
        fake = self._patch()
        freezer = GCFreezer(freeze_on_idle_secs=0.0, dp_lockstep=True)
        (passes,) = self._run([freezer], gc_freezer.LOCKSTEP_INITIAL_GATHERS)
        self.freeze_gc.assert_called_once_with("Scheduler")
        self.assertEqual(fake.threshold, (700, 10, gc_freezer.NO_AUTOMATIC_FULL_GC))
        self.assertEqual(passes, [gc_freezer.LOCKSTEP_INITIAL_GATHERS])

    def _intervals(self, pause_us, gathers):
        fake = self._patch(pause_us=pause_us)
        freezer = GCFreezer(freeze_on_idle_secs=0.0, dp_lockstep=True)
        (passes,) = self._run([freezer], gathers, fake)
        return [b - a for a, b in zip(passes, passes[1:])]

    def test_interval_grows_at_most_twofold_toward_the_pause_budget(self):
        # 0.5 us of garbage per gather: the budget fits 10000 gathers.
        intervals = self._intervals(lambda n: 0.5 * n, 60_000)
        self.assertEqual(intervals[:5], [2000, 4000, 8000, 10000, 10000])

    def test_interval_shrinks_at_once_after_a_long_pass(self):
        # 50 us per gather: the first pass (1000 gathers) takes 50 ms.
        intervals = self._intervals(lambda n: 50 * n, 3000)
        self.assertEqual(set(intervals), {100})

    def test_a_large_fixed_part_is_spread_over_longer_intervals(self):
        # 8 ms of live objects per pass whatever the interval, plus 0.1 us per
        # gather: shrinking cannot meet the 5 ms budget, so the interval grows
        # until the garbage adds about as much again.
        intervals = self._intervals(lambda n: 8000 + 0.1 * n, 400_000)
        self.assertEqual(min(intervals), intervals[0])
        # Non-decreasing, up to the fit's noise from microsecond-rounded pauses.
        self.assertTrue(all(b >= a * 0.999 for a, b in zip(intervals, intervals[1:])))
        self.assertAlmostEqual(intervals[-1], 80_000, delta=1_000)

    def test_ranks_with_different_pauses_collect_at_the_same_gathers(self):
        fake = self._patch()
        ranks = [GCFreezer(freeze_on_idle_secs=0.0, dp_lockstep=True) for _ in range(3)]
        scale = {id(r): 1 + 0.3 * k for k, r in enumerate(ranks)}
        cbs = [r.dp_gather_callbacks() for r in ranks]
        passes = [[] for _ in ranks]
        for gather in range(1, 50_001):
            pauses = []
            for rank, (r, cb) in enumerate(zip(ranks, cbs)):
                fake.pause_us = lambda n, s=scale[id(r)]: s * (300 + 0.4 * n)
                fake.interval = r._interval
                cb["before_gather"]()
                pauses.append(cb["get_gc_pause_us"]())
                if pauses[-1]:
                    passes[rank].append(gather)
            for cb in cbs:
                cb["on_gc_pause"](max(pauses))
        self.assertGreater(len(passes[0]), 3)
        self.assertEqual(passes[0], passes[1])
        self.assertEqual(passes[0], passes[2])

    def test_interval_bounds(self):
        self._patch()
        freezer = GCFreezer(freeze_on_idle_secs=0.0, dp_lockstep=True)
        self.assertEqual(freezer._next_interval(10**9), gc_freezer.LOCKSTEP_MIN_GATHERS)
        freezer._interval = 80_000
        self.assertEqual(freezer._next_interval(1), gc_freezer.LOCKSTEP_MAX_GATHERS)


def _third_threshold_is_settable() -> bool:
    return gc_freezer.third_threshold_works()


@unittest.skipUnless(
    _third_threshold_is_settable(), "the collector has no third threshold"
)
class TestGCLockstepCollector(CustomTestCase):
    """Against the real collector: with a frozen heap, allocation churn starts
    automatic full collections; in lockstep mode only the lockstep passes run."""

    def setUp(self):
        saved_threshold = gc.get_threshold()
        self.addCleanup(gc.set_threshold, *saved_threshold)
        self.addCleanup(gc.unfreeze)
        self.gather = 0
        self.full_collections = []

        def on_gc(phase, info):
            if phase == "start" and info["generation"] == 2:
                self.full_collections.append(self.gather)

        gc.callbacks.append(on_gc)
        self.addCleanup(gc.callbacks.remove, on_gc)

    def _churn(self, keep):
        # Cyclic garbage, plus survivors that the full-collection rule counts.
        for i in range(2000):
            a = [i]
            a.append([a])
            if i % 4 == 0:
                keep.append([i])

    def test_full_collections_run_only_at_lockstep_passes(self):
        freeze_gc("test")
        # Without this pass the collector still compares new survivors to the
        # whole pre-freeze heap and would start no full collection here.
        gc.collect()
        self.full_collections.clear()
        keep = []
        for self.gather in range(1, 201):
            self._churn(keep)
        gc.unfreeze()
        self.assertTrue(self.full_collections, "the churn starts no full collection")

        self.full_collections.clear()
        keep = []
        with patch.object(gc_freezer, "LOCKSTEP_INITIAL_GATHERS", 50):
            freezer = GCFreezer(freeze_on_idle_secs=0.0, dp_lockstep=True)
        cb = freezer.dp_gather_callbacks()
        passes = []
        for self.gather in range(1, 401):
            cb["before_gather"]()
            if cb["get_gc_pause_us"]():
                passes.append(self.gather)
            cb["on_gc_pause"](cb["get_gc_pause_us"]())
            self._churn(keep)
        self.gather = 0
        # Gather 1 is the startup collect-and-freeze.
        self.assertGreaterEqual(len(passes), 2)
        self.assertEqual(passes[0], 50)
        self.assertEqual(self.full_collections, [1] + passes)
        self.assertGreater(gc.get_freeze_count(), 0)


class TestRefreezeGC(CustomTestCase):
    def test_finds_and_frees_frozen_cyclic_garbage(self):
        root = _Node()
        for _ in range(10):
            _Node(root)
        try:
            gc.collect()
            gc.freeze()
            del root
            gc.collect()
            garbage = refreeze_gc("test")
            gc.unfreeze()
            self.assertEqual(garbage["_Node"], 11)
            self.assertFalse(any(type(o) is _Node for o in gc.get_objects()))
        finally:
            gc.unfreeze()


if __name__ == "__main__":
    unittest.main()
