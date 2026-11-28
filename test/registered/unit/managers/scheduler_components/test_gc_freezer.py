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
    """The thresholds and collections lockstep uses. A pass takes
    pause_us(interval), or, with pause_us None, fixed_us plus the garbage
    added to garbage_us since the last pass."""

    def __init__(self, clock, pause_us):
        self.threshold = (700, 10, 10)
        self.clock = clock
        self.pause_us = pause_us
        self.interval = None
        self.fixed_us = 0.0
        self.garbage_us = 0.0

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, *values):
        self.threshold = tuple(values) + self.threshold[len(values) :]

    def collect(self):
        if self.pause_us is None:
            pause_us, self.garbage_us = self.fixed_us + self.garbage_us, 0.0
        else:
            pause_us = self.pause_us(self.interval)
        self.clock[0] += pause_us / 1e6


class TestGCLockstep(CustomTestCase):
    """Lockstep full GC: on by default with attention-DP gathers, with an
    interval that every rank derives from the same gathered pause."""

    def _patch(self, pause_us=lambda n: 1):
        clock = [0.0]
        fake = _FakeGC(clock, pause_us)
        self.logger = Mock()
        for target, value in (
            ("gc", fake),
            ("freeze_gc", Mock()),
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
    def _run_phases(freezer, fake, phases):
        """phases: (gathers, any_tokens, garbage_us per gather). Returns
        (phase index, pause_us) per pass."""
        cb = freezer.dp_gather_callbacks()
        pauses = []
        for index, (gathers, any_tokens, garbage_us) in enumerate(phases):
            for _ in range(gathers):
                cb["before_gather"]()
                pause_us = cb["get_gc_pause_us"]()
                if pause_us:
                    pauses.append((index, pause_us))
                cb["after_gather"](pause_us, any_tokens)
                fake.garbage_us += garbage_us
        return pauses

    def test_on_only_with_dp_gathers_and_without_a_third_gc_threshold(self):
        self._patch()
        self.assertEqual(GCFreezer(freeze_on_idle_secs=0.0).dp_gather_callbacks(), {})
        on = GCFreezer(
            freeze_on_idle_secs=0.0, dp_lockstep=True, gc_threshold=[700, 10]
        )
        self.assertEqual(
            set(on.dp_gather_callbacks()),
            {"before_gather", "get_gc_pause_us", "after_gather"},
        )
        self.logger.info.assert_not_called()
        off = GCFreezer(
            freeze_on_idle_secs=0.0, dp_lockstep=True, gc_threshold=[700, 10, 10]
        )
        self.assertEqual(off.dp_gather_callbacks(), {})
        self.assertIn("--gc-threshold", self.logger.info.call_args.args[0])

    def test_idle_gathers_get_idle_passes_and_do_not_size_the_interval(self):
        # 400 us of live objects; idle gathers leave 0.2 us of garbage, loaded 16 us.
        fake = self._patch(pause_us=None)
        fake.fixed_us = 400
        freezer = GCFreezer(freeze_on_idle_secs=0.0, dp_lockstep=True)
        pauses = self._run_phases(
            freezer, fake, [(300_000, False, 0.2), (20_000, True, 16)]
        )
        idle = [p for phase, p in pauses if phase == 0]
        loaded = [p for phase, p in pauses if phase == 1]
        idle_every = gc_freezer.LOCKSTEP_IDLE_GATHERS
        self.assertEqual(len(idle), (300_000 - 1) // idle_every)
        self.assertEqual(set(idle), {round(400 + 0.2 * idle_every)})
        # The longest loaded pass is the first one at 1000 loaded gathers
        # (16.4 ms), which sizes the interval; every later one stays within
        # twice the budget.
        budget = gc_freezer.LOCKSTEP_PAUSE_BUDGET_US
        longest = loaded.index(max(loaded))
        self.assertLessEqual(longest, 1)
        self.assertLess(loaded[longest], 20_000)
        self.assertGreater(len(loaded), 10)
        self.assertTrue(all(p <= 2 * budget for p in loaded[longest + 1 :]))

    def test_a_jump_in_the_garbage_rate_shrinks_the_interval_at_once(self):
        # Long light load grows the interval; then the rate jumps 80x. The pass
        # that meets the jump is long, but the next interval shrinks to its
        # target instead of reading the long pause as live objects.
        fake = self._patch(pause_us=None)
        fake.fixed_us = 400
        freezer = GCFreezer(freeze_on_idle_secs=0.0, dp_lockstep=True)
        pauses = self._run_phases(
            freezer, fake, [(400_000, True, 0.2), (40_000, True, 16)]
        )
        heavy = [p for phase, p in pauses if phase == 1]
        budget = gc_freezer.LOCKSTEP_PAUSE_BUDGET_US
        self.assertGreater(heavy[0], 50 * budget)
        self.assertGreater(len(heavy), 10)
        self.assertTrue(all(p <= 2 * budget for p in heavy[1:]))

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
                cb["after_gather"](max(pauses), True)
        self.assertGreater(len(passes[0]), 3)
        self.assertEqual(passes[0], passes[1])
        self.assertEqual(passes[0], passes[2])


@unittest.skipUnless(
    gc_freezer.third_threshold_works(), "the collector has no third threshold"
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
            cb["after_gather"](cb["get_gc_pause_us"](), True)
            self._churn(keep)
        self.gather = 0
        # Gather 1 is the startup collect-and-freeze.
        self.assertGreaterEqual(len(passes), 2)
        self.assertEqual(passes[0], 51)
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
