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

    def test_lockstep_is_off_by_default(self):
        self.assertIsNone(self._freezer(freeze_on_idle_secs=5.0).before_dp_gather)

    def test_lockstep_freezes_at_the_first_gather_then_collects_every_n(self):
        self.gc_ops.get_threshold.side_effect = [
            (700, 10, 10),
            (700, 10, gc_freezer.NO_AUTOMATIC_FULL_GC),
        ]
        freezer = GCFreezer(freeze_on_idle_secs=0.0, dp_lockstep_steps=3)
        collects_after_each_gather = []
        for _ in range(7):
            freezer.before_dp_gather()
            collects_after_each_gather.append(self.gc_ops.collect.call_count)
        self.assertEqual(collects_after_each_gather, [0, 0, 1, 1, 1, 2, 2])
        self.gc_ops.freeze_gc.assert_called_once_with("Scheduler")
        self.gc_ops.set_threshold.assert_called_once_with(
            700, 10, gc_freezer.NO_AUTOMATIC_FULL_GC
        )
        self.gc_ops.logger.warning.assert_not_called()

    def test_lockstep_warns_when_the_third_threshold_is_ignored(self):
        self.gc_ops.get_threshold.return_value = (2000, 10, 0)
        GCFreezer(freeze_on_idle_secs=0.0, dp_lockstep_steps=3).before_dp_gather()
        self.gc_ops.logger.warning.assert_called_once()


def _third_threshold_is_settable() -> bool:
    saved = gc.get_threshold()
    try:
        gc.set_threshold(saved[0], saved[1], gc_freezer.NO_AUTOMATIC_FULL_GC)
        return gc.get_threshold()[2] == gc_freezer.NO_AUTOMATIC_FULL_GC
    finally:
        gc.set_threshold(*saved)


@unittest.skipUnless(
    _third_threshold_is_settable(), "the collector has no third threshold"
)
class TestGCLockstepCollector(CustomTestCase):
    """Against the real collector: with a frozen heap, allocation churn starts
    automatic full collections; in lockstep mode only the gathers that are
    multiples of the step count run one."""

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

    def _run(self, before_gather, gathers=200):
        keep = []
        for self.gather in range(1, gathers + 1):
            before_gather()
            self._churn(keep)
        self.gather = 0

    def test_full_collections_run_only_at_lockstep_gathers(self):
        freeze_gc("test")
        # Without this pass the collector still compares new survivors to the
        # whole pre-freeze heap and would start no full collection here.
        gc.collect()
        self.full_collections.clear()
        self._run(lambda: None)
        gc.unfreeze()
        self.assertTrue(self.full_collections, "the churn starts no full collection")

        self.full_collections.clear()
        freezer = GCFreezer(freeze_on_idle_secs=0.0, dp_lockstep_steps=50)
        self._run(freezer.before_dp_gather)
        # Gather 1 is the startup collect-and-freeze.
        self.assertEqual(self.full_collections, [1, 50, 100, 150, 200])
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
