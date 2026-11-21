import collections
import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from sglang.srt.managers.scheduler_components import gc_freezer
from sglang.srt.managers.scheduler_components.gc_freezer import GCFreezer
from sglang.srt.utils.gc_utils import refreeze_gc
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

    def _freezer(self, *, freeze_on_idle_secs=0.0, **batch_trigger):
        return GCFreezer(freeze_on_idle_secs=freeze_on_idle_secs, **batch_trigger)

    def _gc_calls(self):
        names = ("collect", "freeze", "freeze_gc", "refreeze_gc", "unfreeze")
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

    def _frozen_at(self, freezer, forward_cts):
        frozen_at = []
        for forward_ct in forward_cts:
            before = len(self._gc_calls())
            freezer.on_batch(forward_ct)
            if len(self._gc_calls()) > before:
                frozen_at.append(forward_ct)
        return frozen_at

    def test_batch_freeze_collects_young_generations_only_and_repeats(self):
        """A busy scheduler is never idle; its freeze must not run the full
        collection whose pause it removes, and must not walk the frozen list
        (gc.get_freeze_count) on every re-freeze."""
        freezer = self._freezer(freeze_after_batches=10, freeze_every_batches=4)
        self.assertEqual(self._frozen_at(freezer, range(1, 19)), [10, 14, 18])
        self.assertEqual(self._gc_calls(), [call.collect(1), call.freeze()] * 3)
        self.gc_ops.get_freeze_count.assert_called_once()

    def test_batch_freeze_is_off_by_default(self):
        freezer = self._freezer()
        for forward_ct in range(1, 100):
            freezer.on_batch(forward_ct)
        self.assertEqual(self._gc_calls(), [])

    def test_batch_freeze_runs_a_full_collection_and_heap_check_each_interval(self):
        """Frozen cyclic garbage on a never-idle scheduler is bounded by one full
        collection per interval; the heap-growth check runs with it."""
        self.gc_ops.get_freeze_count.return_value = 0
        freezer = self._freezer(freeze_after_batches=1, freeze_every_batches=1)
        freezer.on_batch(1)
        self.now[0] += gc_freezer.GC_CHECK_INTERVAL_SECS
        freezer.on_batch(2)
        freezer.on_batch(3)
        self.assertEqual(
            self._gc_calls(),
            [call.collect(1), call.freeze()]
            + [call.unfreeze(), call.freeze_gc("Scheduler")]
            + [call.collect(1), call.freeze()],
        )
        self.gc_ops.get_objects.assert_called_once_with(2)

    def test_cache_reset_makes_the_next_batch_freeze_collect_fully(self):
        # Unfrozen objects land in the oldest generation, which collect(1) misses.
        freezer = self._freezer(freeze_after_batches=5)
        self._frozen_at(freezer, range(1, 8))
        freezer.on_cache_reset()
        self.assertEqual(self._frozen_at(freezer, (8, 9)), [8])
        self.assertEqual(
            self._gc_calls(),
            [call.collect(1), call.freeze(), call.unfreeze()]
            + [call.unfreeze(), call.freeze_gc("Scheduler")],
        )

    def test_cache_reset_before_the_first_freeze_keeps_the_schedule(self):
        freezer = self._freezer(freeze_after_batches=5, freeze_every_batches=3)
        freezer.on_cache_reset()
        self.assertEqual(self._frozen_at(freezer, range(1, 9)), [5, 8])
        self.assertEqual(
            self._gc_calls(), [call.unfreeze()] + [call.collect(1), call.freeze()] * 2
        )

    def test_cache_reset_restarts_the_cadence_from_the_next_batch(self):
        freezer = self._freezer(freeze_after_batches=10, freeze_every_batches=4)
        self.assertEqual(self._frozen_at(freezer, range(1, 15)), [10, 14])
        freezer.on_cache_reset()
        self.assertEqual(self._frozen_at(freezer, range(15, 24)), [15, 19, 23])

    def test_idle_and_batch_freeze_together(self):
        freezer = self._freezer(freeze_on_idle_secs=5.0, freeze_after_batches=3)
        self.assertEqual(self._frozen_at(freezer, range(1, 4)), [3])
        self.gc_ops.reset_mock()
        self._two_idle_periods(freezer)
        self.assertEqual(self._gc_calls(), [call.freeze_gc("Scheduler")] * 2)

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


class TestBatchFreezeAfterCacheReset(CustomTestCase):
    def test_cyclic_garbage_frozen_by_a_batch_freeze_is_freed_after_a_reset(self):
        """A cache flush must free cyclic garbage that a batch freeze had frozen;
        a young-generation collection never reaches the unfrozen heap."""
        freezer = GCFreezer(freeze_on_idle_secs=0.0, freeze_after_batches=1)
        root = _Node()
        for _ in range(10):
            _Node(root)
        alive = weakref.ref(root)
        gc.disable()  # Only the freezer's own collections may run.
        try:
            freezer.on_batch(1)
            del root
            freezer.on_cache_reset()
            freezer.on_batch(2)
            self.assertIsNone(alive())
        finally:
            gc.unfreeze()
            gc.enable()


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
