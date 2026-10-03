"""Unit tests for keeping the fault detector off a width that is still moving.

``is_active_equal_last`` compares two device tensors, so the per-forward fault check
forces a stream sync at the tail of every forward. Mid resize that sync can land on a
collective a departing or arriving peer will never post, which the cohort sees as a
scheduler watchdog timeout. Two places therefore stand down until commit: the rebalance
itself, and the NIXL poll that writes the mask it reads.

The guard is "any resize in flight", not "a shrink in flight": a grow moves the width too.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    maybe_rebalance_after_rank_fault,
)
from sglang.test.test_utils import CustomTestCase

LAUNCH_SIZE = 4


class TestScalePendingGuard(CustomTestCase):
    def setUp(self):
        self._saved = ElasticEPStateManager._instance
        ElasticEPStateManager._instance = None

    def tearDown(self):
        ElasticEPStateManager._instance = self._saved

    def _state(self, pending):
        inst = ElasticEPStateManager.__new__(ElasticEPStateManager)
        inst.effective_ep_size = LAUNCH_SIZE
        inst.pending_ep_size = pending
        ElasticEPStateManager._instance = inst
        return inst

    def test_no_instance_is_not_pending(self):
        self.assertFalse(ElasticEPStateManager.is_scale_pending())
        self.assertFalse(ElasticEPStateManager.is_shrink_pending())

    def test_idle_width_is_not_pending(self):
        self._state(None)
        self.assertFalse(ElasticEPStateManager.is_scale_pending())

    def test_pending_shrink_is_pending(self):
        self._state(LAUNCH_SIZE - 1)
        self.assertTrue(ElasticEPStateManager.is_scale_pending())
        self.assertTrue(ElasticEPStateManager.is_shrink_pending())

    def test_pending_grow_is_pending_but_not_a_shrink(self):
        """The case the guard was widened for: a grow moves the width too."""
        self._state(LAUNCH_SIZE + 2)
        self.assertTrue(ElasticEPStateManager.is_scale_pending())
        self.assertFalse(ElasticEPStateManager.is_shrink_pending())


class ExplodingManager:
    """Any call means the guard let a rebalance through."""

    def rebalance(self):
        raise AssertionError("rebalance ran while the width was moving")


class TestRebalanceStandsDownMidResize(CustomTestCase):
    def setUp(self):
        self._saved = ElasticEPStateManager._instance
        ElasticEPStateManager._instance = None

    def tearDown(self):
        ElasticEPStateManager._instance = self._saved

    def _state(self, pending):
        inst = ElasticEPStateManager.__new__(ElasticEPStateManager)
        inst.effective_ep_size = LAUNCH_SIZE
        inst.pending_ep_size = pending
        # Reached only if the guard fails, and it fails loudly rather than syncing.
        inst.is_active_equal_last = lambda: (_ for _ in ()).throw(
            AssertionError("compared the mask while the width was moving")
        )
        ElasticEPStateManager._instance = inst
        return inst

    def test_no_state_is_a_noop(self):
        self.assertFalse(
            maybe_rebalance_after_rank_fault(eplb_manager=ExplodingManager())
        )

    def test_pending_shrink_skips_the_comparison(self):
        self._state(LAUNCH_SIZE - 2)
        self.assertFalse(
            maybe_rebalance_after_rank_fault(eplb_manager=ExplodingManager())
        )

    def test_pending_grow_skips_the_comparison(self):
        self._state(LAUNCH_SIZE + 2)
        self.assertFalse(
            maybe_rebalance_after_rank_fault(eplb_manager=ExplodingManager())
        )

    def test_at_a_settled_width_the_comparison_runs(self):
        """The guard must not disable fault detection outright."""
        inst = self._state(None)
        seen = []
        inst.is_active_equal_last = lambda: seen.append(1) or True
        self.assertFalse(
            maybe_rebalance_after_rank_fault(eplb_manager=ExplodingManager())
        )
        self.assertEqual(len(seen), 1)


class _ExplodingSelf:
    """A rebalance that gets past the guard trips the first attribute it touches."""

    _rebalance_disabled_reason = None
    _rebalance_disabled_logged = False

    @property
    def _rebalance_layers_per_chunk(self):
        raise AssertionError("periodic rebalance ran while the width was moving")


class TestPeriodicRebalanceStandsDownMidResize(CustomTestCase):
    """The periodic path all reduces over WORLD and broadcasts from src 0.

    Its own stand-down is gated on ``has_scaled``, so it covers only a cohort that has
    resized once before. A server's first resize has ``has_scaled`` False and would walk
    into both collectives with a peer still arriving or already gone.
    """

    def setUp(self):
        self._saved = ElasticEPStateManager._instance
        ElasticEPStateManager._instance = None

    def tearDown(self):
        ElasticEPStateManager._instance = self._saved

    def _state(self, pending, has_scaled):
        inst = ElasticEPStateManager.__new__(ElasticEPStateManager)
        inst.effective_ep_size = LAUNCH_SIZE
        inst.pending_ep_size = pending
        inst.has_scaled = has_scaled
        inst.scale_phase = "pending" if pending is not None else "serving_shrunk"
        ElasticEPStateManager._instance = inst
        return inst

    def _drain(self):
        from sglang.srt.eplb.eplb_manager import EPLBManager

        return list(EPLBManager.rebalance(_ExplodingSelf()))

    def test_first_shrink_stands_down(self):
        """has_scaled False: the gap this guard closes."""
        self._state(LAUNCH_SIZE - 1, has_scaled=False)
        self.assertEqual(self._drain(), [])

    def test_first_grow_stands_down(self):
        self._state(LAUNCH_SIZE + 2, has_scaled=False)
        self.assertEqual(self._drain(), [])

    def test_later_resize_stands_down(self):
        self._state(LAUNCH_SIZE - 1, has_scaled=True)
        self.assertEqual(self._drain(), [])

    def test_settled_width_still_rebalances(self):
        """The guard must not disable the periodic path outright."""
        self._state(None, has_scaled=False)
        with self.assertRaises(AssertionError):
            self._drain()


if __name__ == "__main__":
    unittest.main()
