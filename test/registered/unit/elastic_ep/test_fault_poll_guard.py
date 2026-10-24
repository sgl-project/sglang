"""The fault detector stands down during any resize: a grow moves the width too."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    maybe_rebalance_after_rank_fault,
)
from sglang.test.test_utils import CustomTestCase

LAUNCH_SIZE = 4


def _explode(*args):
    raise AssertionError("rebalance ran while the width was moving")


def _rebalance_after_fault():
    return maybe_rebalance_after_rank_fault(
        eplb_manager=SimpleNamespace(rebalance=_explode)
    )


class _StateFixture(CustomTestCase):
    def setUp(self):
        saved = ElasticEPStateManager._instance
        self.addCleanup(setattr, ElasticEPStateManager, "_instance", saved)
        ElasticEPStateManager._instance = None

    def _state(self, pending, **attrs):
        inst = ElasticEPStateManager.__new__(ElasticEPStateManager)
        attrs = dict(effective_ep_size=LAUNCH_SIZE, pending_ep_size=pending, **attrs)
        for name, value in attrs.items():
            setattr(inst, name, value)
        ElasticEPStateManager._instance = inst
        return inst


class TestScalePendingGuard(_StateFixture):
    def test_pending_flags(self):
        self.assertFalse(ElasticEPStateManager.is_scale_pending())  # no instance
        self.assertFalse(ElasticEPStateManager.is_shrink_pending())
        # (pending, is_scale_pending, is_shrink_pending): a grow is not a shrink.
        for pending, scale, shrink in (
            (None, False, False),
            (LAUNCH_SIZE - 1, True, True),
            (LAUNCH_SIZE + 2, True, False),
        ):
            with self.subTest(pending=pending):
                self._state(pending)
                self.assertIs(ElasticEPStateManager.is_scale_pending(), scale)
                self.assertIs(ElasticEPStateManager.is_shrink_pending(), shrink)


class TestRebalanceStandsDownMidResize(_StateFixture):
    def test_no_state_is_a_noop(self):
        self.assertFalse(_rebalance_after_fault())

    def test_pending_resize_skips_the_comparison(self):
        for pending in (LAUNCH_SIZE - 2, LAUNCH_SIZE + 2):
            with self.subTest(pending=pending):
                self._state(pending, is_active_equal_last=_explode)
                self.assertFalse(_rebalance_after_fault())

    def test_at_a_settled_width_the_comparison_runs(self):
        """The guard must not disable fault detection outright."""
        compare = mock.Mock(return_value=True)
        self._state(None, is_active_equal_last=compare)
        self.assertFalse(_rebalance_after_fault())
        compare.assert_called_once_with()


class _ExplodingSelf:
    """A rebalance that gets past the guard trips the first attribute it touches."""

    _rebalance_disabled_reason = None
    _rebalance_disabled_logged = False

    @property
    def _rebalance_layers_per_chunk(self):
        raise AssertionError("periodic rebalance ran while the width was moving")


class TestPeriodicRebalanceStandsDownMidResize(_StateFixture):
    """The periodic path collectives over WORLD; a first resize lacks ``has_scaled``."""

    def _drain(self, pending, has_scaled):
        from sglang.srt.eplb.eplb_manager import EPLBManager

        phase = "pending" if pending is not None else "serving_shrunk"
        self._state(pending, has_scaled=has_scaled, scale_phase=phase)
        return list(EPLBManager.rebalance(_ExplodingSelf()))

    def test_any_resize_stands_down(self):
        # First shrink, first grow (has_scaled False: the gap closed), a later resize.
        for pending, has_scaled in (
            (LAUNCH_SIZE - 1, False),
            (LAUNCH_SIZE + 2, False),
            (LAUNCH_SIZE - 1, True),
        ):
            with self.subTest(pending=pending, has_scaled=has_scaled):
                self.assertEqual(self._drain(pending, has_scaled), [])

    def test_settled_width_still_rebalances(self):
        """The guard must not disable the periodic path outright."""
        with self.assertRaises(AssertionError):
            self._drain(None, has_scaled=False)


if __name__ == "__main__":
    unittest.main()
