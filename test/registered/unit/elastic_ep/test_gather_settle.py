"""Unit tests for waiting out Mooncake's membership view on the mlp_sync gather.

A scale converges the view with ``mooncake_world_settle_probe`` before the hot path. A
bare fault has no such probe: the coordinator drops the peer on its own schedule, and the
gather is the first thing a tick does, so it can be refused while the change is still
being digested. The refusal is a precondition check, raised before the collective is
posted, which is what makes waiting safe rather than a way to deadlock the cohort.

The budget is the other half of the contract. It is far enough under the watchdog that a
view which never settles still fails loudly on its own terms.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.elastic_ep import elastic_ep
from sglang.srt.elastic_ep.elastic_ep import mooncake_all_gather_settling
from sglang.test.test_utils import CustomTestCase

INACTIVE = "mooncakePgAllGather failed: invalid state: rank is not active in this group"


class _Gather:
    """Stands in for ``all_gather_single``, refusing the first ``refusals`` calls."""

    def __init__(self, refusals, exc=None):
        self.refusals = refusals
        self.calls = 0
        self.exc = exc or RuntimeError(INACTIVE)

    def __call__(self, output, input_, *, group):
        self.calls += 1
        if self.calls <= self.refusals:
            raise self.exc


class _SettleCase(CustomTestCase):
    def setUp(self):
        import sglang.srt.distributed.utils as dist_utils

        self._utils = dist_utils
        self._saved = dist_utils.all_gather_single

    def tearDown(self):
        self._utils.all_gather_single = self._saved

    def _run(self, gather):
        self._utils.all_gather_single = gather
        mooncake_all_gather_settling(None, None, group=None)


class TestQuietPathIsUntouched(_SettleCase):
    def test_a_gather_that_succeeds_is_posted_once(self):
        gather = _Gather(refusals=0)
        self._run(gather)
        self.assertEqual(gather.calls, 1)

    def test_an_unrelated_runtime_error_is_not_retried(self):
        """Only the inactive refusal is ours to wait on; everything else propagates."""
        gather = _Gather(refusals=1, exc=RuntimeError("CUDA error: out of memory"))
        self._utils.all_gather_single = gather
        with self.assertRaises(RuntimeError) as caught:
            mooncake_all_gather_settling(None, None, group=None)
        self.assertIn("out of memory", str(caught.exception))
        self.assertEqual(gather.calls, 1)


class TestRefusalIsWaitedOut(_SettleCase):
    def test_the_gather_lands_once_the_view_settles(self):
        gather = _Gather(refusals=3)
        self._run(gather)
        self.assertEqual(gather.calls, 4)

    def test_a_view_that_never_settles_still_fails(self):
        """Loudly, and naming the condition rather than passing Mooncake's text up."""
        gather = _Gather(refusals=10**9)
        self._utils.all_gather_single = gather
        saved = elastic_ep._GATHER_SETTLE_BUDGET_S
        elastic_ep._GATHER_SETTLE_BUDGET_S = 0.2
        try:
            with self.assertRaises(RuntimeError) as caught:
                mooncake_all_gather_settling(None, None, group=None)
        finally:
            elastic_ep._GATHER_SETTLE_BUDGET_S = saved
        message = str(caught.exception)
        self.assertIn("membership view never", message)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertIn("not active", str(caught.exception.__cause__))

    def test_the_budget_is_bounded_well_under_the_watchdog(self):
        """300s is the default watchdog; a budget near it would park the cohort."""
        self.assertLess(elastic_ep._GATHER_SETTLE_BUDGET_S, 60.0)
        self.assertGreater(elastic_ep._GATHER_SETTLE_BUDGET_S, 1.0)


if __name__ == "__main__":
    unittest.main()
