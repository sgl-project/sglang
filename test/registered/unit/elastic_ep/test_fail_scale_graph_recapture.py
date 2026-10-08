"""A shrink that fails after the mask flip has to rebuild the decode graphs.

Only a committed scale asks the finalizer for a recapture, but the mask flip lands
before the commit. A failure in between narrows this rank, through
``_narrow_to_reconciled_width`` or through a finalize that already ran ``_apply_dp_size``
before it raised, while the decode graphs stay captured at the pre-shrink width. They
would then replay against a layout that no longer exists, which is the blocking case
from the previous round again, now on the failure path.

The runner records the width it was built at, so ``_fail_scale`` compares against that
instead of tracking the capture separately. It reads it after the narrowing, so a
narrowing that itself failed does not ask for a rebuild it does not need.
"""

import unittest
from unittest import mock

from sglang.srt.model_executor import model_runner as model_runner_module
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

LAUNCH_WIDTH = 4


class _Parallel:
    def __init__(self, num_dp_ranks):
        self.num_dp_ranks = num_dp_ranks


class _DecodeRunner:
    """Stands in for the decode graph runner, which records its capture width."""

    def __init__(self, num_dp_ranks):
        self.num_dp_ranks = num_dp_ranks


class _StateManager:
    """Stands in for the ElasticEPStateManager class methods ``_fail_scale`` calls."""

    def __init__(self, reconciled):
        self._reconciled = reconciled
        self.failed_with = None

    def fail_scale(self, error):
        self.failed_with = error

    def get_effective_ep_size(self):
        return self._reconciled


def _model_runner(*, captured_width, has_decode_graph=True):
    runner = ModelRunner.__new__(ModelRunner)
    runner._elastic_pending_graph_recapture = False
    runner.decode_cuda_graph_runner = (
        _DecodeRunner(captured_width) if has_decode_graph else None
    )
    runner._reset_eplb_after_elastic_scale_failure = lambda: None
    # The narrowing itself is covered by _narrow_to_reconciled_width's own callers; what
    # matters here is the width the failure leaves behind, which the patched
    # get_parallel() below supplies.
    runner._narrow_to_reconciled_width = lambda reconciled: None
    runner._report_elastic_scale_failure = lambda error, size: None
    # Non-zero: rank 0 only adds a log line.
    runner._elastic_global_rank = lambda: 1
    return runner


def _fail_scale(runner, *, live_width):
    state = _StateManager(live_width)
    with (
        mock.patch.object(model_runner_module, "ElasticEPStateManager", state),
        mock.patch.object(
            model_runner_module, "get_parallel", lambda: _Parallel(live_width)
        ),
    ):
        runner._fail_scale("scale failed", live_width)
    return state


class TestFailScaleGraphRecapture(unittest.TestCase):
    def test_a_failure_that_narrowed_the_width_requests_a_rebuild(self):
        """The reachable case: await_retirees_departed expires, so the rank narrows."""
        runner = _model_runner(captured_width=LAUNCH_WIDTH)
        _fail_scale(runner, live_width=LAUNCH_WIDTH - 1)
        self.assertTrue(
            runner._elastic_pending_graph_recapture,
            "graphs captured at 4 would replay against a 3-wide layout",
        )

    def test_a_failure_that_left_the_width_alone_does_not(self):
        """A rejection before the mask flip changes nothing, so nothing to rebuild."""
        runner = _model_runner(captured_width=LAUNCH_WIDTH)
        _fail_scale(runner, live_width=LAUNCH_WIDTH)
        self.assertFalse(runner._elastic_pending_graph_recapture)

    def test_no_decode_graph_has_nothing_to_rebuild(self):
        runner = _model_runner(captured_width=LAUNCH_WIDTH, has_decode_graph=False)
        _fail_scale(runner, live_width=LAUNCH_WIDTH - 1)
        self.assertFalse(runner._elastic_pending_graph_recapture)

    def test_the_failure_is_still_reported(self):
        """The recapture request is additive: it must not displace the report."""
        runner = _model_runner(captured_width=LAUNCH_WIDTH)
        state = _fail_scale(runner, live_width=LAUNCH_WIDTH - 1)
        self.assertEqual(state.failed_with, "scale failed")


if __name__ == "__main__":
    unittest.main()
