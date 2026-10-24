"""A shrink that fails after the mask flip must rebuild decode graphs captured wider."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor import model_runner as model_runner_module
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

LAUNCH_WIDTH = 4


def _fail_scale(live_width, has_decode_graph=True):
    """Fail a scale on a runner captured at LAUNCH_WIDTH; True if it asks to rebuild."""
    runner = ModelRunner.__new__(ModelRunner)
    runner._elastic_pending_graph_recapture = False
    graph = SimpleNamespace(num_dp_ranks=LAUNCH_WIDTH) if has_decode_graph else None
    runner.decode_cuda_graph_runner = graph
    runner._reset_eplb_after_elastic_scale_failure = lambda: None
    # The width the failure leaves behind comes from the patched get_parallel().
    runner._narrow_to_reconciled_width = lambda reconciled: None
    runner._report_elastic_scale_failure = lambda error, size: None
    runner._elastic_global_rank = lambda: 1  # non-zero: rank 0 only adds a log line
    state = mock.Mock(spec=["fail_scale", "get_effective_ep_size"])
    state.get_effective_ep_size.return_value = live_width
    parallel = SimpleNamespace(num_dp_ranks=live_width)
    with (
        mock.patch.object(model_runner_module, "ElasticEPStateManager", state),
        mock.patch.object(model_runner_module, "get_parallel", lambda: parallel),
    ):
        runner._fail_scale("scale failed", live_width)
    # The recapture request is additive: it must not displace the report.
    state.fail_scale.assert_called_once_with("scale failed")
    return runner._elastic_pending_graph_recapture


class TestFailScaleGraphRecapture(unittest.TestCase):
    def test_a_failure_that_narrowed_the_width_requests_a_rebuild(self):
        """The reachable case: await_retirees_departed expires, so the rank narrows."""
        self.assertTrue(
            _fail_scale(LAUNCH_WIDTH - 1),
            "graphs captured at 4 would replay against a 3-wide layout",
        )

    def test_unchanged_width_or_no_decode_graph_has_nothing_to_rebuild(self):
        """A rejection before the mask flip changes nothing, so nothing to rebuild."""
        self.assertFalse(_fail_scale(LAUNCH_WIDTH))
        self.assertFalse(_fail_scale(LAUNCH_WIDTH - 1, has_decode_graph=False))


if __name__ == "__main__":
    unittest.main()
