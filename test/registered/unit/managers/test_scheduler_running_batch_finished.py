"""Finished requests leave the running batch before admission on every step (#43961)."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as scheduler_mod  # noqa: E402
from sglang.srt.managers.request_lists import CoreLists  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.test.scheduler_request_lists import partial_scheduler  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Req:
    def __init__(self):
        self.done = False

    def finished(self):
        return self.done


class TestRunningBatchFinished(unittest.TestCase):
    def test_request_finished_at_prefill_leaves_before_next_admission(self):
        live, short = _Req(), _Req()

        def commit_extend(batch, result):
            # Overlap: the prefill that ran `short` put it in decoding before
            # its result marks it finished.
            short.done = True

        s = partial_scheduler()
        s.scheduler_stage_metrics = None
        s.parts.lists = CoreLists()
        s.parts.lists.decoding = [live, short]
        s.mode.stages = SimpleNamespace(
            commit_extend=commit_extend, hold_finished=None, leave_finished=None
        )
        s.publish_load_snapshot = Mock(return_value=None)
        s.parts.load_publisher = Mock()
        s.hub.load_inquirer = Mock()
        s.parts.kv.tree_cache = Mock()
        s.parts.metrics_sink = Mock(enable_fpm=False)
        s.parts.request_retirer = Mock()
        s.parts.output_streamer = Mock()
        s.parts.scripted_scheduler_hook = None
        batch = SimpleNamespace(
            reqs=[short], forward_mode=ForwardMode.EXTEND, prefill_stats=None
        )
        with (
            patch.object(scheduler_mod, "flush_trace_batch"),
            patch.object(scheduler_mod, "commit_step_counters"),
        ):
            s.process_batch_result(batch, SimpleNamespace(can_run_cuda_graph=True))

        # The next admission pass reads decoding for its running requests.
        self.assertEqual(s.parts.lists.decoding, [live])


if __name__ == "__main__":
    unittest.main()
