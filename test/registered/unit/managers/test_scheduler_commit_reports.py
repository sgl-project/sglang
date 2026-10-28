"""The step commit reports its stats once, right after the mode commit and
before the finished requests leave; the owners only write the sink."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as scheduler_mod  # noqa: E402
from sglang.srt.managers.request_lists import CoreLists  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.mode_owners import (  # noqa: E402
    ModeOwners,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.test.scheduler_request_lists import partial_scheduler  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _scheduler(order: Mock, *, mode: str) -> Scheduler:
    s = partial_scheduler()
    s.parts.mode_owners = ModeOwners()
    s.parts.lists = CoreLists()
    s.scheduler_stage_metrics = None
    s.publish_load_snapshot = Mock(return_value=None)
    s.parts.load_publisher = Mock()
    s.hub.load_inquirer = Mock()
    s.parts.mode_owners.hisparse_coordinator = None
    s.parts.lists.waiting_queue = ["waiting"]
    s.parts.deferred_responses = Mock(held={})
    order.disagg_prefill.commit_extend.return_value = ["requeued"]
    order.dllm.commit_extend.return_value = None
    order.processor.process_batch_result_prefill.return_value = None
    s.mode.stages = SimpleNamespace(
        after_decode_commit=None,
        commit_extend={
            "normal": order.processor.process_batch_result_prefill,
            "pd_prefill": order.disagg_prefill.commit_extend,
            "dllm": order.dllm.commit_extend,
        }[mode],
    )
    s.parts.batch_result_processor = order.processor
    s.parts.batch_result_processor.process_batch_result_decode.return_value = 7
    s.hub.metrics_reporter = order.reporter
    s.parts.metrics_sink = Mock()
    s.leave_finished_reqs = order.leave_finished_reqs
    s.parts.kv.tree_cache = Mock()
    s.parts.metrics_sink.enable_fpm = False
    s.parts.request_retirer = Mock()
    s.parts.output_streamer = Mock()
    s.parts.scripted_scheduler_hook = None
    return s


def _batch(forward_mode, prefill_stats):
    return SimpleNamespace(
        reqs=[],
        forward_mode=forward_mode,
        prefill_stats=prefill_stats,
        dp_cooperation_info="dp",
    )


class TestCommitReports(CustomTestCase):
    def _commit(self, *, mode, forward_mode, prefill_stats="stats"):
        order = Mock()
        s = _scheduler(order, mode=mode)
        batch = _batch(forward_mode, prefill_stats)
        result = SimpleNamespace(can_run_cuda_graph=True)
        with (
            patch.object(scheduler_mod, "flush_trace_batch"),
            patch.object(scheduler_mod, "commit_step_counters"),
        ):
            s.process_batch_result(batch, result)
        names = [c[0] for c in order.mock_calls]
        return s, batch, order, names

    def test_decode_reports_after_the_commit_and_before_leaving(self):
        _, batch, order, names = self._commit(
            mode="normal", forward_mode=ForwardMode.DECODE
        )
        self.assertEqual(
            names,
            [
                "processor.process_batch_result_decode",
                "reporter.report_decode_stats",
                "leave_finished_reqs",
            ],
        )
        order.reporter.report_decode_stats.assert_called_once_with(
            True, batch=batch, num_generated_tokens=7
        )

    def test_each_mode_reports_its_prefill_once(self):
        commits = {
            "normal": "processor.process_batch_result_prefill",
            "pd_prefill": "disagg_prefill.commit_extend",
            "dllm": "dllm.commit_extend",
        }
        for mode, commit in commits.items():
            with self.subTest(mode=mode):
                s, batch, order, names = self._commit(
                    mode=mode, forward_mode=ForwardMode.EXTEND
                )
                self.assertEqual(
                    names,
                    [commit, "reporter.report_prefill_stats", "leave_finished_reqs"],
                )
                order.reporter.report_prefill_stats.assert_called_once_with(
                    batch=batch,
                    prefill_stats="stats",
                    can_run_cuda_graph=True,
                    dp_cooperation_info="dp",
                )
                self.assertEqual(
                    s.parts.lists.waiting_queue,
                    ["requeued", "waiting"] if mode == "pd_prefill" else ["waiting"],
                )

    def test_a_step_without_prefill_stats_reports_nothing(self):
        _, _, order, names = self._commit(
            mode="normal", forward_mode=ForwardMode.EXTEND, prefill_stats=None
        )
        self.assertNotIn("reporter.report_prefill_stats", names)


if __name__ == "__main__":
    unittest.main()
