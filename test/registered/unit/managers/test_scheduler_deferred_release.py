"""A request finished outside a step commit keeps its memory while a launched
step carries it; the processing of the last such step releases it."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler_components.batch_result_processor import (  # noqa: E402
    SchedulerBatchResultProcessor,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

retire_outside_step = SchedulerBatchResultProcessor.retire_outside_step
release_deferred = SchedulerBatchResultProcessor.release_deferred


class _Req:
    def __init__(self, rid="r"):
        self.rid = rid
        self.inflight_steps = 0
        self.time_stats = Mock()


def _processor():
    return SimpleNamespace(
        deferred_releases={},
        _close_finished_req=Mock(),
        _release_memory=Mock(),
    )


class TestDeferredRelease(CustomTestCase):
    def test_memory_waits_for_the_last_carrying_step(self):
        req, other = _Req("r"), _Req("other")
        first = ScheduleBatch(reqs=[req, other])
        second = ScheduleBatch(reqs=[req])
        first.mark_launched()
        second.mark_launched()
        p = _processor()

        retire_outside_step(p, req, checkpoint=False)
        first.mark_processed()
        release_deferred(p, first)

        p._close_finished_req.assert_called_once_with(req)
        p._release_memory.assert_not_called()
        second.mark_processed()
        release_deferred(p, second)
        p._release_memory.assert_called_once_with(req, checkpoint=False)
        self.assertEqual(p.deferred_releases, {})

    def test_memory_goes_at_once_without_a_carrying_step(self):
        req, p = _Req(), _processor()

        retire_outside_step(p, req, checkpoint=True)

        p._release_memory.assert_called_once_with(req, checkpoint=True)
        self.assertEqual(p.deferred_releases, {})

    def test_a_step_never_launched_is_not_processed(self):
        with self.assertRaises(AssertionError):
            ScheduleBatch(reqs=[_Req()]).mark_processed()


if __name__ == "__main__":
    unittest.main()
