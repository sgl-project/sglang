"""A pipeline stage that skips a microbatch's output exchange still retires
that microbatch's forward in the unified pool when it processes the
placeholder result. The host must have seen the forward end by then: the
completion it synchronizes on is ordered after the forward's done event.

BUG REGRESSION. The placeholder's completion was recorded on the scheduler's
current stream, which the local forward does not precede, so the forward was
retired (and its write set dropped from page-move decisions) while it ran.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata, SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Stream:
    def __init__(self, name):
        self.name = name
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)


class _Event:
    def __init__(self):
        self.stream = None

    def record(self, stream=None):
        self.stream = stream


class _Device:
    def __init__(self):
        self.current = _Stream("current")
        self.made = []

    def Event(self):
        return _Event()

    def Stream(self):
        stream = _Stream(f"side{len(self.made)}")
        self.made.append(stream)
        return stream

    def current_stream(self):
        return self.current


class _Stage(SchedulerPPMixin):
    def __init__(self):
        self.device = "cpu"
        self.device_module = _Device()


def _batch():
    return SimpleNamespace(reqs=[object(), object()])


class TestSkippedOutputRetirement(CustomTestCase):
    def test_completion_follows_the_local_forward(self):
        stage = _Stage()
        forward_done = object()
        meta = PPBatchMetadata(
            can_run_cuda_graph=False, unified_forward_done=forward_done
        )
        _, result, completion = stage._pp_make_skip_output_result(_batch(), meta)
        self.assertIs(result.unified_forward_done, forward_done)
        self.assertIsNot(completion.stream, stage.device_module.current)
        self.assertEqual(completion.stream.waited, [forward_done])
        # The scheduler's own stream takes no dependency on the forward.
        self.assertEqual(stage.device_module.current.waited, [])
        # One side stream serves every skipped output.
        _, _, again = stage._pp_make_skip_output_result(_batch(), meta)
        self.assertIs(again.stream, completion.stream)
        self.assertEqual(len(stage.device_module.made), 1)

    def test_without_the_unified_pool_nothing_changes(self):
        stage = _Stage()
        for meta in (PPBatchMetadata(can_run_cuda_graph=False), None):
            with self.subTest(meta=meta):
                _, result, completion = stage._pp_make_skip_output_result(
                    _batch(), meta
                )
                self.assertIsNone(result.unified_forward_done)
                self.assertIs(completion.stream, stage.device_module.current)
                self.assertEqual(stage.device_module.made, [])


if __name__ == "__main__":
    unittest.main()
