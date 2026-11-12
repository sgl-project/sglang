"""A step's processing retires its forward in the unified pool and releases
the pages pending on it, so the host must have seen that forward end by then,
including for the two results that carry no output of their own: a pipeline
stage's skipped output, whose completion must be ordered after the forward's
done event (the scheduler's current stream does not follow the local forward),
and a prebuilt step's inner idle forward.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata, SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.kv.alloc.forward_steps import ForwardLaunch
from sglang.srt.model_executor.forward_mode import ForwardMode
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


class TestPrebuiltInnerForward(CustomTestCase):
    def test_processed_only_after_the_inner_forward_completes(self):
        """A prebuilt step whose inner idle batch ran a forward carries that
        forward's launch: its `copy_done` is synchronized before the step is
        processed."""
        order = []
        scheduler = mock.Mock(enable_fpm=False)
        scheduler.batch_result_processor.process_batch_result_prebuilt.side_effect = (
            lambda batch: order.append("prebuilt")
        )
        scheduler._on_step_processed.side_effect = lambda *a: order.append("processed")
        copy_done = mock.Mock()
        copy_done.synchronize.side_effect = lambda: order.append("synchronized")
        result = GenerationBatchResult(
            copy_done=copy_done, forward_launch=ForwardLaunch(1)
        )
        batch = mock.Mock(reqs=[], forward_mode=ForwardMode.PREBUILT)
        Scheduler.process_batch_result(scheduler, batch, result)
        self.assertEqual(order[:3], ["synchronized", "prebuilt", "processed"])


if __name__ == "__main__":
    unittest.main()
