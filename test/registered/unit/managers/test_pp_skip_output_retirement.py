"""A step's processing releases what its forward used (in the unified pool, the
pages pending on it), so the host must have seen that forward end by then: a
pipeline stage's received or skipped output, whose completion must be ordered
after the local forward's done event (neither the scheduler's current stream nor
the received outputs follow it), and a prebuilt step's inner idle forward.
"""

import contextlib
import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers import scheduler_pp_mixin
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata, SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.kv.alloc.forward_steps import ForwardLaunch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
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
        self.after = None

    def record(self, stream=None):
        self.stream = stream
        # What the stream had waited when the event was recorded.
        self.after = list(stream.waited)


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

    @contextlib.contextmanager
    def stream(self, stream):
        outer, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = outer


class _Stage(SchedulerPPMixin):
    def __init__(self):
        self.device = "cpu"
        self.device_module = _Device()


def _batch():
    return SimpleNamespace(reqs=[object(), object()])


def _launch(stage, forward_result):
    forward_stream = _Stream("forward")
    stage.forward_stream_ctx = stage.device_module.stream(forward_stream)
    stage._pp_wait_forward_dependencies = lambda: None
    stage.run_batch = lambda batch, pp_proxy_tensors: forward_result
    stage.pp_group = SimpleNamespace(is_last_rank=False)
    batch = SimpleNamespace(reqs=[], spec_algorithm=SpeculativeAlgorithm.NONE)
    mb_metadata = [None]
    stage._pp_launch_batch(0, batch, None, mb_metadata, deque())
    return forward_stream, mb_metadata[0]


def _forward_result(unified_forward_done=None):
    return SimpleNamespace(
        can_run_cuda_graph=False,
        spec_verify_out_cache_loc=None,
        fpm_timing=None,
        forward_launch=None,
        unified_forward_done=unified_forward_done,
    )


class TestSkippedOutputRetirement(CustomTestCase):
    def test_completion_follows_the_local_forward(self):
        stage = _Stage()
        unified_forward_done = object()
        forward_done = object()
        meta = PPBatchMetadata(
            can_run_cuda_graph=False,
            unified_forward_done=unified_forward_done,
            forward_done=forward_done,
        )
        _, result, completion = stage._pp_make_skip_output_result(_batch(), meta)
        self.assertIs(result.unified_forward_done, unified_forward_done)
        self.assertIsNot(completion.stream, stage.device_module.current)
        self.assertEqual(completion.after, [forward_done])
        # The scheduler's own stream takes no dependency on the forward.
        self.assertEqual(stage.device_module.current.waited, [])
        # One side stream serves every skipped output.
        _, _, again = stage._pp_make_skip_output_result(_batch(), meta)
        self.assertIs(again.stream, completion.stream)
        self.assertEqual(len(stage.device_module.made), 1)

    def test_completion_follows_the_forward_without_the_unified_pool(self):
        stage = _Stage()
        forward_stream, meta = _launch(stage, _forward_result())
        _, result, completion = stage._pp_make_skip_output_result(_batch(), meta)
        self.assertIsNone(result.unified_forward_done)
        self.assertIsNot(completion.stream, stage.device_module.current)
        (forward_done,) = completion.stream.waited
        self.assertIs(forward_done.stream, forward_stream)
        self.assertEqual(stage.device_module.current.waited, [])


class TestReceivedOutputRetirement(CustomTestCase):
    def test_completion_follows_the_local_forward(self):
        stage = _Stage()
        copy_stream = _Stream("copy")
        copy_stream.wait_stream = lambda stream: None
        stage.copy_stream = copy_stream
        stage.copy_stream_ctx = stage.device_module.stream(copy_stream)
        stage.schedule_stream = _Stream("schedule")
        stage._pp_send_output_to_next_stage = lambda *args: []
        stage._pp_recv_dict_from_prev_stage = lambda: ({}, None)
        stage._pp_prep_batch_result = lambda batch, meta, outputs: SimpleNamespace(
            logits_output=None
        )
        forward_done = object()
        meta = PPBatchMetadata(can_run_cuda_graph=False, forward_done=forward_done)
        target = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        with (
            mock.patch.object(
                scheduler_pp_mixin,
                "get_parallel",
                return_value=SimpleNamespace(pp_rank=0, pp_size=2),
            ),
            mock.patch.object(scheduler_pp_mixin, "_is_npu", False),
            mock.patch.object(
                scheduler_pp_mixin, "_pp_can_skip_output_comm", return_value=False
            ),
        ):
            _, _, completion, _ = stage._pp_send_recv_and_preprocess_output_tensors(
                0, 0, [target], [meta], deque(), None
            )
        self.assertIs(completion.stream, copy_stream)
        self.assertEqual(completion.after, [forward_done])
        self.assertEqual(stage.device_module.current.waited, [])


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
