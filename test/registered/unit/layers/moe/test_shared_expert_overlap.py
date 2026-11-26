"""Check the shared-down/combine schedule with deterministic CPU stream clocks."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Stream:
    def __init__(self):
        self.time = 0

    def wait_stream(self, stream):
        self.time = max(self.time, stream.time)


class _Timeline:
    def __init__(self):
        self.main = _Stream()
        self.shared = _Stream()
        self.current = self.main
        self.intervals = []

    @contextmanager
    def stream(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous

    def enqueue(self, name, duration):
        start = self.current.time
        self.current.time += duration
        self.intervals.append((name, start, self.current.time))


class _Dispatcher(BaseDispatcher):
    def __init__(self, timeline):
        super().__init__()
        self.timeline = timeline

    def dispatch(self, hidden_states, topk_output):
        self.timeline.enqueue("dispatch", 10)
        return hidden_states

    def combine(self, combine_input):
        self.timeline.enqueue("combine", 10)
        return combine_input


class _Experts:
    def __init__(self, timeline):
        self.timeline = timeline
        self.dispatcher = _Dispatcher(timeline)

    def __call__(self, hidden_states, topk_output):
        dispatched = self.dispatcher.dispatch(hidden_states, topk_output)
        self.timeline.enqueue("routed_compute", 15)
        return self.dispatcher.combine(combine_input=2 * dispatched)


class _SharedExperts:
    def __init__(self, timeline):
        self.timeline = timeline

    def __call__(self, hidden_states, *, defer_down_proj=False, **kwargs):
        self.timeline.enqueue("shared_front", 20)
        down_input = hidden_states + 2
        if defer_down_proj:
            return down_input
        return self.down_proj(down_input)[0]

    def down_proj(self, down_input):
        self.timeline.enqueue("shared_down", 20)
        return 3 * down_input, None


class TestSharedDownCombineOverlap(CustomTestCase):
    def test_shared_down_overlaps_combine(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                timeline = _Timeline()
                model = object.__new__(DeepseekV2MoE)
                torch.nn.Module.__init__(model)
                model._fuse_shared_experts_inside_sbo = False
                model._npu_shared_down_stream = timeline.shared
                model.num_fused_shared_experts = 0
                model.is_nextn = False
                model.alt_stream = None
                model.gate = lambda x, forward_batch: x
                model.topk = lambda x, logits: logits
                model.experts = _Experts(timeline)
                model.shared_experts = _SharedExperts(timeline)
                model.routed_scaling_factor = 1.0
                model.routed_scaling_factor_fused_in_topk = False

                x = torch.ones(2, 4)
                with (
                    patch.object(
                        torch.cuda, "current_stream", lambda: timeline.current
                    ),
                    patch.object(torch.cuda, "stream", timeline.stream),
                    patch.object(
                        torch.Tensor, "record_stream", autospec=True
                    ) as record,
                ):
                    output = model.forward_deepep(x, SimpleNamespace(forward_mode=mode))

                stages = {name: (start, end) for name, start, end in timeline.intervals}
                self.assertLess(stages["routed_compute"][0], stages["shared_front"][1])
                down_starts = [
                    start
                    for name, start, _ in timeline.intervals
                    if name == "shared_down"
                ]
                self.assertEqual(down_starts, [stages["combine"][0]])
                self.assertGreaterEqual(timeline.main.time, stages["shared_down"][1])
                torch.testing.assert_close(output, 5 * x + 6)
                record.assert_any_call(timeline.main)


if __name__ == "__main__":
    unittest.main()
