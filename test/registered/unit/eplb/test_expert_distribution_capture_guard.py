"""Unit tests for the expert-distribution pass-boundary capture guard — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

import unittest

from sglang.srt.eplb.expert_distribution import _ExpertDistributionRecorderReal
from sglang.test.test_utils import CustomTestCase


class _StubGatherer:
    def __init__(self):
        self.reset_calls = 0
        self.start_calls = 0
        self.collect_calls = 0

    def reset(self):
        self.reset_calls += 1

    def on_forward_pass_start(self, forward_batch):
        self.start_calls += 1

    def collect(self):
        self.collect_calls += 1
        return {}


class _StubAccumulator:
    def __init__(self):
        self.append_calls = 0

    def append(self, forward_pass_id, gatherer_key, single_pass_data, outputs):
        self.append_calls += 1


def _make_recorder(capturing: bool) -> _ExpertDistributionRecorderReal:
    """Recorder with stubbed collaborators; __init__ needs a model config so
    it is bypassed and only the fields the pass boundary touches are set."""
    recorder = _ExpertDistributionRecorderReal.__new__(_ExpertDistributionRecorderReal)
    recorder._recording = True
    recorder._is_current_stream_capturing = lambda: capturing
    recorder._single_pass_gatherers = {"primary": _StubGatherer()}
    recorder._accumulator = _StubAccumulator()
    return recorder


class TestExpertDistributionCaptureGuard(CustomTestCase):
    """The forward-pass boundary must be a no-op while a stream capture is
    active (e.g. EAGLE multi-step draft CUDA-graph capture drives full
    `ModelRunner.forward` calls inside the capture): the accumulator's rank-0
    GPU->CPU sync would otherwise invalidate the capture."""

    def test_pass_boundary_skipped_while_capturing(self):
        recorder = _make_recorder(capturing=True)
        recorder._on_forward_pass_start(forward_batch=None)
        recorder._on_forward_pass_end(forward_pass_id=1, outputs={})

        gatherer = recorder._single_pass_gatherers["primary"]
        self.assertEqual(gatherer.reset_calls, 0)
        self.assertEqual(gatherer.start_calls, 0)
        self.assertEqual(gatherer.collect_calls, 0)
        self.assertEqual(recorder._accumulator.append_calls, 0)

    def test_pass_boundary_runs_when_not_capturing(self):
        recorder = _make_recorder(capturing=False)
        recorder._on_forward_pass_start(forward_batch=None)
        recorder._on_forward_pass_end(forward_pass_id=1, outputs={})

        gatherer = recorder._single_pass_gatherers["primary"]
        self.assertEqual(gatherer.reset_calls, 1)
        self.assertEqual(gatherer.start_calls, 1)
        self.assertEqual(gatherer.collect_calls, 1)
        self.assertEqual(recorder._accumulator.append_calls, 1)

    def test_pass_boundary_skipped_when_not_recording(self):
        recorder = _make_recorder(capturing=False)
        recorder._recording = False
        recorder._on_forward_pass_start(forward_batch=None)
        recorder._on_forward_pass_end(forward_pass_id=1, outputs={})

        gatherer = recorder._single_pass_gatherers["primary"]
        self.assertEqual(gatherer.reset_calls, 0)
        self.assertEqual(gatherer.collect_calls, 0)
        self.assertEqual(recorder._accumulator.append_calls, 0)


if __name__ == "__main__":
    unittest.main()
