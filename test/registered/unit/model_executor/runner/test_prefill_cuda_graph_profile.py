"""Unit tests for prefill CUDA-graph capture trace naming — CPU-only.

Per-shape traces (``SGLANG_GRAPH_BATCH_CAPTURE``) are named
``{runner}_tokens_<n>[_<variant>]_rank<r>.json.gz``. The single-trace flag
``SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE`` takes precedence and disables the
per-shape schedule.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_CAPTURE_TRACE = "SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE"
_BATCH_CAPTURE = "SGLANG_GRAPH_BATCH_CAPTURE"


def _runner(*, is_draft_worker=False, profiler=None):
    runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
    runner.model_runner = SimpleNamespace(is_draft_worker=is_draft_worker)
    if profiler is not None:
        runner._profiler = profiler
    runner._graph_batch_capture_active = (
        PrefillCudaGraphRunner._graph_batch_capture_active.__get__(runner)
    )
    runner._set_profile_trace_tag = (
        PrefillCudaGraphRunner._set_profile_trace_tag.__get__(runner)
    )
    runner._profile_runner_name = PrefillCudaGraphRunner._profile_runner_name.__get__(
        runner
    )
    return runner


class TestPrefillProfileTraceTag(CustomTestCase):
    def test_token_count_only(self):
        runner = _runner(profiler=mock.Mock())
        runner._set_profile_trace_tag(128, None)
        self.assertEqual(runner._profile_trace_tag, "tokens_128")

    def test_chunked_prefix_variant_is_appended(self):
        runner = _runner(profiler=mock.Mock())
        runner._set_profile_trace_tag(64, "prefix_chunks_4")
        self.assertEqual(runner._profile_trace_tag, "tokens_64_prefix_chunks_4")

    def test_noop_without_scheduled_profiler(self):
        runner = _runner()
        runner._set_profile_trace_tag(32, "prefix_chunks_1")
        self.assertFalse(hasattr(runner, "_profile_trace_tag"))

    def test_draft_worker_prefixes_the_base_class_name(self):
        self.assertEqual(
            _runner(is_draft_worker=True)._profile_runner_name(),
            "DraftPrefillCudaGraphRunner",
        )
        self.assertEqual(
            _runner(is_draft_worker=False)._profile_runner_name(),
            "PrefillCudaGraphRunner",
        )


class TestPrefillGraphBatchCaptureGate(CustomTestCase):
    def _active(self, env):
        runner = _runner()
        with mock.patch.dict(os.environ, env, clear=False):
            for key in (_CAPTURE_TRACE, _BATCH_CAPTURE):
                if key not in env:
                    os.environ.pop(key, None)
            return runner._graph_batch_capture_active()

    def test_batch_capture_alone_is_active(self):
        self.assertTrue(self._active({_BATCH_CAPTURE: "1"}))

    def test_original_trace_flag_wins_when_both_are_set(self):
        self.assertFalse(self._active({_CAPTURE_TRACE: "1", _BATCH_CAPTURE: "1"}))

    def test_neither_flag_is_inactive(self):
        self.assertFalse(self._active({}))


if __name__ == "__main__":
    unittest.main()
