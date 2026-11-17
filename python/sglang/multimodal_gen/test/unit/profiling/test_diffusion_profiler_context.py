# SPDX-License-Identifier: Apache-2.0
"""Request cleanup stops profiling without masking inference errors."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.pipelines_core.executors import pipeline_executor


class TestDiffusionProfilerContext(unittest.TestCase):
    def setUp(self):
        self.factory = self.enterContext(
            patch.object(pipeline_executor, "SGLDiffusionProfiler")
        )
        self.profiler = self.factory.return_value
        self.enterContext(
            patch.object(
                pipeline_executor.current_platform,
                "is_npu",
                side_effect=AssertionError(
                    "The executor must not select profiler backends"
                ),
            )
        )
        self.enterContext(
            patch.object(pipeline_executor, "get_world_rank", return_value=0)
        )
        self.batch = SimpleNamespace(
            profile=True,
            is_warmup=False,
            request_id="request",
            profile_all_stages=True,
            num_profiled_timesteps=2,
            num_inference_steps=2,
        )

    def context(self):
        # The profiling context is independent of executor scheduling state.
        return pipeline_executor.PipelineExecutor.profile_execution(None, self.batch)

    def test_request_cleanup_uses_stop_without_a_separate_finish(self):
        with self.context():
            self.profiler.stop.assert_not_called()
        self.profiler.stop.assert_called_once_with(dump_rank=0)

    def test_capture_stop_failure_propagates(self):
        self.profiler.stop.side_effect = RuntimeError("capture stop failed")
        with (
            self.assertRaisesRegex(RuntimeError, "capture stop failed"),
            self.context(),
        ):
            pass

    def test_inference_error_or_cancel_delegates_cleanup(self):
        for error in (ValueError("model failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error)):
                self.profiler.reset_mock()
                with self.assertRaises(type(error)), self.context():
                    raise error
                self.profiler.stop_on_error.assert_called_once_with(dump_rank=0)
                self.profiler.stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
