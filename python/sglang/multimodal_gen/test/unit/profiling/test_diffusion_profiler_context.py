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
        self.npu_platform = self.enterContext(
            patch.object(
                pipeline_executor.current_platform, "is_npu", return_value=True
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

    def test_inference_error_or_cancel_skips_analysis(self):
        for error in (ValueError("model failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error)):
                self.profiler.reset_mock()
                with self.assertRaises(type(error)), self.context():
                    raise error
                self.profiler.stop.assert_called_once_with(
                    export_trace=False, dump_rank=0
                )

    def test_cleanup_error_does_not_replace_inference_error(self):
        self.profiler.stop.side_effect = RuntimeError("stop failed")
        with self.assertRaisesRegex(ValueError, "model failed"), self.context():
            raise ValueError("model failed")
        self.profiler.stop.assert_called_once_with(export_trace=False, dump_rank=0)

    def test_non_npu_error_cleanup_preserves_existing_export(self):
        self.npu_platform.return_value = False
        with self.assertRaisesRegex(ValueError, "model failed"), self.context():
            raise ValueError("model failed")
        self.profiler.stop.assert_called_once_with(export_trace=True, dump_rank=0)

    def test_warmup_and_unprofiled_requests_do_not_start_profiler(self):
        for profile, warmup in ((False, False), (True, True)):
            with self.subTest(profile=profile, warmup=warmup):
                self.batch.profile = profile
                self.batch.is_warmup = warmup
                with self.context():
                    pass
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
