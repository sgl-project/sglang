"""Device activity follows launch correlation, not CPU/GPU time overlap."""

import json
import runpy
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCaptureTraceAttribution(CustomTestCase):
    def setUp(self):
        script = (
            Path(__file__).resolve().parents[4]
            / "mooncake-study/experiments/summarize_capture_trace.py"
        )
        self.summarize = runpy.run_path(str(script))["summarize"]
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "trace.json"

    def event(self, name, category, timestamp, duration, *, thread=1, **args):
        return {
            "name": name,
            "cat": category,
            "ph": "X",
            "ts": timestamp,
            "dur": duration,
            "pid": 1,
            "tid": thread,
            "args": args,
        }

    def evaluate(self, events):
        self.path.write_text(json.dumps({"traceEvents": events}))
        return self.summarize(self.path)

    def test_async_work_outlives_scope_and_other_thread_is_not_attributed(self):
        result = self.evaluate(
            [
                self.event("training_capture.kv", "user_annotation", 10, 10),
                self.event(
                    "training_capture.kv", "gpu_user_annotation", 30, 6, thread=9
                ),
                self.event("cudaLaunchKernel", "cuda_runtime", 12, 1, correlation=1),
                self.event("cudaMemcpyAsync", "cuda_runtime", 14, 1, correlation=2),
                self.event(
                    "cudaLaunchKernel", "cuda_runtime", 12, 1, thread=2, correlation=3
                ),
                self.event("capture_gather", "kernel", 30, 4, correlation=1),
                self.event(
                    "Memcpy DtoH (Device -> Pinned)",
                    "gpu_memcpy",
                    35,
                    2,
                    correlation=2,
                    bytes=2048,
                ),
                self.event("unrelated", "kernel", 12, 100, correlation=3),
            ]
        )
        capture = result["groups"]["training_capture.kv"]
        self.assertEqual(capture["cpu_calls"], 1)
        self.assertEqual(capture["cpu_scope_us"], 10)
        self.assertEqual(capture["gpu_kernel_us"], 4)
        self.assertEqual(capture["gpu_kernel_calls"], 1)
        self.assertEqual(capture["d2h_us"], 2)
        self.assertEqual(capture["d2h_bytes"], 2048)
        self.assertEqual(result["all_kernel_us"], 104)

    def test_missing_bytes_are_unknown_instead_of_zero(self):
        result = self.evaluate(
            [
                self.event("training_capture.teacher_d2h", "user_annotation", 10, 10),
                self.event("cudaMemcpyAsync", "cuda_runtime", 14, 1, correlation=1),
                self.event(
                    "Memcpy DtoH (Device -> Pinned)", "gpu_memcpy", 35, 2, correlation=1
                ),
            ]
        )
        group = result["groups"]["training_capture.teacher_d2h"]
        self.assertIsNone(group["d2h_bytes"])
        self.assertEqual(group["d2h_events_missing_bytes"], 1)

    def test_ambiguous_scope_and_correlation_are_rejected(self):
        scope = self.event("training_capture.kv", "user_annotation", 10, 10)
        for extra in (
            [self.event("training_capture.teacher", "user_annotation", 15, 10)],
            [
                self.event("cudaLaunchKernel", "cuda_runtime", 12, 1, correlation=1),
                self.event("training_capture.teacher", "user_annotation", 25, 10),
                self.event("cudaLaunchKernel", "cuda_runtime", 27, 1, correlation=1),
            ],
        ):
            with self.subTest(events=extra), self.assertRaises(ValueError):
                self.evaluate([scope, *extra])


if __name__ == "__main__":
    unittest.main()
