import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sglang.srt.environ import envs
from sglang.srt.observability.kernel_profile import (
    export_kernel_summary,
    summarize_profile,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def kernel(start, duration, device=0, stream=1):
    return {
        "cat": "kernel",
        "ph": "X",
        "ts": start,
        "dur": duration,
        "name": "test_kernel",
        "args": {"device": device, "stream": stream},
    }


class TestKernelProfile(unittest.TestCase):
    def test_multi_stream_union_is_clipped_and_separate_per_device(self):
        payload = {
            "traceEvents": [
                {"name": "PyTorch Profiler (0)", "ph": "X", "ts": 1, "dur": 9},
                kernel(0, 5),
                kernel(3, 4, stream=2),
                kernel(8, 8),
                kernel(0, 20, device=1),
                {"cat": "gpu_memcpy", "ph": "X", "ts": 0, "dur": 20},
            ]
        }
        result = summarize_profile(payload)
        a, b = result["devices"]
        self.assertEqual(a["kernel_busy_union_ns"], 8000)
        self.assertEqual(a["kernel_time_sum_in_window_ns"], 10000)
        self.assertEqual(a["uncovered_ns"], 1000)
        self.assertEqual(b["kernel_busy_union_ns"], 9000)
        self.assertIsNone(a["dropped_records"])

    def test_precision_and_bad_records(self):
        result = summarize_profile(
            {
                "traceEvents": [
                    kernel("3803905721017.743", "0.801"),
                    kernel(1, -2),
                    {"cat": "kernel", "ph": "X"},
                    "bad",
                ]
            }
        )
        self.assertEqual(result["devices"][0]["kernel_busy_union_ns"], 801)
        self.assertEqual(result["skipped_kernel_records"], 3)
        self.assertEqual(result["window_source"], "observed_kernel_envelope")

    def test_cpu_trace_is_an_empty_observation(self):
        self.assertEqual(summarize_profile({"traceEvents": []})["devices"], [])

    def test_truncated_gzip_summary_does_not_interrupt_profile_stop(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json.gz"
            with gzip.open(path, "wt") as f:
                json.dump({"traceEvents": [kernel(2, 5)]}, f)
            path.write_bytes(path.read_bytes()[:-3])
            with envs.SGLANG_PROFILE_KERNEL_SUMMARY.override(True):
                export_kernel_summary(path)
            self.assertFalse(path.with_name("broken.kernel-summary.json").exists())

    def test_export_is_opt_in_and_supports_compressed_trace(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.trace.json.gz"
            payload = {"traceEvents": [kernel(2, 5)]}
            with gzip.open(path, "wt") as f:
                json.dump(payload, f)
            with envs.SGLANG_PROFILE_KERNEL_SUMMARY.override(False):
                export_kernel_summary(path)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)
            with envs.SGLANG_PROFILE_KERNEL_SUMMARY.override(True):
                export_kernel_summary(path)
            summary = json.loads(
                path.with_name("profile.trace.kernel-summary.json").read_text()
            )
            self.assertEqual(summary["devices"][0]["kernel_busy_union_ns"], 5000)


if __name__ == "__main__":
    unittest.main()
