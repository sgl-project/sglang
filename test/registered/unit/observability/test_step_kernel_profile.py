import json
import unittest

from sglang.srt.observability.step_kernel_profile import (
    attribute_step_kernels,
    render_recent_steps,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def step(start, duration=10, pid=10, tid=11):
    return {
        "cat": "user_annotation",
        "ph": "X",
        "name": "step[DECODE bs=1]",
        "ts": start,
        "dur": duration,
        "pid": pid,
        "tid": tid,
    }


def launch(start, corr, name="cudaLaunchKernel", tid=11):
    return {
        "cat": "cuda_runtime",
        "ph": "X",
        "name": name,
        "ts": start,
        "dur": 1,
        "pid": 10,
        "tid": tid,
        "args": {"correlation": corr},
    }


def kernel(start, corr, stream=1):
    return {
        "cat": "kernel",
        "ph": "X",
        "name": "matmul",
        "ts": start,
        "dur": 2,
        "pid": 0,
        "tid": stream,
        "args": {"device": 0, "stream": stream, "correlation": corr},
    }


class TestStepKernels(unittest.TestCase):
    def test_overlap_links_launch_and_not_device_time(self):
        result = attribute_step_kernels(
            {
                "traceEvents": [
                    step(0),
                    launch(2, 1),
                    step(10),
                    launch(12, 2),
                    kernel(30, 1),
                    kernel(20, 2),
                ]
            }
        )
        self.assertEqual(result["matched_kernels"], 2)
        by_index = {s["step_index"]: s for s in result["steps"]}
        self.assertEqual(by_index[0]["device_start_ns"], 30000)
        self.assertEqual(by_index[1]["device_start_ns"], 20000)

    def test_replays_of_one_graph_are_distinct_and_multistream(self):
        result = attribute_step_kernels(
            {
                "traceEvents": [
                    step(0),
                    launch(2, 51, "cudaGraphLaunch"),
                    step(10),
                    launch(12, 52, "cudaGraphLaunch"),
                    kernel(20, 51),
                    kernel(21, 51, stream=2),
                    kernel(25, 52),
                ]
            }
        )
        self.assertEqual(result["matched_kernels"], 3)
        self.assertEqual(
            [s["kernel_busy_union_ns"] for s in result["steps"]], [3000, 2000]
        )

    def test_missing_duplicate_and_wrong_thread_proof_stays_unmatched(self):
        for events in (
            [step(0), kernel(2, 1)],
            [step(0), launch(2, 0), kernel(4, 0)],
            [step(0), launch(2, 1), launch(3, 1), kernel(4, 1)],
            [step(0), launch(2, 1, tid=99), kernel(4, 1)],
            [step(0, 20), step(1), launch(2, 1), kernel(4, 1)],
        ):
            with self.subTest(events=events):
                result = attribute_step_kernels({"traceEvents": events})
                self.assertEqual(result["matched_kernels"], 0)
                self.assertEqual(result["unmatched_kernels"], 1)

    def test_recent_64_and_byte_cap_preserve_whole_steps(self):
        events = []
        for i in range(70):
            events.extend(
                [step(i * 10), launch(i * 10 + 2, i + 1), kernel(i * 10 + 20, i + 1)]
            )
        a = attribute_step_kernels({"traceEvents": events})
        data = render_recent_steps(a, max_bytes=100000)
        doc = json.loads(data)
        self.assertEqual(doc["metadata"]["retained_steps"], 64)
        self.assertEqual(doc["metadata"]["evicted_for_count"], 6)
        self.assertEqual(doc["traceEvents"][0]["args"]["step_index"], 6)
        smaller = render_recent_steps(a, max_bytes=2000)
        self.assertLessEqual(len(smaller), 2000)
        self.assertGreater(json.loads(smaller)["metadata"]["evicted_for_bytes"], 0)
        self.assertEqual(len(json.loads(smaller)["traceEvents"]) % 2, 0)

    def test_exact_utf8_byte_boundary_and_oversized_newest_forward(self):
        events = []
        for i in range(3):
            k = kernel(i * 10 + 20, i + 1)
            k["name"] = "算子" * (300 if i == 2 else 1)
            events.extend([step(i * 10), launch(i * 10 + 2, i + 1), k])
        a = attribute_step_kernels({"traceEvents": events})
        full = render_recent_steps(a, max_bytes=10000)
        self.assertEqual(full, render_recent_steps(a, max_bytes=len(full)))
        clipped = json.loads(render_recent_steps(a, max_bytes=len(full) - 1))
        self.assertEqual(clipped["metadata"]["retained_steps"], 2)
        empty = json.loads(render_recent_steps(a, max_bytes=1000))
        self.assertEqual(empty["metadata"]["evicted_for_bytes"], 3)
        self.assertEqual(empty["traceEvents"], [])
        with self.assertRaises(ValueError):
            render_recent_steps(a, max_bytes=1)

    def test_retention_keeps_all_devices_of_one_forward(self):
        second_device = kernel(25, 2)
        second_device["args"]["device"] = 1
        result = attribute_step_kernels(
            {
                "traceEvents": [
                    step(0),
                    launch(2, 1),
                    launch(3, 2),
                    kernel(20, 1),
                    second_device,
                ]
            }
        )
        doc = json.loads(render_recent_steps(result, max_bytes=4000, max_steps=1))
        self.assertEqual(doc["metadata"]["retained_steps"], 1)
        self.assertEqual(doc["metadata"]["evicted_for_count"], 0)
        self.assertEqual(
            {e["pid"] for e in doc["traceEvents"] if e["cat"] == "kernel"}, {0, 1}
        )

    def test_malformed_annotations_and_kernel_names_are_skipped(self):
        bad_step = step(0)
        bad_step["name"] = None
        bad_launch = launch(2, 2)
        bad_launch["args"] = None
        bad_kernel = kernel(4, 1)
        bad_kernel.pop("name")
        result = attribute_step_kernels(
            {
                "traceEvents": [
                    step(0),
                    launch(2, 1),
                    bad_step,
                    bad_launch,
                    bad_kernel,
                ]
            }
        )
        self.assertEqual(result["skipped_kernel_records"], 1)
        self.assertEqual(result["matched_kernels"], 0)

    def test_no_match_exports_explicit_coverage(self):
        a = attribute_step_kernels({"traceEvents": [kernel(1, 1)]})
        doc = json.loads(render_recent_steps(a, max_bytes=1000))
        self.assertEqual(doc["metadata"]["retained_steps"], 0)
        self.assertEqual(doc["metadata"]["unmatched_kernels"], 1)
        self.assertIsNone(doc["metadata"]["dropped_records"])


if __name__ == "__main__":
    unittest.main()
