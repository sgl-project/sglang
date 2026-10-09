"""Request identity and latency attribution must agree with native timing."""

import asyncio
import copy
import hashlib
import json
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.training_capture_benchmark_client import (
    RequestRecorder,
    summarize_requests,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@dataclass
class Input:
    extra_request_body: dict
    prompt: str = "synthetic input"


class TestRequestRecorder(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_requests_preserve_body_and_native_outputs(self):
        received = []
        second_finished = asyncio.Event()

        async def request(value, pbar):
            received.append(value)
            if len(received) == 1:
                await second_finished.wait()
            else:
                second_finished.set()
            return SimpleNamespace(
                start_time=100.0,
                latency=0.3,
                ttft=0.1,
                itl=[0.1, 0.1],
                prompt_len=16,
                output_len=3,
                success=True,
                error="",
                cached_tokens=0,
            )

        original = Input({"rid": "original", "sampling_params": {"temperature": 0}})
        recorder = RequestRecorder(request)
        outputs = await asyncio.gather(recorder(original), recorder(original))
        self.assertEqual(original.extra_request_body["rid"], "original")
        self.assertEqual(len({row.extra_request_body["rid"] for row in received}), 2)
        self.assertEqual([row["request_index"] for row in recorder.records], [1, 0])
        for row, value in zip(
            sorted(recorder.records, key=lambda x: x["request_index"]), received
        ):
            self.assertEqual(
                row["trace_id"],
                hashlib.sha256(value.extra_request_body["rid"].encode()).hexdigest(),
            )
            self.assertNotIn("prompt", row)
            self.assertEqual(
                value.extra_request_body["sampling_params"], {"temperature": 0}
            )
        self.assertTrue(all(output.latency == 0.3 for output in outputs))


class TestLatencySummary(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "request_index": index,
                "trace_id": str(index),
                "start_time": start,
                "latency": latency,
                "ttft": ttft,
                "itl": [0.001, 0.002],
                "output_len": 3,
                "success": True,
                "error": "",
            }
            for index, start, latency, ttft in (
                (0, 100.0, 0.3, 0.1),
                (1, 100.1, 0.5, 0.1),
                (2, 100.2, 0.6, 0.4),
            )
        ]

    def summarize(self, rows=None, published=None):
        return summarize_requests(
            self.rows if rows is None else rows,
            {"1"} if published is None else published,
            count=3,
            output_len=3,
        )

    def test_join_uses_identity_and_native_completion_latency(self):
        summary = self.summarize(list(reversed(self.rows)))
        self.assertEqual(summary["groups"]["published"]["requests"], 1)
        self.assertEqual(summary["groups"]["not_published"]["requests"], 2)
        self.assertAlmostEqual(summary["groups"]["published"]["p99_tpot_ms"], 200.0)
        self.assertAlmostEqual(summary["groups"]["all"]["p99_tpot_ms"], 198.0)
        self.assertEqual(summary["worst_tpot"][0]["request_index"], 1)
        self.assertTrue(summary["worst_tpot"][0]["published"])
        self.assertEqual(summary["worst_ttft"][0]["request_index"], 2)
        self.assertFalse(summary["worst_ttft"][0]["published"])
        self.assertAlmostEqual(summary["worst_ttft"][0]["start_offset_ms"], 200.0)

    def test_empty_publication_group_is_not_reported_as_zero_latency(self):
        self.assertEqual(
            self.summarize(published=set())["groups"]["published"], {"requests": 0}
        )

    def test_saved_publication_identities_reproduce_all_group_metrics(self):
        summary = self.summarize(published={"2", "0"})
        artifact = json.loads(json.dumps({"requests": self.rows, "summary": summary}))
        self.assertEqual(artifact["summary"]["published_trace_ids"], ["0", "2"])
        self.assertEqual(
            self.summarize(
                artifact["requests"],
                published=set(artifact["summary"]["published_trace_ids"]),
            ),
            summary,
        )

    def test_missing_duplicate_or_unknown_request_identity_fails(self):
        with self.assertRaises(ValueError):
            self.summarize(self.rows[:-1])
        with self.assertRaises(ValueError):
            self.summarize(published={"foreign"})
        for field in ("request_index", "trace_id"):
            rows = copy.deepcopy(self.rows)
            rows[1][field] = rows[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.summarize(rows)

    def test_failed_truncated_or_invalid_timing_is_rejected(self):
        for field, value in (
            ("success", False),
            ("error", "broken stream"),
            ("output_len", 2),
            ("ttft", 0),
            ("ttft", 0.8),
            ("latency", float("nan")),
            ("start_time", float("inf")),
            ("itl", [-1]),
        ):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.summarize(rows)


if __name__ == "__main__":
    unittest.main()
