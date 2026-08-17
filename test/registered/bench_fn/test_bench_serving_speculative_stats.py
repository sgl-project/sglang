"""Unit coverage for benchmark collection of per-request spec decoding stats."""

from sglang.benchmark.serving import (
    RequestFuncOutput,
    collect_speculative_decoding_stats,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSpeculativeDecodingStatsCollection(CustomTestCase):
    def test_collection_keeps_request_alignment_and_missing_entries(self):
        with_stats = RequestFuncOutput(success=True)
        with_stats.speculative_decoding_stats = {
            "schema_version": 1,
            "mode": "detailed",
            "verify_lengths": [6, 5],
            "accept_lengths": [4, 3],
        }
        failed = RequestFuncOutput(success=False, error="boom")
        missing = RequestFuncOutput(success=True)

        collected = collect_speculative_decoding_stats([with_stats, failed, missing])

        self.assertEqual(collected["num_requests"], 3)
        self.assertEqual(collected["num_requests_with_stats"], 1)
        self.assertEqual(
            [request["request_index"] for request in collected["requests"]],
            [0, 1, 2],
        )
        self.assertIsNone(collected["requests"][1]["stats"])
        self.assertIsNone(collected["requests"][2]["stats"])

    def test_collection_is_omitted_when_server_returns_no_stats(self):
        self.assertIsNone(
            collect_speculative_decoding_stats([RequestFuncOutput(success=True)])
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
