import asyncio
import unittest
from unittest.mock import patch

from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.serving import get_request
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestFinalBenchmarkArrival(CustomTestCase):
    def test_no_arrival_gap_after_the_final_request(self):
        """A single-request benchmark has no following arrival to delay."""
        row = DatasetRow(prompt="hello", prompt_len=1, output_len=1)

        async def collect():
            return [item async for item in get_request([row], request_rate=1e-6)]

        with patch(
            "sglang.benchmark.serving.asyncio.sleep",
            side_effect=AssertionError("idle tail"),
        ):
            self.assertEqual(asyncio.run(collect()), [row])


if __name__ == "__main__":
    unittest.main()
