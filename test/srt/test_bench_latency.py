import unittest

from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MOE_MODEL_NAME_FOR_TEST,
    is_in_ci,
    run_bench_latency,
)


class TestBenchLatency(unittest.TestCase):
    def test_default(self):
        output_throughput = run_bench_latency(DEFAULT_MODEL_NAME_FOR_TEST, [])

        if is_in_ci():
            self.assertGreater(output_throughput, 135)

    def test_moe_default(self):
        output_throughput = run_bench_latency(
            DEFAULT_MOE_MODEL_NAME_FOR_TEST, ["--tp", "2"]
        )

        if is_in_ci():
            self.assertGreater(output_throughput, 125)


if __name__ == "__main__":
    unittest.main()
