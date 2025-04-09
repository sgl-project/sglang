import unittest

from sglang.test.test_utils import CustomTestCase, is_in_ci, run_bench_one_batch


class TestDummyGrok1(CustomTestCase):
    def test_dummy_grok_1(self):
        output_throughput = run_bench_one_batch(
            "/dummy-grok",
            [
                "--batch-size",
                "32",
                "--input",
                "1024",
                "--output",
                "8",
                "--tokenizer-path",
                "Xenova/grok-1-tokenizer",
                "--tp",
                "8",
                "--quantization",
                "fp8",
                "--load-format",
                "dummy",
            ],
        )

        if is_in_ci():
            assert output_throughput > 0, f"{output_throughput=}"


if __name__ == "__main__":
    unittest.main()
