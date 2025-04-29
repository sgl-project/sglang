import unittest

from sglang.test.test_utils import CustomTestCase, is_in_ci, run_bench_one_batch


class TestDummyGrok1(CustomTestCase):
    def test_dummy_grok_1(self):
        output_throughput = run_bench_one_batch(
            None,
            [
                "--model",
                "/dummy-grok",
                "--tokenizer-path",
                "Xenova/grok-1-tokenizer",
                "--batch-size",
                "2",
                "--tp",
                "2",
                "--quantization",
                "fp8",
                "--load-format",
                "dummy",
                "--json-model-override-args",
                '{"num_hidden_layers": 2}',
            ],
        )

        if is_in_ci():
            self.assertGreater(output_throughput, 0)


if __name__ == "__main__":
    unittest.main()
