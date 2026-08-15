import unittest

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    run_bench_offline_throughput,
)


class TestTorchTP(CustomTestCase):
    def test_torch_native_llama(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            [
                "--tp",
                "2",
                # This cannot run anymore with the new torch version.
                # "--json-model-override-args",
                # '{"architectures": ["TorchNativeLlamaForCausalLM"]}',
                "--disable-cuda-graph",
            ],
        )

        if is_in_ci():
            self.assertGreater(output_throughput, 0)


if __name__ == "__main__":
    unittest.main()
