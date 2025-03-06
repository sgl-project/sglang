import unittest

from sglang.test.test_utils import is_in_ci, run_bench_one_batch


class TestTorchTP(unittest.TestCase):
    def test_torch_native_llama(self):
        output_throughput = run_bench_one_batch(
            "meta-llama/Meta-Llama-3-8B",
            [
                "--tp",
                "2",
                "--json-model-override-args",
                '{"architectures": ["TorchNativeLlamaForCausalLM"]}',
                "--disable-cuda-graph",
            ],
        )

        if is_in_ci():
            assert output_throughput > 0, f"{output_throughput=}"


if __name__ == "__main__":
    unittest.main()
