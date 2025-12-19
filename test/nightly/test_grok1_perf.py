"""Nightly performance benchmark for Grok-1 MOE model.

This test benchmarks the Grok-1 model with FP8 quantization on 8 GPUs.
The model path can be configured via GROK1_MODEL_PATH environment variable.

Typical model paths:
- AMD quantized: amd/grok-1-W4A8KV8 (INT4 weights, FP8 activations)
- Standard: xai-org/grok-1

Example usage:
    GROK1_MODEL_PATH=/path/to/grok-1 python -m pytest test_grok1_perf.py -v
"""

import os
import unittest

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

# Default to AMD quantized model path, can be overridden via environment variable
GROK1_MODEL_PATH = os.environ.get("GROK1_MODEL_PATH", "amd/grok-1-W4A8KV8")
GROK1_TOKENIZER_PATH = os.environ.get("GROK1_TOKENIZER_PATH", "Xenova/grok-1-tokenizer")
PROFILE_DIR = "performance_profiles_grok1"


class TestNightlyGrok1Performance(unittest.TestCase):
    """Nightly performance benchmark for Grok-1 MOE model.

    Tests the Grok-1 model with FP8 quantization across different configurations:
    - basic: Standard TP=8 configuration with FP8 quantization
    """

    @classmethod
    def setUpClass(cls):
        cls.model = GROK1_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "1024"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # Define variant configurations for Grok-1
        # Grok-1 is a 314B parameter MOE model requiring 8 GPUs
        cls.variants = [
            {
                "name": "basic",
                "other_args": [
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--quantization",
                    "fp8",
                    "--mem-fraction-static",
                    "0.85",
                    "--tokenizer-path",
                    GROK1_TOKENIZER_PATH,
                    "--attention-backend",
                    "aiter",
                ],
                "env_vars": {
                    "RCCL_MSCCL_ENABLE": "0",
                    "SGLANG_USE_AITER": "1",
                    "SGLANG_INT4_WEIGHT": "1",
                },
            },
        ]

        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()

    def test_bench_one_batch(self):
        """Run benchmark across all configured variants."""
        failed_variants = []

        try:
            for variant_config in self.variants:
                with self.subTest(variant=variant_config["name"]):
                    results, success = self.runner.run_benchmark_for_model(
                        model_path=self.model,
                        batch_sizes=self.batch_sizes,
                        input_lens=self.input_lens,
                        output_lens=self.output_lens,
                        other_args=variant_config["other_args"],
                        variant=variant_config["name"],
                    )

                    if not success:
                        failed_variants.append(variant_config["name"])

                    self.runner.add_report(results)
        finally:
            self.runner.write_final_report()

        if failed_variants:
            raise AssertionError(
                f"Benchmark failed for {self.model} with the following variants: "
                f"{', '.join(failed_variants)}"
            )


if __name__ == "__main__":
    unittest.main()
