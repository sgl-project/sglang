"""Nightly performance benchmark for DeepSeek-V3 model.

This test benchmarks the DeepSeek-V3 model with multiple configurations on 8 GPUs:
- basic: Standard TP=8 configuration
- mtp: Multi-Token Prediction with EAGLE speculative decoding
- dp: Data Parallel attention mode
- dp+mtp: Data Parallel + MTP combined

The model path can be configured via DEEPSEEK_V3_MODEL_PATH environment variable.

Example usage:
    DEEPSEEK_V3_MODEL_PATH=deepseek-ai/DeepSeek-V3-0324 python -m pytest test_deepseek_v3_perf.py -v
"""

import os
import unittest

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

# Model path can be overridden via environment variable
DEEPSEEK_V3_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V3_MODEL_PATH", "deepseek-ai/DeepSeek-V3-0324"
)
PROFILE_DIR = "performance_profiles_deepseek_v3"


class TestNightlyDeepseekV3Performance(unittest.TestCase):
    """Nightly performance benchmark for DeepSeek-V3 model.

    Tests the DeepSeek-V3 model across different configurations:
    - basic: Standard TP=8 configuration
    - mtp: Multi-Token Prediction with EAGLE speculative decoding
    - dp: Data Parallel attention with DP=8
    - dp+mtp: Combined DP attention and MTP
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # Define variant configurations for DeepSeek-V3
        # Environment variables for ROCm 7.0+ support
        deepseek_env = {
            "SGLANG_USE_ROCM700A": "1",
            "SGLANG_USE_AITER": "1",
        }

        cls.variants = [
            {
                "name": "basic",
                "other_args": [
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--mem-fraction-static",
                    "0.85",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
                "env_vars": deepseek_env,
            },
            {
                "name": "mtp",
                "other_args": [
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--mem-fraction-static",
                    "0.7",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
                "env_vars": deepseek_env,
            },
            {
                "name": "dp",
                "other_args": [
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--dp",
                    "8",
                    "--enable-dp-attention",
                    "--chunked-prefill-size",
                    "131072",
                    "--mem-fraction-static",
                    "0.85",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
                "env_vars": deepseek_env,
            },
            {
                "name": "dp+mtp",
                "other_args": [
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--dp",
                    "8",
                    "--enable-dp-attention",
                    "--chunked-prefill-size",
                    "131072",
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--mem-fraction-static",
                    "0.7",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                ],
                "env_vars": deepseek_env,
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
                        extra_bench_args=["--trust-remote-code"],
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
