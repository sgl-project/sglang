"""Nightly performance benchmark for Grok-2 model.

This test benchmarks the Grok-2 model with FP8 quantization on 8 GPUs.
The model path can be configured via GROK2_MODEL_PATH environment variable.
A separate tokenizer path can be specified via GROK2_TOKENIZER_PATH.

Run via AMD CI with:
    AMD_TEST_MODEL_GROUP=grok2-perf python3 run_suite.py --suite nightly-amd-8-gpu

Example usage:
    GROK2_MODEL_PATH=/path/to/grok-2 python -m pytest test_grok2_perf.py -v
"""

import os
import unittest

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

# Register with AMD CI - 2 hour estimated time for Grok-2 perf test
register_amd_ci(est_time=7200, suite="nightly-amd-8-gpu", nightly=True)

# Model and tokenizer paths can be overridden via environment variables
GROK2_MODEL_PATH = os.environ.get("GROK2_MODEL_PATH", "xai-org/grok-2")
GROK2_TOKENIZER_PATH = os.environ.get(
    "GROK2_TOKENIZER_PATH", "alvarobartt/grok-2-tokenizer"
)
PROFILE_DIR = "performance_profiles_grok2"


def get_model_group() -> str:
    """Get the model group to test from environment variable."""
    return os.environ.get("AMD_TEST_MODEL_GROUP", "")


def should_run_test() -> bool:
    """Check if this test should run based on AMD_TEST_MODEL_GROUP."""
    group = get_model_group()
    # Run if group is grok2-perf, all, or empty (direct invocation)
    return group in ("grok2-perf", "all", "")


@unittest.skipUnless(should_run_test(), "Skipping: AMD_TEST_MODEL_GROUP != grok2-perf")
class TestNightlyGrok2Performance(unittest.TestCase):
    """Nightly performance benchmark for Grok-2 model.

    Tests the Grok-2 model with FP8 quantization across different configurations:
    - basic: Standard TP=8 configuration with FP8 quantization
    """

    @classmethod
    def setUpClass(cls):
        cls.model = GROK2_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "1024"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # Define variant configurations for Grok-2
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
                    GROK2_TOKENIZER_PATH,
                    "--attention-backend",
                    "aiter",
                ],
                "env_vars": {
                    "RCCL_MSCCL_ENABLE": "0",
                    "SGLANG_USE_AITER": "1",
                    "SGLANG_INT4_WEIGHT": "0",
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
