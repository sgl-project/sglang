"""Nightly performance benchmark for Grok models (Grok-1 and Grok-2).

This test benchmarks both Grok-1 and Grok-2 models with FP8 quantization on 8 GPUs.

Model paths can be configured via environment variables:
- GROK1_MODEL_PATH: Path to Grok-1 model (default: amd/grok-1-W4A8KV8)
- GROK1_TOKENIZER_PATH: Path to Grok-1 tokenizer (default: Xenova/grok-1-tokenizer)
- GROK2_MODEL_PATH: Path to Grok-2 model (default: xai-org/grok-2)
- GROK2_TOKENIZER_PATH: Path to Grok-2 tokenizer (default: alvarobartt/grok-2-tokenizer)

Example usage:
    python -m pytest test_grok_perf.py -v
"""

import os
import unittest
from typing import List

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import BenchmarkResult
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

# Register for AMD CI - combined Grok-1 + Grok-2 benchmark (~60 min)
register_amd_ci(est_time=3600, suite="nightly-perf-8-gpu-grok", nightly=True)


def generate_simple_markdown_report(results: List[BenchmarkResult]) -> str:
    """Generate a simplified markdown report without traces and cost columns."""
    model_header = results[0].model_path
    if results[0].run_name and results[0].run_name != "default":
        model_header += f" ({results[0].run_name})"

    gpu_config = os.getenv("GPU_CONFIG", "")
    if gpu_config:
        model_header += f" [{gpu_config}]"

    summary = f"### {model_header}\n"
    summary += "| batch size | input len | latency (s) | input throughput (tok/s) | output throughput (tok/s) | ITL (ms) |\n"
    summary += "| ---------- | --------- | ----------- | ------------------------ | ------------------------- | -------- |\n"

    for result in results:
        itl = 1 / (result.output_throughput / result.batch_size) * 1000
        summary += f"| {result.batch_size} | {result.input_len} | {result.latency:.2f} | {result.input_throughput:.2f} | {result.output_throughput:.2f} | {itl:.2f} |\n"

    return summary


# Model and tokenizer paths can be overridden via environment variables
GROK1_MODEL_PATH = os.environ.get("GROK1_MODEL_PATH", "amd/grok-1-W4A8KV8")
GROK1_TOKENIZER_PATH = os.environ.get("GROK1_TOKENIZER_PATH", "Xenova/grok-1-tokenizer")
GROK2_MODEL_PATH = os.environ.get("GROK2_MODEL_PATH", "xai-org/grok-2")
GROK2_TOKENIZER_PATH = os.environ.get(
    "GROK2_TOKENIZER_PATH", "alvarobartt/grok-2-tokenizer"
)
PROFILE_DIR = "performance_profiles_grok"


class TestNightlyGrokPerformance(unittest.TestCase):
    """Nightly performance benchmark for Grok models (Grok-1 and Grok-2).

    Tests both Grok-1 (314B MOE) and Grok-2 models with FP8 quantization on TP=8.
    Combined runtime: ~43 minutes (Grok-1: ~23min, Grok-2: ~20min)
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "1024"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # Define model configurations for both Grok-1 and Grok-2
        cls.models = [
            {
                "name": "grok1",
                "model_path": GROK1_MODEL_PATH,
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
            {
                "name": "grok2",
                "model_path": GROK2_MODEL_PATH,
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
        # Override full_report to remove traces help text
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_one_batch(self):
        """Run benchmark across all Grok models."""
        failed_models = []

        try:
            for model_config in self.models:
                with self.subTest(model=model_config["name"]):
                    # Set environment variables for this model
                    old_env = {}
                    for key, value in model_config.get("env_vars", {}).items():
                        old_env[key] = os.environ.get(key)
                        os.environ[key] = value
                        print(f"Setting env: {key}={value}")

                    try:
                        result_tuple = self.runner.run_benchmark_for_model(
                            model_path=model_config["model_path"],
                            batch_sizes=self.batch_sizes,
                            input_lens=self.input_lens,
                            output_lens=self.output_lens,
                            other_args=model_config["other_args"],
                            variant=model_config["name"],
                            extra_bench_args=["--trust-remote-code"],
                        )
                        results = result_tuple[0]
                        success = result_tuple[1]

                        if not success:
                            failed_models.append(model_config["name"])

                        # Use simplified report format without traces
                        if results:
                            self.runner.full_report += (
                                generate_simple_markdown_report(results) + "\n"
                            )
                    finally:
                        # Restore original environment
                        for key, value in old_env.items():
                            if value is None:
                                os.environ.pop(key, None)
                            else:
                                os.environ[key] = value
        finally:
            self.runner.write_final_report()

        if failed_models:
            raise AssertionError(
                f"Benchmark failed for the following models: {', '.join(failed_models)}"
            )


if __name__ == "__main__":
    unittest.main()
