"""AMD Nightly performance benchmark for DeepSeek-V3.2 model (basic variant).

This test benchmarks the DeepSeek-V3.2 model with basic TP=8 configuration on 8 GPUs.

The model path can be configured via DEEPSEEK_V32_MODEL_PATH environment variable.

Registry: nightly-perf-8-gpu-deepseek-v32-basic suite

Example usage:
    DEEPSEEK_V32_MODEL_PATH=deepseek-ai/DeepSeek-V3.2 python -m pytest test_deepseek_v32_basic_perf_amd.py -v
"""

import os
import unittest
from typing import List

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import BenchmarkResult
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

# Register for AMD CI - DeepSeek-V3.2 basic benchmark (~90 min)
register_amd_ci(
    est_time=5400, suite="nightly-perf-8-gpu-deepseek-v32-basic", nightly=True
)


def generate_simple_markdown_report(results: List[BenchmarkResult]) -> str:
    """Generate a simplified markdown report without traces and cost columns.

    Skips the first result if it's a warmup run (duplicate batch_size).
    """
    model_header = results[0].model_path
    if results[0].run_name and results[0].run_name != "default":
        model_header += f" ({results[0].run_name})"

    gpu_config = os.getenv("GPU_CONFIG", "MI325")
    if gpu_config:
        model_header += f" [{gpu_config}]"

    summary = f"### {model_header}\n"
    summary += "| batch size | input len | latency (s) | input throughput (tok/s) | output throughput (tok/s) | ITL (ms) |\n"
    summary += "| ---------- | --------- | ----------- | ------------------------ | ------------------------- | -------- |\n"

    # Skip first result if it's a warmup (same batch_size as second result)
    report_results = (
        results[1:]
        if len(results) > 1 and results[0].batch_size == results[1].batch_size
        else results
    )

    for result in report_results:
        itl = 1 / (result.output_throughput / result.batch_size) * 1000
        summary += f"| {result.batch_size} | {result.input_len} | {result.latency:.2f} | {result.input_throughput:.2f} | {result.output_throughput:.2f} | {itl:.2f} |\n"

    return summary


# Model path can be overridden via environment variable
DEEPSEEK_V32_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V32_MODEL_PATH", "deepseek-ai/DeepSeek-V3.2"
)
PROFILE_DIR = "performance_profiles_deepseek_v32_basic_mi325"


class TestNightlyDeepseekV32BasicPerformance(unittest.TestCase):
    """AMD Nightly performance benchmark for DeepSeek-V3.2 model (basic variant).

    Tests the DeepSeek-V3.2 model with basic TP=8 configuration on MI325/MI300X.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V32_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # Basic variant configuration for DeepSeek-V3.2
        # MI325 uses aiter attention backend
        cls.variant_config = {
            "name": "basic",
            "other_args": [
                "--trust-remote-code",
                "--tp",
                "8",
                "--attention-backend",
                "aiter",
                "--chunked-prefill-size",
                "131072",
                "--mem-fraction-static",
                "0.85",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true}',
                "--watchdog-timeout",
                "1200",
            ],
            "env_vars": {"SGLANG_USE_AITER": "1"},
        }

        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()
        # Override full_report to remove traces help text
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_one_batch(self):
        """Run benchmark for basic variant."""
        try:
            result_tuple = self.runner.run_benchmark_for_model(
                model_path=self.model,
                batch_sizes=self.batch_sizes,
                input_lens=self.input_lens,
                output_lens=self.output_lens,
                other_args=self.variant_config["other_args"],
                variant=self.variant_config["name"],
                extra_bench_args=["--trust-remote-code"],
            )
            results = result_tuple[0]
            success = result_tuple[1]
            avg_spec_accept_length = result_tuple[2] if len(result_tuple) > 2 else None

            # Log speculative decoding accept length
            if avg_spec_accept_length is not None:
                print(f"  avg_spec_accept_length={avg_spec_accept_length:.2f}")

            # Use simplified report format without traces
            if results:
                self.runner.full_report += (
                    generate_simple_markdown_report(results) + "\n"
                )

            if not success:
                raise AssertionError(
                    f"Benchmark failed for {self.model} (basic variant)"
                )
        finally:
            self.runner.write_final_report()


if __name__ == "__main__":
    unittest.main()
