"""MI35x Nightly performance benchmark for Kimi-K2.6 model.

This test benchmarks moonshotai/Kimi-K2.6 with TP=8 on MI35x.

Kimi-K2.6 shares the same architecture as Kimi-K2.5 (per the model card the
deployment method is directly reused), so the AMD server arguments match the
existing Kimi-K2.5 MI35x accuracy test (mixed aiter prefill + triton decode).

The model path can be configured via KIMI_K26_MODEL_PATH environment variable.

Registry: nightly-perf-8-gpu-mi35x-kimi-k26 suite

Example usage:
    KIMI_K26_MODEL_PATH=moonshotai/Kimi-K2.6 python -m pytest test_kimi_k26_perf_mi35x.py -v
"""

import os

os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")

import unittest
from typing import List

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import BenchmarkResult
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

# Register for AMD CI - Kimi K2.6 perf benchmark on MI35x (~90 min)
register_amd_ci(est_time=5400, suite="nightly-perf-8-gpu-mi35x-kimi-k26", nightly=True)


def generate_simple_markdown_report(results: List[BenchmarkResult]) -> str:
    """Generate a simplified markdown report without traces and cost columns.

    Skips the first result if it's a warmup run (duplicate batch_size).
    """
    model_header = results[0].model_path
    if results[0].run_name and results[0].run_name != "default":
        model_header += f" ({results[0].run_name})"

    gpu_config = os.getenv("GPU_CONFIG", "MI35x")
    if gpu_config:
        model_header += f" [{gpu_config}]"

    summary = f"### {model_header}\n"
    summary += "| batch size | input len | latency (s) | input throughput (tok/s) | output throughput (tok/s) | ITL (ms) |\n"
    summary += "| ---------- | --------- | ----------- | ------------------------ | ------------------------- | -------- |\n"

    report_results = (
        results[1:]
        if len(results) > 1 and results[0].batch_size == results[1].batch_size
        else results
    )

    for result in report_results:
        itl = 1 / (result.output_throughput / result.batch_size) * 1000
        summary += f"| {result.batch_size} | {result.input_len} | {result.latency:.2f} | {result.input_throughput:.2f} | {result.output_throughput:.2f} | {itl:.2f} |\n"

    return summary


KIMI_K26_MODEL_PATH = os.environ.get("KIMI_K26_MODEL_PATH", "moonshotai/Kimi-K2.6")
PROFILE_DIR = "performance_profiles_kimi_k26_mi35x"


class TestNightlyKimiK26PerformanceMI35x(unittest.TestCase):
    """MI35x Nightly performance benchmark for Kimi-K2.6 model.

    Tests Kimi-K2.6 with TP=8 mixed-attention configuration on MI35x.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # Kimi-K2.6 shares Kimi-K2.5's architecture: aiter for prefill,
        # triton for decode (aiter ASM MLA decode requires heads_per_gpu % 16,
        # but TP=8 with 64 heads gives 8 heads/GPU, so we use triton decode).
        cls.model_config = {
            "name": "default",
            "model_path": KIMI_K26_MODEL_PATH,
            "other_args": [
                "--trust-remote-code",
                "--tp",
                "8",
                "--decode-attention-backend",
                "triton",
                "--prefill-attention-backend",
                "aiter",
                "--mem-fraction-static",
                "0.85",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true}',
                "--watchdog-timeout",
                "1200",
            ],
            "env_vars": {
                "SGLANG_USE_AITER": "1",
                "SGLANG_ROCM_FUSED_DECODE_MLA": "0",
            },
        }

        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_kimi_k26(self):
        """Run benchmark for Kimi-K2.6."""
        old_env = {}
        for key, value in self.model_config.get("env_vars", {}).items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            result_tuple = self.runner.run_benchmark_for_model(
                model_path=self.model_config["model_path"],
                batch_sizes=self.batch_sizes,
                input_lens=self.input_lens,
                output_lens=self.output_lens,
                other_args=self.model_config["other_args"],
                variant=self.model_config["name"],
                extra_bench_args=["--trust-remote-code"],
                enable_profile=False,
                timeout=5400,
            )
            results = result_tuple[0]
            success = result_tuple[1]

            if results:
                self.runner.full_report += (
                    generate_simple_markdown_report(results) + "\n"
                )

            self.assertTrue(
                success, f"Benchmark failed for {self.model_config['model_path']}"
            )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            self.runner.write_final_report()


if __name__ == "__main__":
    unittest.main()
