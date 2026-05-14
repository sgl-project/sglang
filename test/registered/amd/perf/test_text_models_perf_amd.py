"""AMD Nightly performance benchmark for text models (2-GPU).

This test benchmarks text models on AMD MI30x/MI35x with 2 GPUs.

Registry: nightly-amd-perf-text-2-gpu suite

Example usage:
    python -m pytest test_text_models_perf_amd.py -v
"""

import os
import unittest
from typing import List

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import BenchmarkResult
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    ModelLaunchSettings,
    _parse_int_list_env,
    parse_models,
)

# Register for AMD CI - Text models benchmark (~60 min)
register_amd_ci(est_time=3600, suite="nightly-amd-perf-text-2-gpu", nightly=True)

PROFILE_DIR = "performance_profiles_text_models_amd"


def generate_simple_markdown_report(results: List[BenchmarkResult]) -> str:
    """Generate a simplified markdown report without traces and cost columns.

    Skips the first result if it's a warmup run (duplicate batch_size).
    """
    model_header = results[0].model_path
    if results[0].run_name and results[0].run_name != "default":
        model_header += f" ({results[0].run_name})"

    gpu_config = os.getenv("GPU_CONFIG", "AMD")
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


class TestNightlyTextModelsPerfAMD(unittest.TestCase):
    """AMD Nightly performance benchmark for text models (2-GPU)."""

    @classmethod
    def setUpClass(cls):
        cls.models = []
        # Llama-3.1-8B on TP=1
        for model_path in parse_models("meta-llama/Llama-3.1-8B-Instruct"):
            cls.models.append(
                ModelLaunchSettings(
                    model_path,
                    tp_size=1,
                    extra_args=["--attention-backend", "aiter"],
                )
            )
        # Qwen2-57B MoE on TP=2
        for model_path in parse_models("Qwen/Qwen2-57B-A14B-Instruct"):
            cls.models.append(
                ModelLaunchSettings(
                    model_path,
                    tp_size=2,
                    extra_args=["--attention-backend", "aiter"],
                )
            )

        cls.base_url = DEFAULT_URL_FOR_TEST
        # First batch_size=1 is warmup (standalone job, no accuracy test to warm up)
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))
        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_one_batch(self):
        """Run benchmark for all configured text models."""
        all_model_succeed = True

        try:
            for model_setup in self.models:
                with self.subTest(model=model_setup.model_path):
                    other_args = list(model_setup.extra_args or [])
                    if model_setup.tp_size and model_setup.tp_size > 1:
                        other_args.extend(["--tp", str(model_setup.tp_size)])

                    result_tuple = self.runner.run_benchmark_for_model(
                        model_path=model_setup.model_path,
                        batch_sizes=self.batch_sizes,
                        input_lens=self.input_lens,
                        output_lens=self.output_lens,
                        other_args=other_args,
                    )
                    results = result_tuple[0]
                    success = result_tuple[1]

                    if not success:
                        all_model_succeed = False

                    if results:
                        self.runner.full_report += (
                            generate_simple_markdown_report(results) + "\n"
                        )
        finally:
            self.runner.write_final_report()

        if not all_model_succeed:
            raise AssertionError("Some models failed the perf tests.")


if __name__ == "__main__":
    unittest.main()
