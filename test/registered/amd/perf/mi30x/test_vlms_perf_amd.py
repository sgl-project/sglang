"""AMD Nightly performance benchmark for VLM models (2-GPU).

This test benchmarks Vision-Language Models on AMD MI30x/MI35x with 2 GPUs.

Registry: nightly-amd-perf-vlm-2-gpu suite

Example usage:
    python -m pytest test_vlms_perf_amd.py -v
"""

import os
import unittest
import warnings
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

# Register for AMD CI - VLM models benchmark (~120 min)
register_amd_ci(est_time=7200, suite="nightly-amd-perf-vlm-2-gpu", nightly=True)

PROFILE_DIR = "performance_profiles_vlms_amd"

# VLM models suitable for AMD
MODEL_DEFAULTS = [
    ModelLaunchSettings(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        extra_args=["--mem-fraction-static=0.7"],
    ),
    ModelLaunchSettings(
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        tp_size=2,
    ),
]


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


class TestNightlyVLMsPerfAMD(unittest.TestCase):
    """AMD Nightly performance benchmark for VLM models (2-GPU)."""

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )

        nightly_vlm_models_str = os.environ.get("NIGHTLY_VLM_MODELS")
        if nightly_vlm_models_str:
            cls.models = []
            model_paths = parse_models(nightly_vlm_models_str)
            for model_path in model_paths:
                cls.models.append(ModelLaunchSettings(model_path))
        else:
            cls.models = MODEL_DEFAULTS

        cls.base_url = DEFAULT_URL_FOR_TEST
        # First batch_size=1 is warmup (standalone job, no accuracy test to warm up)
        cls.batch_sizes = _parse_int_list_env("NIGHTLY_VLM_BATCH_SIZES", "1,1,2,8,16")
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_OUTPUT_LENS", "512"))
        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_one_batch(self):
        """Run benchmark for all configured VLM models."""
        all_model_succeed = True

        try:
            for model_setup in self.models:
                with self.subTest(model=model_setup.model_path):
                    other_args = list(model_setup.extra_args or [])
                    if model_setup.tp_size and model_setup.tp_size > 1:
                        other_args.extend(["--tp", str(model_setup.tp_size)])

                    # VLMs need additional benchmark args for dataset and trust-remote-code
                    extra_bench_args = [
                        "--trust-remote-code",
                        "--dataset-name=mmmu",
                    ]

                    result_tuple = self.runner.run_benchmark_for_model(
                        model_path=model_setup.model_path,
                        batch_sizes=self.batch_sizes,
                        input_lens=self.input_lens,
                        output_lens=self.output_lens,
                        other_args=other_args,
                        extra_bench_args=extra_bench_args,
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
