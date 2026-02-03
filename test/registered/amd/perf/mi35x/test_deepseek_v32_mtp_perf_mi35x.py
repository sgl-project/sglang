"""MI35x Nightly performance benchmark for DeepSeek-V3.2 model (MTP variant).

This test benchmarks the DeepSeek-V3.2 model with MTP (EAGLE speculative decoding)
configuration on 8 GPUs.

The model path can be configured via DEEPSEEK_V32_MODEL_PATH environment variable.

Registry: nightly-perf-8-gpu-mi35x-deepseek-v32-mtp suite

Example usage:
    DEEPSEEK_V32_MODEL_PATH=deepseek-ai/DeepSeek-V3.2 python -m pytest test_deepseek_v32_mtp_perf_mi35x.py -v
"""

import os
import unittest
from typing import List, Optional, Tuple

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import BenchmarkResult
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    _parse_int_list_env,
    popen_launch_server,
)

# Register for AMD CI - DeepSeek-V3.2 MTP benchmark (~90 min)
register_amd_ci(
    est_time=5400, suite="nightly-perf-8-gpu-mi35x-deepseek-v32-mtp", nightly=True
)


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


def _run_benchmark_with_timeout(
    runner: NightlyBenchmarkRunner,
    model_path: str,
    batch_sizes: List[int],
    input_lens: Tuple[int, ...],
    output_lens: Tuple[int, ...],
    other_args: List[str],
    variant: str,
    extra_bench_args: Optional[List[str]],
    timeout: int,
) -> Tuple[List[BenchmarkResult], bool, Optional[float]]:
    """Run benchmark with a custom server launch timeout."""
    model_description = f"{model_path}" + (f" ({variant})" if variant else "")
    process = popen_launch_server(
        model=model_path,
        base_url=runner.base_url,
        other_args=other_args,
        timeout=timeout,
    )
    try:
        profile_path_prefix, json_output_file = runner.generate_profile_filename(
            model_path, variant
        )
        bench_args = list(extra_bench_args) if extra_bench_args else []
        if variant:
            bench_args.extend(["--run-name", variant])
        command = runner.build_benchmark_command(
            model_path,
            batch_sizes,
            input_lens,
            output_lens,
            profile_path_prefix,
            json_output_file,
            extra_args=bench_args,
        )
        _, cmd_success = runner.run_benchmark_command(command, model_description)
        if not cmd_success:
            return [], False, None
        benchmark_results, load_success = runner.load_benchmark_results(
            json_output_file, model_description
        )
        return benchmark_results, load_success, None
    finally:
        kill_process_tree(process.pid)


# Model path can be overridden via environment variable
DEEPSEEK_V32_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V32_MODEL_PATH", "deepseek-ai/DeepSeek-V3.2"
)
PROFILE_DIR = "performance_profiles_deepseek_v32_mtp"
SERVER_LAUNCH_TIMEOUT = 5400


class TestNightlyDeepseekV32MTPPerformance(unittest.TestCase):
    """MI35x Nightly performance benchmark for DeepSeek-V3.2 model (MTP variant).

    Tests the DeepSeek-V3.2 model with MTP (EAGLE speculative decoding) on TP=8.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V32_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # MTP variant configuration for DeepSeek-V3.2
        # MI35x uses tilelang NSA backends + EAGLE speculative decoding
        cls.variant_config = {
            "name": "mtp",
            "other_args": [
                "--trust-remote-code",
                "--tp",
                "8",
                "--nsa-prefill-backend",
                "tilelang",
                "--nsa-decode-backend",
                "tilelang",
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
                "--watchdog-timeout",
                "1200",
            ],
        }

        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()
        # Override full_report to remove traces help text
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_one_batch(self):
        """Run benchmark for MTP variant."""
        try:
            result_tuple = _run_benchmark_with_timeout(
                runner=self.runner,
                model_path=self.model,
                batch_sizes=self.batch_sizes,
                input_lens=self.input_lens,
                output_lens=self.output_lens,
                other_args=self.variant_config["other_args"],
                variant=self.variant_config["name"],
                extra_bench_args=["--trust-remote-code"],
                timeout=SERVER_LAUNCH_TIMEOUT,
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
                raise AssertionError(f"Benchmark failed for {self.model} (MTP variant)")
        finally:
            self.runner.write_final_report()


if __name__ == "__main__":
    unittest.main()
