"""MI35x Nightly performance benchmark for Grok-1 INT4 (W4A8KV8).

This test benchmarks Grok-1 (314B MOE) with INT4 weight quantization on 8 GPUs.

Registry: nightly-perf-8-gpu-mi35x-grok1-int4 suite
"""

import os
import unittest
from typing import List

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import BenchmarkResult
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

# Register for AMD CI - Grok-1 INT4 benchmark on MI35x (~25 min)
register_amd_ci(
    est_time=1500, suite="nightly-perf-8-gpu-mi35x-grok1-int4", nightly=True
)


def generate_simple_markdown_report(results: List[BenchmarkResult]) -> str:
    """Generate a simplified markdown report without traces and cost columns."""
    model_header = results[0].model_path
    if results[0].run_name and results[0].run_name != "default":
        model_header += f" ({results[0].run_name})"

    gpu_config = os.getenv("GPU_CONFIG", "MI35x")
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
PROFILE_DIR = "performance_profiles_grok1_int4_mi35x"


class TestGrok1INT4PerfMI35x(unittest.TestCase):
    """Test suite for Grok-1 INT4 performance benchmarks on MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.model = GROK1_MODEL_PATH
        cls.tokenizer = GROK1_TOKENIZER_PATH

    def test_grok1_int4_perf(self):
        """Run Grok-1 INT4 performance benchmark on MI35x."""
        batch_sizes = _parse_int_list_env("BATCH_SIZES", [1, 8, 16, 32])
        input_lens = _parse_int_list_env("INPUT_LENS", [1024, 4096])
        output_len = int(os.getenv("OUTPUT_LEN", "256"))

        env = os.environ.copy()
        env["RCCL_MSCCL_ENABLE"] = "0"
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_INT4_WEIGHT"] = "1"

        other_args = [
            "--tp",
            "8",
            "--quantization",
            "fp8",
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.85",
            "--tokenizer-path",
            self.tokenizer,
            "--trust-remote-code",
        ]

        runner = NightlyBenchmarkRunner(
            base_url=self.base_url,
            model=self.model,
            batch_sizes=batch_sizes,
            input_lens=input_lens,
            output_len=output_len,
            other_args=other_args,
            env=env,
            profile_dir=PROFILE_DIR,
            timeout=3600,
        )

        results = runner.run_benchmarks()
        self.assertGreater(len(results), 0, "No benchmark results")

        summary = generate_simple_markdown_report(results)
        runner.write_github_summary(summary)


if __name__ == "__main__":
    unittest.main()
