"""MI355X nightly performance benchmark for MiMo-V2-Flash (4-GPU).

Benchmarks MiMo-V2-Flash with the aiter attention backend on MI355X using TP=4.
MiMo-V2-Flash has non-square attention (qk_head_dim=192, v_head_dim=128) with a
128-token sliding window; this benchmark validates throughput after the fallback
paths introduced for the aiter backend.

The model path can be configured via MIMO_V2_FLASH_MODEL_PATH environment variable.

Registry: nightly-perf-8-gpu-mi355x-mimo-v2-flash suite

Example usage:
    python -m pytest test_mimo_v2_flash_perf_mi35x.py -v
"""

import os

os.environ.setdefault("HF_HOME", "/mnt/hf_hub_cache")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/hf_hub_cache")

import unittest
from typing import List

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import BenchmarkResult
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

register_amd_ci(
    est_time=5400, suite="nightly-perf-8-gpu-mi355x-mimo-v2-flash", nightly=True
)

MIMO_V2_FLASH_MODEL_PATH = os.environ.get(
    "MIMO_V2_FLASH_MODEL_PATH", "/mnt/hf_hub_cache/MiMo-V2-Flash"
)
PROFILE_DIR = "performance_profiles_mimo_v2_flash_mi355x"


def generate_simple_markdown_report(results: List[BenchmarkResult]) -> str:
    """Generate a markdown table from benchmark results.

    Skips the first entry if it is a warmup run (same batch_size as second entry).
    """
    model_header = results[0].model_path
    if results[0].run_name and results[0].run_name != "default":
        model_header += f" ({results[0].run_name})"

    gpu_config = os.getenv("GPU_CONFIG", "MI355X")
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
        summary += (
            f"| {result.batch_size} | {result.input_len} | {result.latency:.2f} "
            f"| {result.input_throughput:.2f} | {result.output_throughput:.2f} "
            f"| {itl:.2f} |\n"
        )

    return summary


class TestNightlyMiMoV2FlashPerformanceMI355x(unittest.TestCase):
    """MI355X nightly performance benchmark for MiMo-V2-Flash.

    Tests MiMo-V2-Flash with TP=4 and the aiter attention backend.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        cls.model_config = {
            "name": "mimo-v2-flash-tp4-aiter",
            "model_path": MIMO_V2_FLASH_MODEL_PATH,
            "other_args": [
                "--trust-remote-code",
                "--tp",
                "4",
                "--attention-backend",
                "aiter",
                "--disable-radix-cache",
                "--mem-fraction-static",
                "0.8",
                "--chunked-prefill-size",
                "131072",
                "--max-running-requests",
                "128",
                "--watchdog-timeout",
                "1200",
            ],
            "env_vars": {},
        }

        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_mimo_v2_flash(self):
        """Run throughput benchmark for MiMo-V2-Flash on MI355X."""
        old_env = {}
        for key, value in self.model_config.get("env_vars", {}).items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
            print(f"Setting env: {key}={value}")

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

            self.assertTrue(success, "Benchmark failed for MiMo-V2-Flash on MI355X")
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            self.runner.write_final_report()


if __name__ == "__main__":
    unittest.main()
