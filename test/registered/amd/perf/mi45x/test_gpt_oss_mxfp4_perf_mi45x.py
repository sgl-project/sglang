"""MI455X nightly performance benchmark for GPT-OSS-120B MXFP4 (1-GPU).

Companion to
`test/registered/amd/accuracy/mi45x/test_gpt_oss_mxfp4_eval_mi45x.py`: same
checkpoint, same server flags, but measures throughput/latency instead of
accuracy. Reports only — perf jobs are deliberately excluded from the nightly
`check-all-jobs` gate, so a regression here shows up in the step summary
without blocking CI.

Registry: nightly-perf-1-gpu-mi45x-gpt-oss suite
"""

import os
import unittest
from typing import List

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import BenchmarkResult
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

# Register for AMD CI - GPT-OSS-120B MXFP4 benchmark on MI455X (~40 min)
register_amd_ci(
    est_time=2400,
    suite="nightly-perf-1-gpu-mi45x-gpt-oss",
    nightly=True,
)

# Server-side env for the gfx1250 MXFP4 path. Kept identical to the accuracy
# test so the two measure the same configuration — if you change one, change
# both, or perf will be benchmarking a kernel path accuracy never validated.
MI45X_GPT_OSS_ENV = {
    "SGLANG_USE_AITER": "1",
    # See the accuracy test for why this is unverified on gfx1250.
    "SGLANG_USE_AITER_MOE_GU_ITLV": "1",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    "AITER_FORCE_A8W4": "1",
    "ENABLE_CK": "0",
    "HSA_COREDUMP_PATTERN": "/dev/null",
}


def generate_simple_markdown_report(results: List[BenchmarkResult]) -> str:
    """Generate a simplified markdown report without traces and cost columns.

    Skips the first result if it's a warmup run (duplicate batch_size).
    """
    model_header = results[0].model_path
    if results[0].run_name and results[0].run_name != "default":
        model_header += f" ({results[0].run_name})"

    gpu_config = os.getenv("GPU_CONFIG", "MI455X")
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


PROFILE_DIR = "performance_profiles_gpt_oss_mxfp4_mi45x"


class TestGptOssMxfp4PerfMI45x(unittest.TestCase):
    """MI455X nightly performance benchmark for GPT-OSS-120B MXFP4 at TP=1."""

    @classmethod
    def setUpClass(cls):
        cls.model = "amd/gpt-oss-120b-w-mxfp4-a-fp8"
        print(f"Using model path: {cls.model}")
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Capped at 128 to match --max-running-requests on the server side.
        cls.batch_sizes = [1, 8, 32, 128]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        cls.variants = [
            {
                "name": "mxfp4-a8w4",
                "other_args": [
                    "--trust-remote-code",
                    "--tp",
                    "1",
                    "--prefill-attention-backend",
                    "triton",
                    "--decode-attention-backend",
                    "aiter",
                    "--max-running-requests",
                    "128",
                    "--mem-fraction-static",
                    "0.9",
                    "--disable-radix-cache",
                    "--page-size",
                    "64",
                ],
            },
        ]

        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_one_batch(self):
        """Run benchmark across all configured variants."""
        failed_variants = []

        try:
            for variant_config in self.variants:
                with self.subTest(variant=variant_config["name"]):
                    env = os.environ.copy()
                    env.update(MI45X_GPT_OSS_ENV)

                    result_tuple = self.runner.run_benchmark_for_model(
                        model_path=self.model,
                        batch_sizes=self.batch_sizes,
                        input_lens=self.input_lens,
                        output_lens=self.output_lens,
                        other_args=variant_config["other_args"],
                        variant=variant_config["name"],
                        extra_bench_args=["--trust-remote-code"],
                        enable_profile=False,
                        env=env,
                    )
                    results = result_tuple[0]
                    success = result_tuple[1]

                    if not success:
                        failed_variants.append(variant_config["name"])

                    if results:
                        self.runner.full_report += (
                            generate_simple_markdown_report(results) + "\n"
                        )
        finally:
            self.runner.write_final_report()

        if failed_variants:
            raise AssertionError(
                f"Benchmark failed for {self.model} with the following variants: "
                f"{', '.join(failed_variants)}"
            )


if __name__ == "__main__":
    unittest.main()
