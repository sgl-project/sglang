"""MI35x nightly performance benchmark for GPT-OSS (8-GPU).

Benchmarks the MXFP4 GPT-OSS checkpoints (openai/gpt-oss-20b,
openai/gpt-oss-120b) with the same TP8 AITER server configuration the MI35x
GPT-OSS accuracy test uses, so a throughput change here points at the
serving stack rather than at a different recipe.

Registry: nightly-perf-8-gpu-mi35x-gpt-oss suite

Example usage:
    python3 test_gpt_oss_perf_mi35x.py
"""

import os
import unittest

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import generate_simple_markdown_report
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

# Register for AMD CI - MI35x GPT-OSS perf benchmark (~60 min for both sizes)
register_amd_ci(est_time=3600, suite="nightly-perf-8-gpu-mi35x-gpt-oss", nightly=True)

RESULT_DIR = "performance_results_gpt_oss_mi35x"

# MI35x serves the MXFP4 checkpoints directly; MI30x uses the bf16 conversions.
GPT_OSS_20B_MODEL_PATH = os.environ.get("GPT_OSS_20B_MODEL_PATH", "openai/gpt-oss-20b")
GPT_OSS_120B_MODEL_PATH = os.environ.get(
    "GPT_OSS_120B_MODEL_PATH", "openai/gpt-oss-120b"
)

# Matches test/registered/amd/accuracy/mi35x/test_gpt_oss_eval_mi35x.py.
SERVER_ARGS = [
    "--trust-remote-code",
    "--tp",
    "8",
    "--attention-backend",
    "triton",
    "--chunked-prefill-size",
    "130172",
    "--max-running-requests",
    "128",
    "--mem-fraction-static",
    "0.85",
]

# AITER's MXFP4 fused MoE for gpt-oss uses the separated gate/up tile layout;
# other AITER MXFP4 callers default to interleave, so opt out explicitly.
ENV_VARS = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_MOE_GU_ITLV": "1",
}


class TestNightlyGptOssPerformanceMI35x(unittest.TestCase):
    """MI35x nightly performance benchmark for the MXFP4 GPT-OSS models."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # The leading duplicate is a warmup: this benchmark launches its own
        # server, so nothing else has paid for JIT and autotuning first.
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))
        cls.models = [GPT_OSS_20B_MODEL_PATH, GPT_OSS_120B_MODEL_PATH]

        cls.runner = NightlyBenchmarkRunner(RESULT_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_result_directory()
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_one_batch(self):
        """Benchmark every GPT-OSS size."""
        failures = []
        env = os.environ.copy()
        env.update(ENV_VARS)

        try:
            for model_path in self.models:
                with self.subTest(model=model_path):
                    results, success, _ = self.runner.run_benchmark_for_model(
                        model_path=model_path,
                        batch_sizes=self.batch_sizes,
                        input_lens=self.input_lens,
                        output_lens=self.output_lens,
                        other_args=SERVER_ARGS,
                        extra_bench_args=["--trust-remote-code"],
                        env=env,
                    )

                    if results:
                        self.runner.full_report += (
                            generate_simple_markdown_report(results, "MI35x") + "\n"
                        )

                    if not success:
                        failures.append(f"benchmark failed for {model_path}")
        finally:
            self.runner.write_final_report()

        if failures:
            self.fail("\n".join(failures))


if __name__ == "__main__":
    unittest.main()
