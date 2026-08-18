"""MI35x nightly performance benchmark for GLM-5.2-FP8 (8-GPU).

Benchmarks zai-org/GLM-5.2-FP8 at TP8 with the cookbook's MI355X / FP8 /
low-latency / single-node server configuration. The 8K-input / 1K-output
batch-1 and batch-16 workloads match the published cookbook cells, making the
nightly report useful for detecting drift from the recipe it protects.

This runs after the GLM-5.2 accuracy test in the same ROCm 7.2 job. The
checkpoint is therefore already cached, accuracy gates performance, and the
two tests consume one scarce 8-GPU MI35x slot rather than two.

Registry: nightly-perf-8-gpu-mi35x-glm52-fp8 suite
"""

import os
import unittest

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import generate_simple_markdown_report
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

register_amd_ci(
    est_time=5400,
    suite="nightly-perf-8-gpu-mi35x-glm52-fp8",
    nightly=True,
)

GLM_52_FP8_MODEL_PATH = os.environ.get("GLM52_FP8_MODEL_PATH", "zai-org/GLM-5.2-FP8")
RESULT_DIR = "performance_results_glm52_fp8_mi35x"


class TestGLM52FP8PerfMI35x(unittest.TestCase):
    """GLM-5.2-FP8 low-latency throughput on AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Repeat batch 1 as an unreported warmup, then measure the two
        # concurrency points published for the low-latency cookbook recipe.
        cls.batch_sizes = [1, 1, 16]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "8192"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "1024"))
        cls.model_config = {
            "name": "low-latency-tp8",
            "model_path": GLM_52_FP8_MODEL_PATH,
            "other_args": [
                "--trust-remote-code",
                "--reasoning-parser",
                "glm45",
                "--tool-call-parser",
                "glm47",
                "--tp",
                "8",
                "--dsa-prefill-backend",
                "tilelang",
                "--dsa-decode-backend",
                "tilelang",
                "--chunked-prefill-size",
                "131072",
                "--mem-fraction-static",
                "0.80",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true}',
                "--watchdog-timeout",
                "1200",
            ],
        }

        os.environ.setdefault("SGLANG_BENCH_TIMEOUT", "3600")
        cls.runner = NightlyBenchmarkRunner(RESULT_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_result_directory()
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_glm52_fp8_perf(self):
        """Run the GLM-5.2-FP8 batch-size sweep."""
        try:
            result_tuple = self.runner.run_benchmark_for_model(
                model_path=self.model_config["model_path"],
                batch_sizes=self.batch_sizes,
                input_lens=self.input_lens,
                output_lens=self.output_lens,
                other_args=self.model_config["other_args"],
                variant=self.model_config["name"],
                extra_bench_args=["--trust-remote-code"],
                timeout=5400,
            )
            results, success = result_tuple[0], result_tuple[1]

            if results:
                self.runner.full_report += (
                    generate_simple_markdown_report(results, default_gpu_config="MI35x")
                    + "\n"
                )

            self.assertTrue(
                success, f"Benchmark failed for {GLM_52_FP8_MODEL_PATH} on MI35x"
            )
        finally:
            self.runner.write_final_report()


if __name__ == "__main__":
    unittest.main()
