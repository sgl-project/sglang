"""MI35x nightly performance benchmark for Kimi-K3 (8-GPU).

Benchmarks moonshotai/Kimi-K3 at TP8 on MI35x using the same non-speculative
Day-0 recipe as the accuracy test (sgl-project/sglang#32548), so the two are
directly comparable and a perf regression cannot be confused with a config
difference.

This runs as the step after the eval inside nightly-8-gpu-mi35x-kimi-k3-rocm720
rather than as a job of its own, which is how every other combined accuracy plus
performance job in that workflow is arranged. Sharing the job means the 1.56 TB
checkpoint is already in the container cache, and it takes one 8-GPU MI35x slot
instead of two on a runner scarce enough that the difference is real wall time.
Step ordering also supplies the gate for free: a failed eval fails the job, so
this never measures throughput on a build that got the tokens wrong.

#32548 reports the reference numbers to compare against: 102.5 tok/s/GPU at
concurrency 2 rising to 612.3 at concurrency 32, non-speculative.

Registry: nightly-perf-8-gpu-mi35x-kimi-k3 suite
"""

import os
import unittest

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import generate_simple_markdown_report
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

# Register for AMD CI - Kimi K3 perf benchmark on MI35x (~150 min: the 1.56 TB
# MXFP4 checkpoint dominates startup before any batch is timed)
register_amd_ci(est_time=9000, suite="nightly-perf-8-gpu-mi35x-kimi-k3", nightly=True)

KIMI_K3_MODEL_PATH = os.environ.get("KIMI_K3_MODEL_PATH", "moonshotai/Kimi-K3")
RESULT_DIR = "performance_results_kimi_k3_mi35x"
# 64 is what the ~53 GB per GPU left after the MXFP4 weights admits, which is
# also why the accuracy test caps concurrency there. The largest batch timed,
# --max-running-requests and --cuda-graph-max-bs-decode are then held equal to
# it, so capture across K3's 93 attention + 92 MoE layers is not spent on
# batches the server would never admit.
MAX_BATCH_SIZE = 64


class TestNightlyKimiK3PerformanceMI35x(unittest.TestCase):
    """Kimi-K3 TP8 throughput on AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # The leading 1 is repeated so the report helper drops it as a warmup
        # run. The perf step launches its own server, so the first request pays
        # for warmup, and batch 1 is the row that distorts most.
        cls.batch_sizes = [1, 1, 8, 16, MAX_BATCH_SIZE]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        cls.model_config = {
            "name": "default",
            "model_path": KIMI_K3_MODEL_PATH,
            "other_args": [
                "--trust-remote-code",
                "--tp",
                "8",
                "--attention-backend",
                "triton",
                "--dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.85",
                "--disable-radix-cache",
                "--cuda-graph-max-bs-decode",
                str(MAX_BATCH_SIZE),
                "--max-running-requests",
                str(MAX_BATCH_SIZE),
                "--reasoning-parser",
                "kimi_k3",
                "--tool-call-parser",
                "kimi_k3",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true}',
                "--watchdog-timeout",
                "1200",
            ],
            # See the accuracy test for what each of these does; AITER_FLYDSL_FORCE
            # is read by AITER itself rather than sglang.
            "env_vars": {
                "SGLANG_USE_AITER": "1",
                "SGLANG_AITER_K3_OPT": "1",
                "AITER_FLYDSL_FORCE": "1",
                "AITER_SITUV2_A8W4": "1",
            },
        }

        cls.runner = NightlyBenchmarkRunner(RESULT_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_result_directory()
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_kimi_k3(self):
        """Run the Kimi-K3 batch-size sweep."""
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
                timeout=9000,
            )
            results = result_tuple[0]
            success = result_tuple[1]

            if results:
                self.runner.full_report += (
                    generate_simple_markdown_report(results, default_gpu_config="MI35x")
                    + "\n"
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
