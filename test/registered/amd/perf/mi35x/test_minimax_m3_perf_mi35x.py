"""MI35x nightly performance benchmark for MiniMax-M3-MXFP8 (4-GPU, TP=4).

Benchmarks MiniMaxAI/MiniMax-M3-MXFP8 with the same TP=4 recipe the MI35x
accuracy test validated (aiter attention, fp8 e4m3 KV, block-fp8 linear path,
quick INT4 all-reduce), so a throughput regression cannot be confused with a
configuration difference. MI35x (gfx950 / CDNA4) has hardware MX-scaled matmul,
so the MXFP8 MoE weights are served natively.

This runs as the step after the eval inside nightly-4-gpu-mi35x-minimax-m3-rocm720
rather than as a job of its own, which is how the other combined accuracy plus
performance jobs in that workflow are arranged. Sharing the job reuses the
already-cached checkpoint and one MI35x runner slot, and step ordering supplies
the gate for free: a failed eval fails the job, so this never measures
throughput on a build that got the tokens wrong.

1K-input / 1K-output is the workload shape the cookbook publishes for MI355X, and
it weights decode heavily, which is what matters for a reasoning model that
spends most of its tokens inside `<mm:think>`. That published row is a tp8
bench_serving run (~1678 output tok/s at concurrency 64) while this sweep is tp4
under bench_one_batch_server, so treat it as a shape reference, not a target.

Registry: nightly-perf-4-gpu-mi35x-minimax-m3 suite
"""

import os
import unittest

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import generate_simple_markdown_report
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

register_amd_ci(
    est_time=5400,
    suite="nightly-perf-4-gpu-mi35x-minimax-m3",
    nightly=True,
)

MINIMAX_M3_MODEL_PATH = os.environ.get(
    "MINIMAX_M3_MODEL_PATH", "MiniMaxAI/MiniMax-M3-MXFP8"
)
RESULT_DIR = "performance_results_minimax_m3_mi35x"
MAX_BATCH_SIZE = 64


class TestNightlyMiniMaxM3PerformanceMI35x(unittest.TestCase):
    """MiniMax-M3-MXFP8 TP=4 throughput on AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # bench_one_batch_server already warms every unique batch size before it
        # times anything, but only at output_len 16. The repeated leading 1 buys
        # one throwaway full-length decode -- the report helper drops it -- so
        # whatever is first touched over 1024 output tokens lands there instead
        # of in a published row.
        cls.batch_sizes = [1, 1, 8, 16, MAX_BATCH_SIZE]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "1024"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "1024"))

        cls.model_config = {
            "name": "TP4+MXFP8+aiterAttn+fp8KV+blockFP8+quickAR",
            "model_path": MINIMAX_M3_MODEL_PATH,
            # Mirrors test_minimax_m3_tp4_eval_mi35x.py, plus multithread weight
            # load and a decode graph capped at the largest timed batch (capture
            # is not spent on batches this sweep never sends).
            "other_args": [
                "--quantization",
                "mxfp8",
                "--dtype",
                "bfloat16",
                "--trust-remote-code",
                "--tp",
                "4",
                "--attention-backend",
                "aiter",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "8192",
                "--mem-fraction-static",
                "0.80",
                "--cuda-graph-max-bs-decode",
                str(MAX_BATCH_SIZE),
                "--max-running-requests",
                str(MAX_BATCH_SIZE),
                "--model-loader-extra-config",
                '{"enable_multithread_load": true}',
                "--watchdog-timeout",
                "1200",
            ],
            # See the accuracy test for what each of these does: the block-fp8
            # linear path (#32036) and custom/quick INT4 all-reduce (#32230) are
            # both opt-in on gfx950, and the fp32 router GEMM is what ROCm 7.0's
            # rocBLAS accepts.
            "env_vars": {
                "SGLANG_USE_AITER": "1",
                "SGLANG_OPT_USE_BF16_ROUTER_GEMM": "0",
                "SGLANG_FORCE_MXFP8_BLOCK_CONVERT": "1",
                "SGLANG_M3_ALLOW_CUSTOM_AR": "1",
                "ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
                "ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": "1",
            },
        }

        cls.runner = NightlyBenchmarkRunner(RESULT_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_result_directory()
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_minimax_m3(self):
        """Run the MiniMax-M3-MXFP8 batch-size sweep."""
        old_env = {}
        for key, value in self.model_config["env_vars"].items():
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
                timeout=5400,
            )
            results, success = result_tuple[0], result_tuple[1]

            if results:
                self.runner.full_report += (
                    generate_simple_markdown_report(results, default_gpu_config="MI35x")
                    + "\n"
                )

            self.assertTrue(
                success, f"Benchmark failed for {MINIMAX_M3_MODEL_PATH} on MI35x"
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
