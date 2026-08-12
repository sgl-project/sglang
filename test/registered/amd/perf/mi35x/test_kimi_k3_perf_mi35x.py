"""MI35x nightly performance benchmark for Kimi-K3 (8-GPU).

Benchmarks moonshotai/Kimi-K3 with the single-node TP8 ROCm/AITER recipe from
the Kimi-K3 cookbook page (AITER A8W4 FlyDSL MoE, Triton attention, fp8 KV
cache), which is the MI350X/MI355X serving shape.

Registry: nightly-perf-8-gpu-mi35x-kimi-k3 suite

Example usage:
    KIMI_K3_MODEL_PATH=moonshotai/Kimi-K3 python3 test_kimi_k3_perf_mi35x.py
"""

import os
import unittest

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import (
    drop_warmup_result,
    generate_simple_markdown_report,
)
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.perf_baseline import check_output_throughput
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

# Register for AMD CI - MI35x Kimi-K3 perf benchmark (~3h including weight load)
register_amd_ci(est_time=10800, suite="nightly-perf-8-gpu-mi35x-kimi-k3", nightly=True)

RESULT_DIR = "performance_results_kimi_k3_mi35x"

KIMI_K3_MODEL_PATH = os.environ.get("KIMI_K3_MODEL_PATH", "moonshotai/Kimi-K3")
# K3 is 2.8T; weight load and warmup dominate startup.
SERVER_LAUNCH_TIMEOUT = 7200

ENV_VARS = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_AITER_K3_OPT": "1",
    "AITER_FLYDSL_FORCE": "1",
    "AITER_SITUV2_A8W4": "1",
}

# K3 splits static memory into a KDA state pool, which caps concurrency, and a
# paged MLA KV pool. --mamba-full-memory-ratio is that split, and its balance
# point moves with request length, so derive it from the lengths this run
# benchmarks rather than pinning a value that only holds at one shape. The two
# constants are K3's fixed geometry at attention-TP 8 with fp32 SSM state and
# fp8 KV (one state slot is 56.2 MB, one KV token 13.8 KB), and the 5 state
# slots per running request that the default extra_buffer cache strategy holds
# with the overlap scheduler on. Same formula as the cookbook's calculator in
# docs/src/snippets/_kimi_k3_mamba_ratio_calculator.jsx.
STATE_SLOT_IN_KV_TOKENS = 4063
STATE_SLOTS_PER_REQUEST = 5


def mamba_full_memory_ratio(input_len: int, output_len: int) -> float:
    return round(
        STATE_SLOTS_PER_REQUEST * STATE_SLOT_IN_KV_TOKENS / (input_len + output_len), 1
    )


# No baseline recorded yet: the numbers this job produces over its first
# nightly runs become a ThroughputBaseline here, the same way the DeepSeek-V4
# tests carry theirs.
PERF_BASELINE = None


class TestNightlyKimiK3PerformanceMI35x(unittest.TestCase):
    """MI35x nightly performance benchmark for Kimi-K3."""

    @classmethod
    def setUpClass(cls):
        cls.model = KIMI_K3_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        # The leading duplicate is a warmup: this job launches its own server,
        # so nothing else has paid for JIT and autotuning first.
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        cls.server_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--attention-backend",
            "triton",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.85",
            "--cuda-graph-max-bs",
            "256",
            "--mamba-full-memory-ratio",
            str(mamba_full_memory_ratio(cls.input_lens[0], cls.output_lens[0])),
            "--reasoning-parser",
            "kimi_k3",
            "--tool-call-parser",
            "kimi_k3",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--watchdog-timeout",
            "1200",
        ]

        cls.runner = NightlyBenchmarkRunner(RESULT_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_result_directory()
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_one_batch(self):
        """Benchmark Kimi-K3, then gate it on its baseline."""
        env = os.environ.copy()
        env.update(ENV_VARS)

        try:
            results, success, _ = self.runner.run_benchmark_for_model(
                model_path=self.model,
                batch_sizes=self.batch_sizes,
                input_lens=self.input_lens,
                output_lens=self.output_lens,
                other_args=self.server_args,
                extra_bench_args=["--trust-remote-code"],
                timeout=SERVER_LAUNCH_TIMEOUT,
                env=env,
            )

            if results:
                self.runner.full_report += (
                    generate_simple_markdown_report(results, "MI35x") + "\n"
                )

            self.assertTrue(success, f"Benchmark failed for {self.model}")

            check = check_output_throughput(
                drop_warmup_result(results), PERF_BASELINE, f"{self.model} [MI35x]"
            )
            self.runner.full_report += check.markdown + "\n"
            if not check.ok:
                self.fail(check.failure_message())
        finally:
            self.runner.write_final_report()


if __name__ == "__main__":
    unittest.main()
