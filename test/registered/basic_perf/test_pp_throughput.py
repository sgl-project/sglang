"""Throughput of pipeline parallelism on two GPUs, decode and long prefill."""

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.perf_bench_kit import at_least, check_perf
from sglang.test.test_utils import (
    DEFAULT_MOE_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    run_bench_serving,
)

register_cuda_ci(est_time=490, stage="extra-a", runner_config="2-gpu-large")
register_amd_ci(est_time=1030, suite="stage-b-test-2-gpu-large-amd")


class TestPPThroughput(CustomTestCase):
    def test_pp_offline_throughput_default_decode(self):
        res = run_bench_serving(
            model=DEFAULT_MOE_MODEL_NAME_FOR_TEST,
            num_prompts=1000,
            request_rate=float("inf"),
            random_input_len=1,
            random_output_len=1024,
            other_server_args=["--pp-size", "2"],
            need_warmup=True,
            seed=42,
        )

        check_perf(
            self,
            at_least(
                "output_throughput", res["output_throughput"], 6250, unit="token/s"
            ),
        )

    def test_pp_long_context_prefill(self):
        res = run_bench_serving(
            model="meta-llama/Llama-3.3-70B-Instruct",
            num_prompts=4,
            request_rate=float("inf"),
            random_input_len=128000,
            random_output_len=1,
            dataset_name="random",
            other_server_args=[
                "--quantization",
                "fp8",
                "--pp-size",
                "2",
            ]
            + (["--mem-fraction-static", "0.7"] if is_in_amd_ci() else []),
            need_warmup=False,
            seed=42,
        )

        check_perf(
            self,
            at_least(
                "input_throughput",
                res["input_throughput"],
                4380,
                amd=2190,
                unit="token/s",
            ),
        )


if __name__ == "__main__":
    unittest.main()
