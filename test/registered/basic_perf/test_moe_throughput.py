"""Throughput of the MoE model on two GPUs, batched and at batch size one."""

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.perf_bench_kit import at_least, check_perf
from sglang.test.test_utils import (
    DEFAULT_MOE_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    run_bench_offline_throughput,
    run_bench_serving,
)

register_cuda_ci(est_time=290, stage="extra-a", runner_config="2-gpu-large")
register_amd_ci(est_time=770, suite="stage-b-test-2-gpu-large-amd")


class TestMoEThroughput(CustomTestCase):
    def test_moe_offline_throughput_default(self):
        res = run_bench_serving(
            model=DEFAULT_MOE_MODEL_NAME_FOR_TEST,
            num_prompts=300,
            request_rate=float("inf"),
            other_server_args=["--tp", "2"],
        )

        check_perf(
            self,
            at_least(
                "output_throughput",
                res["output_throughput"],
                2660,
                amd=2100,
                unit="token/s",
            ),
        )

    def test_moe_tp2_bs1(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MOE_MODEL_NAME_FOR_TEST,
            ["--tp", "2", "--cuda-graph-max-bs-decode", "2"],
        )

        check_perf(
            self,
            at_least(
                "output_throughput", output_throughput, 139, amd=85, unit="token/s"
            ),
        )


if __name__ == "__main__":
    unittest.main()
