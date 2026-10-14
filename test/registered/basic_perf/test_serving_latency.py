"""Latency of the default serving path on one large GPU."""

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.perf_bench_kit import at_most, check_perf
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    run_bench_serving,
)

register_cuda_ci(est_time=190, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=165, suite="stage-b-test-1-gpu-large-amd")


class TestServingLatency(CustomTestCase):
    def test_online_latency_default(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=100,
            request_rate=1,
            other_server_args=[],
        )

        check_perf(
            self,
            at_most(
                "median_e2e_latency_ms",
                res["median_e2e_latency_ms"],
                9140,
                unit="ms",
            ),
            at_most("median_ttft_ms", res["median_ttft_ms"], 84, amd=115, unit="ms"),
            at_most("median_itl_ms", res["median_itl_ms"], 9, unit="ms"),
        )


if __name__ == "__main__":
    unittest.main()
