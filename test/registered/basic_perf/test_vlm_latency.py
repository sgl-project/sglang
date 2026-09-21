"""Latency of the VLM serving path on one large GPU."""

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.perf_bench_kit import at_most, check_perf
from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    run_bench_serving,
)

register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-large-amd")


class TestVLMLatency(CustomTestCase):
    def test_vlm_online_latency(self):
        res = run_bench_serving(
            model=DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            num_prompts=250,
            request_rate=1,
            other_server_args=[
                "--mem-fraction-static",
                "0.7",
            ],
            dataset_name="mmmu",
        )

        check_perf(
            self,
            "test_vlm_online_latency",
            at_most(
                "median_e2e_latency_ms",
                res["median_e2e_latency_ms"],
                16500,
                unit="ms",
            ),
            at_most("median_ttft_ms", res["median_ttft_ms"], 100, amd=150, unit="ms"),
            at_most("median_itl_ms", res["median_itl_ms"], 8, unit="ms"),
        )


if __name__ == "__main__":
    unittest.main()
