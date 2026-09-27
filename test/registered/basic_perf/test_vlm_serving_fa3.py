"""VLM serving perf on the fa3 attention backend."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.vlm_perf_kit import check_vlm_serving_perf
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=150, stage="extra-a", runner_config="1-gpu-large")


class TestVLMServingFa3(CustomTestCase):
    def test_vlm_serving_fa3(self):
        check_vlm_serving_perf(
            self,
            "fa3",
            # No offline bound: never measured on this lane.
            output_throughput=15640,
            e2e_ms=11000,
            ttft_ms=100,
            itl_ms=5.2,
        )


if __name__ == "__main__":
    unittest.main()
