"""VLM serving perf on the fa3 attention backend."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.vlm_perf_kit import check_vlm_serving_perf
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=150, stage="extra-a", runner_config="1-gpu-large")


class TestVLMServingFA3(CustomTestCase):
    def test_vlm_serving_fa3(self):
        check_vlm_serving_perf(
            self,
            "test_vlm_serving_fa3",
            "fa3",
            # Offline throughput on this lane has never been measured; the number is
            # reported so a bound can be set from the first run.
            e2e_ms=16500,
            ttft_ms=100,
            itl_ms=8,
        )


if __name__ == "__main__":
    unittest.main()
