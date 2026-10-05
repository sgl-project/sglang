"""VLM serving perf on the aiter attention backend."""

import unittest

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.vlm_perf_kit import check_vlm_serving_perf
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd")


class TestVLMServingAiter(CustomTestCase):
    def test_vlm_serving_aiter(self):
        check_vlm_serving_perf(
            self,
            "aiter",
            output_throughput=2000,
            e2e_ms=16500,
            ttft_ms=150,
            itl_ms=8,
        )


if __name__ == "__main__":
    unittest.main()
