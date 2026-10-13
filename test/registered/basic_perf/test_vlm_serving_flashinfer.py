"""VLM serving perf on the flashinfer attention backend."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.vlm_perf_kit import check_vlm_serving_perf
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=195, stage="extra-a", runner_config="1-gpu-small")


class TestVLMServingFlashinfer(CustomTestCase):
    def test_vlm_serving_flashinfer(self):
        check_vlm_serving_perf(
            self,
            "flashinfer",
            output_throughput=5940,
            e2e_ms=17480,
            ttft_ms=83,
            itl_ms=8.4,
        )


if __name__ == "__main__":
    unittest.main()
