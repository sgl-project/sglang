import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gpt_oss_common import BaseTestGptOss

register_cuda_ci(est_time=500, suite="stage-b-test-small-1-gpu")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
class TestGptOssSm120(BaseTestGptOss):
    @classmethod
    def setUpClass(cls):
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability != (12, 0):
            raise unittest.SkipTest(
                f"GPT-OSS SM120 test requires SM 12.0, but found {compute_capability[0]}.{compute_capability[1]}"
            )

    def test_mxfp4_20b(self):
        self.run_test(
            model_variant="20b",
            quantization="mxfp4",
            expected_score_of_reasoning_effort={
                "low": 0.34,
                "medium": 0.34,
                "high": 0.27,
            },
        )


if __name__ == "__main__":
    unittest.main()
