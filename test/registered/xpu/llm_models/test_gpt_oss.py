import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.gpt_oss_common import BaseTestGptOss

register_xpu_ci(est_time=2400, suite="nightly-xpu-8-gpu", nightly=True)


@unittest.skipIf(not torch.xpu.is_available(), "XPU is not available")
class TestGptOssXPU(BaseTestGptOss):
    def test_mxfp4_20b_bf16(self):
        self.run_test(
            model_variant="20b",
            quantization="mxfp4",
            expected_score_of_reasoning_effort={
                "low": 0.34,
                "medium": 0.34,
                "high": 0.27,
            },
            other_args=[
                "--tp",
                "8",
                "--moe-runner-backend",
                "triton",
                "--mem-fraction-static",
                "0.8",
            ],
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
            other_args=[
                "--tp",
                "8",
                "--moe-runner-backend",
                "triton_kernel",
                "--mem-fraction-static",
                "0.8",
            ],
        )


if __name__ == "__main__":
    unittest.main()
