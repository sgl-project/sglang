import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.gpt_oss_common import BaseTestGptOss

# TODO: re-enable once GPQA fits in memory at tp=4, or an 8-GPU
# XPU nightly suite exists; then drop `disabled=` and match --tp to the runner.
register_xpu_ci(
    est_time=2400,
    suite="nightly-xpu-kernel-main-4-gpu",
    nightly=True,
    disabled="GPQA eval OOMs on the XPU nightly runners; needs 8 cards, no 8-GPU XPU suite",
)


@unittest.skipIf(not torch.xpu.is_available(), "XPU is not available")
@unittest.skipIf(torch.xpu.device_count() < 8, "requires 8 cards")
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

    def test_mxfp4_20b_moe_intel_xpu(self):
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
                "intel_xpu",
                "--mem-fraction-static",
                "0.8",
            ],
        )


if __name__ == "__main__":
    unittest.main()
