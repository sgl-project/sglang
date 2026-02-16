import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gpt_oss_common import BaseTestGptOss

register_cuda_ci(est_time=300, suite="stage-c-test-4-gpu-h100")
register_cuda_ci(est_time=300, suite="stage-c-test-4-gpu-b200")


class TestGptOss4Gpu(BaseTestGptOss):
    def test_bf16_120b(self):
        self.run_test(
            model_variant="120b",
            quantization="bf16",
            expected_score_of_reasoning_effort={
                "low": 0.60,
            },
            other_args=["--tp", "4", "--cuda-graph-max-bs", "200"],
        )

    def test_mxfp4_120b(self):
        self.run_test(
            model_variant="120b",
            quantization="mxfp4",
            expected_score_of_reasoning_effort={
                "low": 0.60,
            },
            other_args=[
                "--tp",
                "4",
                "--cuda-graph-max-bs",
                "200",
            ],
        )


if __name__ == "__main__":
    unittest.main()
