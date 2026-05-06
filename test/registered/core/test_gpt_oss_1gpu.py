import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.gpt_oss_common import BaseTestGptOss

register_cuda_ci(est_time=519, suite="nightly-1-gpu", nightly=True)
register_amd_ci(est_time=750, suite="stage-b-test-1-gpu-small-amd-mi35x")


class TestGptOss1Gpu(BaseTestGptOss):
    def test_mxfp4_20b(self):
        self.run_test(
            model_variant="20b",
            quantization="mxfp4",
            expected_score_of_reasoning_effort={
                "low": 0.34,
                "medium": 0.34,
                "high": 0.27,  # TODO investigate
            },
        )

    def test_bf16_20b(self):
        self.run_test(
            model_variant="20b",
            quantization="bf16",
            expected_score_of_reasoning_effort={
                "low": 0.34,
                "medium": 0.34,
                "high": 0.27,  # TODO investigate
            },
        )


if __name__ == "__main__":
    unittest.main()
