import sys
import unittest
from pathlib import Path

# Add test/srt to path for importing test_gpt_oss_common
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "srt"))

from test_gpt_oss_common import BaseTestGptOss

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=402, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=750, suite="stage-b-test-small-1-gpu-amd")


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
