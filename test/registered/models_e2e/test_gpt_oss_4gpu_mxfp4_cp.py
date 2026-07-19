import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gpt_oss_common import BaseTestGptOss

register_cuda_ci(est_time=220, stage="base-c", runner_config="4-gpu-h100")
register_cuda_ci(est_time=220, stage="base-c", runner_config="4-gpu-b200")


class TestGptOss4GpuMxfp4CP(BaseTestGptOss):
    def test_mxfp4_120b(self):
        self.run_test(
            model_variant="120b",
            quantization="mxfp4",
            expected_score_of_reasoning_effort={
                "low": 0.58,
            },
            other_args=[
                "--tp",
                "4",
                "--enable-prefill-cp",
                "--attn-cp-size",
                "4",
                "--cp-strategy",
                "zigzag",
                "--cuda-graph-max-bs-decode",
                "200",
            ],
        )


if __name__ == "__main__":
    unittest.main()
