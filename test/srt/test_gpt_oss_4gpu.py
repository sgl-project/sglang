import unittest

from test_gpt_oss_common import BaseTestGptOss


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
                "--mem-fraction-static",
                "0.93",
            ],
        )


if __name__ == "__main__":
    unittest.main()
