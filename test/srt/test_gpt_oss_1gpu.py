import unittest

from test_gpt_oss_common import BaseTestGptOss


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
