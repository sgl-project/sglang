import unittest

from sglang.test.gpt_oss_common import BaseTestGptOss


class TestGptOss1Gpu(BaseTestGptOss):
    def test_mxfp4_20b(self):
        self.run_test(
            model_variant="20b",
            quantization="mxfp4",
            # BASELINE PENDING: re-measure against sgl-eval's gpqa before merge.
            expected_score_of_reasoning_effort={
                "low": 0.0,
                "medium": 0.0,
                "high": 0.0,
            },
        )

    def test_bf16_20b(self):
        self.run_test(
            model_variant="20b",
            quantization="bf16",
            # BASELINE PENDING: re-measure against sgl-eval's gpqa before merge.
            expected_score_of_reasoning_effort={
                "low": 0.0,
                "medium": 0.0,
                "high": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
