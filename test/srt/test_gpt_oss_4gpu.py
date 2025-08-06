import unittest

from test_gpt_oss_common import BaseTestGptOss


class TestGptOss4Gpu(BaseTestGptOss):
    def test_bf16_120b(self):
        self.run_test(
            model_variant="120b",
            quantization="bf16",
            # TODO
            expected_score_of_reasoning_effort={
                "low": 0.50,
                "medium": 0.50,
                "high": 0.50,
            },
        )


if __name__ == "__main__":
    unittest.main()
