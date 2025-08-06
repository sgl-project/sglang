import unittest

from test_gpt_oss_common import BaseTestGptOss


class TestGptOss1Gpu(BaseTestGptOss):
    def test_mxfp4_20b(self):
        self.run_test(
            model_variant="20b",
            quantization="mxfp4",
            # TODO
            expected_score=0.50,
        )


if __name__ == "__main__":
    unittest.main()
