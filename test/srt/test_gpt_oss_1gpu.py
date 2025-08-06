import unittest

from test_gpt_oss_common import BaseTestGptOss


class TestGptOss1Gpu(BaseTestGptOss):
    def test_mxfp4_20b(self):
        self.run_test(
            model=TODO,

        )


if __name__ == "__main__":
    unittest.main()
