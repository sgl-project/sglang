import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestPDBase


class TestChunkedFeatureDisagg(ChunkedTestPDBase):
    __test__ = True  # re-enable: the shared base sets __test__ = False
    gsm8k_threshold = 0.50


if __name__ == "__main__":
    unittest.main()
