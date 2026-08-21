import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestBase


class TestChunkedFeaturePP(ChunkedTestBase):
    __test__ = True
    gsm8k_threshold = 0.50
    feature_args = [
        "--tp-size",
        "2",
        "--pp-size",
        "2",
        "--enable-dynamic-chunking",
    ]


if __name__ == "__main__":
    unittest.main()
