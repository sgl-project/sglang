import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedRefactorTestBase


class TestChunkedFeaturePP(ChunkedRefactorTestBase):
    feature_args = [
        "--tp-size",
        "2",
        "--pp-size",
        "2",
        "--enable-dynamic-chunking",
    ]


if __name__ == "__main__":
    unittest.main()
