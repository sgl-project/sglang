import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestPDBase


class TestChunkedFeaturePDPP(ChunkedTestPDBase):
    __test__ = True
    gsm8k_threshold = 0.50
    feature_args = [
        "--tp-size",
        "2",
        "--pp-size",
        "2",
        "--disable-overlap-schedule",
    ]
    decode_feature_args = [
        "--tp-size",
        "2",
        "--base-gpu-id",
        "4",
    ]


if __name__ == "__main__":
    unittest.main()
