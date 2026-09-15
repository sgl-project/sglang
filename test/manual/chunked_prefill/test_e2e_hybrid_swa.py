import unittest

from sglang.test.chunked_prefill_test_utils import (
    ChunkedTestBase,
)


class TestChunkedFeatureHybridSWA(ChunkedTestBase):
    __test__ = True
    model = "openai/gpt-oss-20b"
    prefix_repetitions = 768
    gsm8k_threshold = 0.50
    feature_args = [
        "--mem-fraction-static",
        "0.70",
        "--cuda-graph-backend-prefill=disabled",
    ]


if __name__ == "__main__":
    unittest.main()
