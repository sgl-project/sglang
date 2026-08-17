import unittest

from sglang.test.chunked_prefill_test_utils import (
    LONG_PROMPT_NUM_SHOTS,
    ChunkedTestBase,
)


class TestChunkedFeatureHybridSWA(ChunkedTestBase):
    __test__ = True
    model = "openai/gpt-oss-20b"
    num_shots = LONG_PROMPT_NUM_SHOTS
    gsm8k_threshold = 0.50
    feature_args = [
        "--mem-fraction-static",
        "0.70",
        "--disable-piecewise-cuda-graph",
    ]


if __name__ == "__main__":
    unittest.main()
