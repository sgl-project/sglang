import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestBase
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestChunkedFeatureRadix(ChunkedTestBase):
    __test__ = True
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    gsm8k_threshold = 0.30
    feature_args = [
        "--max-total-tokens",
        "20000",
        "--schedule-policy",
        "fcfs",
    ]


if __name__ == "__main__":
    unittest.main()
