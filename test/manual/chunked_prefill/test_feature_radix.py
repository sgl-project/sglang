import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedRefactorTestBase
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestChunkedFeatureRadix(ChunkedRefactorTestBase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    feature_args = [
        "--max-total-tokens",
        "20000",
        "--schedule-policy",
        "fcfs",
    ]


if __name__ == "__main__":
    unittest.main()
