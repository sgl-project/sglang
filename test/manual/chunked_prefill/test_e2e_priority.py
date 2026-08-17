import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestBase
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestChunkedFeaturePriority(ChunkedTestBase):
    __test__ = True
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    gsm8k_threshold = 0.30
    feature_args = [
        "--enable-priority-scheduling",
        "--max-running-requests",
        "8",
        "--max-queued-requests",
        "128",
    ]


if __name__ == "__main__":
    unittest.main()
