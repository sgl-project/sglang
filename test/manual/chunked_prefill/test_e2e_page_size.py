import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestBase
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestChunkedFeaturePageSize(ChunkedTestBase):
    __test__ = True  # re-enable: the shared base sets __test__ = False
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    gsm8k_threshold = 0.50
    feature_args = [
        "--page-size",
        "16",
    ]


if __name__ == "__main__":
    unittest.main()
