import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedRefactorTestBase
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST


class TestChunkedFeaturePP(ChunkedRefactorTestBase):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    feature_args = [
        "--tp-size",
        "2",
        "--pp-size",
        "2",
        "--enable-dynamic-chunking",
    ]


if __name__ == "__main__":
    unittest.main()
