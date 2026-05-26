import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedRefactorTestBase
from sglang.test.test_utils import DEFAULT_MLA_MODEL_NAME_FOR_TEST


class TestChunkedFeatureDPAttention(ChunkedRefactorTestBase):
    model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
    feature_args = [
        "--trust-remote-code",
        "--tp",
        "2",
        "--enable-dp-attention",
        "--dp",
        "2",
        "--enable-mixed-chunk",
    ]


if __name__ == "__main__":
    unittest.main()
