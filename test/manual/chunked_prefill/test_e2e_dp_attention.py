import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestBase
from sglang.test.test_utils import DEFAULT_MLA_MODEL_NAME_FOR_TEST


class TestChunkedFeatureDPAttention(ChunkedTestBase):
    model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
    feature_args = [
        "--trust-remote-code",
        "--tp",
        "2",
        "--enable-dp-attention",
        "--dp",
        "2",
    ]


if __name__ == "__main__":
    unittest.main()
