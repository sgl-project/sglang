import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestBase
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestChunkedFeaturePiecewiseCudaGraph(ChunkedTestBase):
    __test__ = True
    use_kv_canary = False
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    gsm8k_threshold = 0.30
    feature_args = []


if __name__ == "__main__":
    unittest.main()
