import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestBase
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestChunkedFeaturePiecewiseCudaGraph(ChunkedTestBase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    gsm8k_threshold = 0.50
    # Empty feature_args is intentional: piecewise CG is the default, so the
    # whole point of this fixture is "default flags + small chunk_size".
    feature_args = []


if __name__ == "__main__":
    unittest.main()
