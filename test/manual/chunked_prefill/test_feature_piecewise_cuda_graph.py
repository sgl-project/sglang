import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedRefactorTestBase
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestChunkedFeaturePiecewiseCudaGraph(ChunkedRefactorTestBase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    # Empty feature_args is intentional: piecewise CG is the default, so the
    # whole point of this fixture is "default flags + small chunk_size".
    feature_args = []


if __name__ == "__main__":
    unittest.main()
