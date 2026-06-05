import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestBase
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestChunkedFeaturePiecewiseCudaGraph(ChunkedTestBase):
    __test__ = True  # re-enable: the shared base sets __test__ = False
    # piecewise cuda graph IS the feature under test; canary requires it disabled
    use_kv_canary = False
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    # Llama-3.2-1B-Instruct measures 0.38-0.41 on mixed_prefix_gsm8k (H200,
    # greedy); chunking corruption collapses the score to ~0, so 0.30 keeps
    # full detection power for the model this class actually runs.
    gsm8k_threshold = 0.30
    feature_args = []


if __name__ == "__main__":
    unittest.main()
