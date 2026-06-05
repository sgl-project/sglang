import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestBase
from sglang.test.test_utils import DEFAULT_DRAFT_MODEL_EAGLE, DEFAULT_TARGET_MODEL_EAGLE


class TestChunkedFeatureSpec(ChunkedTestBase):
    __test__ = True  # re-enable: the shared base sets __test__ = False
    model = DEFAULT_TARGET_MODEL_EAGLE
    gsm8k_threshold = 0.50
    feature_args = [
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-draft-model-path",
        DEFAULT_DRAFT_MODEL_EAGLE,
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "8",
        "--speculative-num-draft-tokens",
        "64",
        "--mem-fraction-static",
        "0.7",
    ]


if __name__ == "__main__":
    unittest.main()
