import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestBase


class TestChunkedFeatureSpec(ChunkedTestBase):
    __test__ = True
    use_kv_canary = False
    model = "Qwen/Qwen3-8B"
    gsm8k_threshold = 0.50
    feature_args = [
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        "Tengyunw/qwen3_8b_eagle3",
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
